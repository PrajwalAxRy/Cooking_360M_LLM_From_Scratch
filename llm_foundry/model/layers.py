import torch
import torch.nn as nn

def compute_rope_params(head_dim, context_len, theta_base=10_000, dtype=torch.float32):
    """
    Computes the Rotary Positional Embedding parameters (cosine and sine).
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_len, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    Applies Rotary Positional Embedding to the input tensor.
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # Adjust sin and cos shapes for broadcasting
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    """
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())

        if self.shift is not None:
            out = out + self.shift.float()

        return out.to(input_dtype)


class FeedForward(nn.Module):
    """
    Gated Feed-Forward Network module.
    """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=torch.bfloat16 if cfg['dtype'] == 'bfloat16' else torch.float16, bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=torch.bfloat16 if cfg['dtype'] == 'bfloat16' else torch.float16, bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=torch.bfloat16 if cfg['dtype'] == 'bfloat16' else torch.float16, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)