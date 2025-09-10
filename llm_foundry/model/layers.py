import torch
import torch.nn as nn



def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """
    Precompute sine and cosine for Rotary Positional Embeddings (interleaved version).
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Compute inverse frequencies for half-dim
    freq_exponents = torch.arange(0, head_dim // 2, dtype=dtype) / head_dim
    inv_freq = 1.0 / (theta_base ** freq_exponents)

    # Positions (size: context_length)
    positions = torch.arange(context_length, dtype=dtype)

    # Outer product → (context_length, head_dim//2)
    angles_half = positions[:, None] * inv_freq[None, :]

    # Interleave [θ0, θ0, θ1, θ1, ...]
    angles = torch.zeros(context_length, head_dim, dtype=dtype)
    angles[:, 0::2] = angles_half
    angles[:, 1::2] = angles_half

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """
    Apply RoPE in the interleaved style.
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Prepare cos/sin for broadcasting
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Split into even/odd parts
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]

    # Apply 2D rotation for each pair
    x_rot_even = x_even * cos[..., 0::2] - x_odd * sin[..., 0::2]
    x_rot_odd  = x_even * sin[..., 1::2] + x_odd * cos[..., 1::2]

    # Re-interleave results
    x_out = torch.stack([x_rot_even, x_rot_odd], dim=-1).reshape(batch_size, num_heads, seq_len, head_dim)

    return x_out.to(dtype=x.dtype)

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
        ## We are usig Bias in RMSNorm, so we don't use bias in Linear layers
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=torch.bfloat16 if cfg['dtype'] == 'bfloat16' else torch.float16, bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=torch.bfloat16 if cfg['dtype'] == 'bfloat16' else torch.float16, bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=torch.bfloat16 if cfg['dtype'] == 'bfloat16' else torch.float16, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)