import torch
import torch.nn as nn
from .layers import RMSNorm, apply_rope, compute_rope_params

class MultiHeadAttention(nn.Module):
    """
    A standard Multi-Head Attention mechanism.
    """
    def __init__(
        self, d_in, num_heads, head_dim=None, qk_norm=False, dtype=None,
    ):
        super().__init__()
        self.num_heads = num_heads

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim # d_in

        # Projections for Query, Key, and Value for all heads
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        self.q_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm else None

        self.scaling = 1 / (head_dim ** 0.5)

    def forward(self, x, mask=True):
        batch_size, seq_length, d_in = x.shape

        # Apply projections
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape for multi-head processing
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Calculate cos and sin
        cos, sin = compute_rope_params(head_dim=self.head_dim, context_len=seq_length)
        cos = cos.to(x.device)
        sin = sin.to(x.device)
        
        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Apply scaling to queries
        queries = queries * self.scaling

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)
        
        if mask is True:
            # Create a causal mask
            mask = torch.triu(torch.ones((seq_length, seq_length), device=x.device), diagonal=1).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, seq_length)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(queries.dtype)

        context = (attn_weights @ values).transpose(1, 2).reshape(batch_size, seq_length, self.d_out)
        
        #Output projection
        out = self.out_proj(context)

        return out