import torch
import torch.nn as nn
from .layers import RMSNorm, apply_rope

### TO REVIEW, AI GENERATED, NOT TESTED YET ###


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention mechanism. If n_kv_groups equals n_heads, this functions
    as standard Multi-Head Attention.
    """
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False,
        query_pre_attn_scalar=None, dtype=None,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        self.q_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = RMSNorm(head_dim, eps=1e-6) if qk_norm else None
        
        self.scaling = (query_pre_attn_scalar if query_pre_attn_scalar is not None else head_dim) ** -0.5

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape for multi-head processing
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads if using GQA
        if self.group_size > 1:
            keys = keys.repeat_interleave(self.group_size, dim=1)
            values = values.repeat_interleave(self.group_size, dim=1)

        # Apply scaling to queries
        queries = queries * self.scaling

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(queries.dtype)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)

