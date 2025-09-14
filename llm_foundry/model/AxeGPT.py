import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, FeedForward
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        # Use the new MultiHeadAttention class.
        # Note: The new attention class does not take `query_pre_attn_scalar`,
        # so we don't pass it from the config.
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"],
            dtype=torch.bfloat16 if cfg['dtype'] == 'bfloat16' else torch.float16,
        )
        self.ff = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x):
        # Pre-normalization and shortcut for attention
        shortcut = x
        x_norm = self.input_layernorm(x)
        # The call to attention is simple as it handles its own masking and RoPE
        x_attn = self.att(x_norm)
        x = shortcut + x_attn

        # Pre-normalization and shortcut for feed-forward
        shortcut = x
        x_norm = self.post_attention_layernorm(x)
        x_ffn = self.ff(x_norm)
        x = shortcut + x_ffn
        return x

class AxeGPT(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=torch.bfloat16 if cfg['dtype'] == 'bfloat16' else torch.float16)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) ## Imp, ModuleList is a special list in PyTorch that registers the layers properly in the model
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=torch.bfloat16 if cfg['dtype'] == 'bfloat16' else torch.float16)


    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.out_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):

        for next_token in range(max_new_tokens):
            context_len = self.cfg["context_length"]
            #input_ids.shape = (batch_size, seq_length)
            input_ids = input_ids if input_ids.size(1) <= context_len else input_ids[:, -context_len:] 

            logits, _ = self(input_ids) # _ is the loss, we don't need it heree
            # Focus on the last token only
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # Top K should not be greater than vocab size.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # _ are the indices we don't need

                # Smallest values for each.
                smallest = v[:, -1].unsqueeze(1)
                logits[logits < smallest] = -float('Inf')

            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)

        return input_ids

