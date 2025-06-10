import torch
from .functions import scaled_dot_product_attention
from .rope import RoPE
from .linear import Linear
from einops import rearrange


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 1000, theta: float = 10000, device=None):
        super().__init__()
        d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_k = d_k
        self.q_proj = Linear(in_features=d_model, out_features=num_heads * d_k)
        self.k_proj = Linear(in_features=d_model, out_features=num_heads * d_k)
        self.v_proj = Linear(in_features=d_model, out_features=num_heads * d_k)
        self.output_proj = Linear(in_features=num_heads * d_k, out_features=d_model)
        self.rope = RoPE(d_k=d_k, max_seq_len=max_seq_len, theta=theta, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        _, L, _ = x.shape
        device = x.device

        # Project and reshape for multi-head attention
        q = rearrange(self.q_proj(x), "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b l (h d) -> b h l d", h=self.num_heads)
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Causal mask [1, 1, seq_len, seq_len]
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        mask = rearrange(mask, "i j -> 1 1 i j")

        # Scaled dot-product attention
        attn = scaled_dot_product_attention(q, k, v, mask=mask)  # [b, h, l, d]

        # Merge heads
        out = rearrange(attn, "b h l d -> b l (h d)")

        return self.output_proj(out)
