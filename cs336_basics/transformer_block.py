import torch
from .multi_head_self_attention import MultiHeadSelfAttention
from .rms_norm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        token_positions = torch.arange(x.shape[-2], dtype=torch.int, device=x.device)
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        y = x + self.ffn(self.ln2(x))

        return y.to(in_dtype)
