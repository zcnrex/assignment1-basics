import torch
from einops import rearrange, einsum


class RoPE(torch.nn.Module):
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000, device=None):
        super().__init__()
        seq_idx = torch.arange(max_seq_len, dtype=torch.float, device=device)
        theta_ik = torch.pow(theta, torch.arange(0, d_k, 2, dtype=torch.float, device=device) / d_k)
        freqs = seq_idx[:, None] / theta_ik[None, :]
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.reshape(*x.shape)
        return x_rotated.to(in_dtype)
