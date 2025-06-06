import torch
from .linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w2 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        silu = self._silu(self.w1(x))
        res = self.w2(silu * self.w3(x))
        return res.to(in_dtype)

    def _silu(self, x):
        return x * torch.sigmoid(x)
