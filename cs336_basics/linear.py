import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.randn(out_features, in_features, device=device, dtype=dtype))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
