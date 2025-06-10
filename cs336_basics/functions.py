import torch
from einops import rearrange


def silu(x):
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1):
    max_x = x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x - max_x)
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
    d_k = Q.shape[-1]
    k_t = rearrange(K, "... keys d_k -> ... d_k keys")
    qk = torch.matmul(Q, k_t)
    scores = qk / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = softmax(scores)
    return torch.matmul(attn_weights, V)
