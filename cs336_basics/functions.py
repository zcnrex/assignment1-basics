import torch
from einops import rearrange
import math
from collections.abc import Iterable


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


def log_softmax(inputs, dim=1):
    max_vals, _ = torch.max(inputs, dim=dim, keepdim=True)
    shifted = inputs - max_vals
    logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=dim, keepdim=True))
    return shifted - logsumexp


def cross_entropy(inputs, targets):
    log_probs = log_softmax(inputs, dim=1)
    loss = -log_probs[torch.arange(inputs.shape[0]), targets]
    return loss.mean()


def learning_rate_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return

    total_norm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in parameters]), 2)
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
