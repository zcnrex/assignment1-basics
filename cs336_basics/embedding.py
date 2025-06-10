import torch


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, d_model, device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]
