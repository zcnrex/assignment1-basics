import torch


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]
