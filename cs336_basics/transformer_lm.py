import torch
from .transformer_block import TransformerBlock
from .embedding import Embedding
from .rms_norm import RMSNorm
from .linear import Linear


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, theta=theta)
                for _ in range(num_layers)
            ]
        )
        self.token_embeddings = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)

        x = self.lm_head(self.ln_final(x))
        return x
