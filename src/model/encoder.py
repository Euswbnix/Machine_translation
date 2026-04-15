"""Transformer Encoder."""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import FeedForward, ResidualConnection


class EncoderLayer(nn.Module):
    """Single encoder layer: self-attention + feed-forward, each with residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.residual2(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """Stack of N encoder layers + final layer norm."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
