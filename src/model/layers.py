"""Transformer building blocks: FeedForward, residual connection, layer norm."""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network: FFN(x) = ReLU(xW1 + b1)W2 + b2."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class ResidualConnection(nn.Module):
    """Pre-norm residual: LayerNorm(x) -> sublayer -> dropout -> + x."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))
