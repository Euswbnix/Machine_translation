"""Token and positional embeddings."""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learned token embedding scaled by sqrt(d_model)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self.scale


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, not learned)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """Combines token embedding + positional encoding."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding(self.token_embedding(x))
