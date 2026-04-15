"""Complete Transformer model for sequence-to-sequence translation."""

import torch
import torch.nn as nn

from .embeddings import TransformerEmbedding
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    """Transformer encoder-decoder model.

    Supports shared embeddings between source, target, and output projection
    (weight tying), which reduces parameters and improves performance.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        share_embeddings: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len, dropout)
        self.tgt_embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len, dropout)

        self.encoder = Encoder(n_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_decoder_layers, d_model, n_heads, d_ff, dropout)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        if share_embeddings:
            # Share all three embedding weights (src, tgt, output)
            self.src_embedding.token_embedding.embedding.weight = (
                self.tgt_embedding.token_embedding.embedding.weight
            )
            self.output_proj.weight = self.tgt_embedding.token_embedding.embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        """Xavier uniform initialization for all parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Padding mask for source: (batch, 1, 1, src_len)."""
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Combined padding + causal mask for target: (batch, 1, tgt_len, tgt_len)."""
        batch_size, tgt_len = tgt.size()

        # Padding mask: (batch, 1, 1, tgt_len)
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # Causal mask: (1, 1, tgt_len, tgt_len)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        return pad_mask & causal_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embedding(src), src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(self.tgt_embedding(tgt), enc_output, tgt_mask, src_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) source token ids
            tgt: (batch, tgt_len) target token ids (teacher forcing)

        Returns:
            (batch, tgt_len, vocab_size) logits
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)

        return self.output_proj(dec_output)
