"""Label-smoothed cross entropy loss for sequence-to-sequence training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing.

    Instead of one-hot targets, distributes (1 - smoothing) to the correct class
    and smoothing / (vocab_size - 1) to all other classes. This regularizes the
    model and prevents overconfident predictions.
    """

    def __init__(self, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch * seq_len, vocab_size)
            target: (batch * seq_len,)

        Returns:
            Scalar loss (averaged over non-pad tokens).
        """
        vocab_size = logits.size(-1)

        log_probs = F.log_softmax(logits, dim=-1)

        # Non-pad mask
        non_pad_mask = target != self.pad_idx
        n_tokens = non_pad_mask.sum().item()

        if n_tokens == 0:
            return logits.sum() * 0.0

        # NLL loss (gather the log prob for the correct token)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

        # Smooth loss (average log prob over all tokens)
        smooth_loss = -log_probs.sum(dim=-1) / vocab_size

        # Combine
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        # Mask padding and average
        loss = (loss * non_pad_mask).sum() / n_tokens
        return loss
