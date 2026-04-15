"""Transformer learning rate schedule with warmup."""

import torch


class TransformerScheduler:
    """Implements the learning rate schedule from 'Attention Is All You Need'.

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Linearly increases LR during warmup, then decays proportionally to 1/sqrt(step).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        lr_scale: float = 1.0,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr_scale = lr_scale
        self._step = 0

    def step(self):
        """Update learning rate and step the optimizer."""
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self) -> float:
        step = max(self._step, 1)
        return (
            self.lr_scale
            * (self.d_model ** -0.5)
            * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        )

    @property
    def current_lr(self) -> float:
        return self._get_lr()

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state_dict: dict):
        self._step = state_dict["step"]
