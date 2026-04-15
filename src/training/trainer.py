"""Main training loop for the Transformer model."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from src.model import Transformer
from src.data.tokenizer import Tokenizer, PAD_ID
from src.data.dataset import create_dataloader
from src.training.loss import LabelSmoothedCrossEntropy
from src.training.optimizer import TransformerScheduler
from src.inference.translate import beam_search_translate
from src.evaluate import compute_bleu


class Trainer:
    """Handles the full training loop with mixed precision, checkpointing, and evaluation."""

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.global_step = 0
        self.best_bleu = 0.0
        self.patience_counter = 0

        # Model
        model_cfg = config["model"]
        self.model = Transformer(
            vocab_size=model_cfg["vocab_size"],
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_encoder_layers=model_cfg["n_encoder_layers"],
            n_decoder_layers=model_cfg["n_decoder_layers"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
            max_seq_len=model_cfg["max_seq_len"],
            share_embeddings=model_cfg["share_embeddings"],
            pad_idx=PAD_ID,
        ).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")

        # Tokenizer
        data_cfg = config["data"]
        self.tokenizer = Tokenizer(data_cfg["spm_model"])

        # Loss
        train_cfg = config["training"]
        self.criterion = LabelSmoothedCrossEntropy(
            smoothing=train_cfg["label_smoothing"],
            pad_idx=PAD_ID,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0,  # will be set by scheduler
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        self.scheduler = TransformerScheduler(
            self.optimizer,
            d_model=model_cfg["d_model"],
            warmup_steps=train_cfg["warmup_steps"],
        )

        # Mixed precision
        self.use_fp16 = train_cfg.get("fp16", True) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_fp16)

        # Data loaders
        self.train_loader = create_dataloader(
            data_cfg["train_src"],
            data_cfg["train_tgt"],
            self.tokenizer,
            max_tokens_per_batch=train_cfg["batch_size"],
            max_sentences=train_cfg["max_sentences"],
            max_seq_len=model_cfg["max_seq_len"],
            shuffle=True,
            num_workers=data_cfg.get("num_workers", 4),
        )
        self.valid_loader = create_dataloader(
            data_cfg["valid_src"],
            data_cfg["valid_tgt"],
            self.tokenizer,
            max_tokens_per_batch=train_cfg["batch_size"],
            max_sentences=train_cfg["max_sentences"],
            max_seq_len=model_cfg["max_seq_len"],
            shuffle=False,
            num_workers=0,
        )

        # Checkpointing
        self.ckpt_dir = Path(config["checkpoint"]["dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.ckpt_dir / "logs"))

        # Config shortcuts
        self.max_steps = train_cfg["max_steps"]
        self.accumulate_steps = train_cfg.get("accumulate_steps", 1)
        self.clip_grad_norm = train_cfg.get("clip_grad_norm", 1.0)
        self.save_interval = train_cfg["save_interval"]
        self.eval_interval = train_cfg["eval_interval"]
        self.log_interval = train_cfg["log_interval"]
        self.patience = train_cfg.get("patience", 10)

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.max_steps} steps...")
        self.model.train()

        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()

        while self.global_step < self.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.max_steps:
                    break

                loss, n_tokens = self._train_step(batch)
                total_loss += loss * n_tokens
                total_tokens += n_tokens

                if self.global_step % self.log_interval == 0 and self.global_step > 0:
                    avg_loss = total_loss / total_tokens
                    elapsed = time.time() - start_time
                    tok_per_sec = total_tokens / elapsed
                    lr = self.scheduler.current_lr

                    print(
                        f"Step {self.global_step:>7d} | "
                        f"Loss {avg_loss:.4f} | "
                        f"LR {lr:.2e} | "
                        f"Tok/s {tok_per_sec:.0f}"
                    )
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    self.writer.add_scalar("train/tok_per_sec", tok_per_sec, self.global_step)

                    total_loss = 0.0
                    total_tokens = 0
                    start_time = time.time()

                if self.global_step % self.eval_interval == 0 and self.global_step > 0:
                    bleu = self._evaluate()
                    print(f"Step {self.global_step:>7d} | Valid BLEU: {bleu:.2f}")
                    self.writer.add_scalar("valid/bleu", bleu, self.global_step)

                    if bleu > self.best_bleu:
                        self.best_bleu = bleu
                        self.patience_counter = 0
                        self._save_checkpoint("best.pt")
                        print(f"  New best BLEU: {bleu:.2f}")
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            print(f"Early stopping at step {self.global_step}")
                            return

                    self.model.train()

                if self.global_step % self.save_interval == 0 and self.global_step > 0:
                    self._save_checkpoint(f"step_{self.global_step}.pt")
                    self._cleanup_checkpoints()

        self._save_checkpoint("final.pt")
        print(f"Training complete. Best BLEU: {self.best_bleu:.2f}")

    def _train_step(self, batch: dict) -> tuple[float, int]:
        """Single training step (forward + backward + optimizer step)."""
        src = batch["src"].to(self.device)
        tgt = batch["tgt"].to(self.device)

        # Target input (all but last token) and labels (all but first token)
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        with autocast(device_type="cuda", enabled=self.use_fp16):
            logits = self.model(src, tgt_input)
            loss = self.criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_labels.contiguous().view(-1),
            )
            loss = loss / self.accumulate_steps

        self.scaler.scale(loss).backward()

        self.global_step += 1

        if self.global_step % self.accumulate_steps == 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        n_tokens = (tgt_labels != PAD_ID).sum().item()
        return loss.item() * self.accumulate_steps, n_tokens

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate on validation set, returns BLEU score."""
        self.model.eval()

        data_cfg = self.config["data"]
        infer_cfg = self.config["inference"]

        # Read reference and source
        src_lines = Path(data_cfg["valid_src"]).read_text().strip().split("\n")
        ref_lines = Path(data_cfg["valid_tgt"]).read_text().strip().split("\n")

        # Translate
        hypotheses = beam_search_translate(
            model=self.model,
            tokenizer=self.tokenizer,
            src_sentences=src_lines,
            beam_size=infer_cfg["beam_size"],
            max_len=infer_cfg["max_decode_len"],
            length_penalty=infer_cfg["length_penalty"],
            device=self.device,
        )

        bleu = compute_bleu(hypotheses, ref_lines)
        return bleu

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.ckpt_dir / filename
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "global_step": self.global_step,
                "best_bleu": self.best_bleu,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load a checkpoint to resume training."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["global_step"]
        self.best_bleu = ckpt.get("best_bleu", 0.0)
        print(f"Resumed from step {self.global_step} (best BLEU: {self.best_bleu:.2f})")

    def _cleanup_checkpoints(self):
        """Keep only the last N step checkpoints."""
        keep = self.config["checkpoint"].get("keep_last", 5)
        step_ckpts = sorted(self.ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
        for ckpt in step_ckpts[:-keep]:
            ckpt.unlink()
