"""Main training loop for the Transformer model."""

import datetime
import os
import signal
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from src.model import Transformer
from src.data.tokenizer import Tokenizer, PAD_ID
from src.data.dataset import create_dataloader
from src.training.loss import LabelSmoothedCrossEntropy
from src.training.optimizer import TransformerScheduler
from src.inference.translate import beam_search_translate, TranslationInterrupted
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
            lr_scale=train_cfg.get("lr_scale", 1.0),
            min_lr=train_cfg.get("min_lr", 0.0),
        )

        # Mixed precision
        self.use_fp16 = train_cfg.get("fp16", True) and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_fp16)

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
        self.log_interval = train_cfg["log_interval"]

        # Adaptive eval interval: starts at eval_interval_max when loss is high,
        # shrinks to eval_interval_min as loss drops below eval_loss_threshold.
        self.eval_interval_max = train_cfg.get("eval_interval", 20000)
        self.eval_interval_min = train_cfg.get("eval_interval_min", 2000)
        # Loss range for interpolation: when loss >= high, use max interval;
        # when loss <= low, use min interval; linear interpolation in between.
        self.eval_loss_high = train_cfg.get("eval_loss_high", 4.0)
        self.eval_loss_low = train_cfg.get("eval_loss_low", 3.0)
        self._current_eval_interval = self.eval_interval_max
        self._next_eval_step = self.eval_interval_max  # first eval after warming up
        self._smoothed_loss = None  # EMA of training loss for adaptive eval
        # Early stopping: set patience=0 or early_stopping=false to disable.
        self.patience = train_cfg.get("patience", 10)
        self.early_stopping = train_cfg.get("early_stopping", True) and self.patience > 0
        # Only enforce early stopping after this many global steps. Prevents
        # stopping during the "mode collapse" phase where BLEU is noise.
        self.min_steps_before_early_stop = train_cfg.get("min_steps_before_early_stop", 100000)
        # Emergency rolling save: overwrites emergency.pt every N steps.
        # Caps max data loss on UPS power-off to this many steps.
        self.emergency_save_interval = train_cfg.get("emergency_save_interval", 500)

        # Training history for report
        self.train_start_time = None
        self.history = {
            "train_loss": [],   # (step, loss)
            "valid_bleu": [],   # (step, bleu)
            "lr": [],           # (step, lr)
        }

        # Graceful interrupt handling
        self._interrupted = False
        self._main_pid = os.getpid()

    def _update_eval_interval(self, current_loss: float):
        """Adjust eval interval based on smoothed training loss (EMA).

        Uses exponential moving average to avoid reacting to single-batch
        noise. Only logs when the interval actually changes by a meaningful
        amount (>= 10%).
        """
        # Update EMA (alpha=0.1 → roughly smooths over last ~10 readings)
        if self._smoothed_loss is None:
            self._smoothed_loss = current_loss
        else:
            self._smoothed_loss = 0.9 * self._smoothed_loss + 0.1 * current_loss

        loss = self._smoothed_loss
        if loss >= self.eval_loss_high:
            t = 0.0
        elif loss <= self.eval_loss_low:
            t = 1.0
        else:
            t = (self.eval_loss_high - loss) / (self.eval_loss_high - self.eval_loss_low)

        new_interval = int(
            self.eval_interval_max + t * (self.eval_interval_min - self.eval_interval_max)
        )
        # Round to nearest 1000 for clean numbers
        new_interval = max(self.eval_interval_min, round(new_interval / 1000) * 1000)

        # Only log if interval changed by >= 10% (avoid spamming on noise)
        if new_interval != self._current_eval_interval:
            change_pct = abs(new_interval - self._current_eval_interval) / self._current_eval_interval
            if change_pct >= 0.1:
                print(f"  Eval interval: {self._current_eval_interval} → {new_interval} (smoothed_loss={loss:.3f})")
            self._current_eval_interval = new_interval

    def _handle_signal(self, signum, frame):
        """Handle SIGINT/SIGTERM: set flag to save and exit after current step."""
        # Only the main process should react. DataLoader workers (and any
        # other fork()ed children) inherit this handler; we rely on
        # worker_init_fn to ignore signals there, but this PID check is a
        # belt-and-suspenders safeguard so we never print the message twice.
        if os.getpid() != self._main_pid:
            return
        if self._interrupted:
            print("\nForce exiting...")
            raise SystemExit(1)
        print(f"\nInterrupt received (signal {signum}). "
              "Saving checkpoint after current step... (press again to force quit)")
        self._interrupted = True

    def train(self):
        """Main training loop."""
        # Register signal handlers
        prev_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        prev_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)

        # Epoch tracking
        n_batches = len(self.train_loader.batch_sampler)
        self._epoch = self.global_step // n_batches
        self._epoch_step = self.global_step % n_batches  # steps into current epoch

        print(f"Starting training for {self.max_steps} steps...")
        print(f"  {n_batches:,} batches/epoch, resuming at epoch {self._epoch + 1} (step {self._epoch_step}/{n_batches})")
        print("Press Ctrl+C to gracefully stop and save checkpoint.")
        self.model.train()
        self.train_start_time = time.time()

        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()

        try:
            while self.global_step < self.max_steps:
                for batch in self.train_loader:
                    if self.global_step >= self.max_steps:
                        break

                    # Check for graceful interrupt
                    if self._interrupted:
                        print(f"Saving checkpoint at step {self.global_step}...")
                        self._save_checkpoint(f"interrupted_step_{self.global_step}.pt")
                        self._generate_report("interrupted")
                        print(f"Checkpoint saved. Resume with: --resume {self.ckpt_dir}/interrupted_step_{self.global_step}.pt")
                        return

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

                        self.history["train_loss"].append((self.global_step, avg_loss))
                        self.history["lr"].append((self.global_step, lr))

                        # Update adaptive eval interval based on current loss
                        self._update_eval_interval(avg_loss)

                        total_loss = 0.0
                        total_tokens = 0
                        start_time = time.time()

                    if self.global_step >= self._next_eval_step and self.global_step > 0:
                        bleu = self._evaluate()
                        # Schedule next eval based on current adaptive interval
                        self._next_eval_step = self.global_step + self._current_eval_interval
                        if bleu is None:
                            # Eval was aborted by Ctrl+C; let the next loop
                            # iteration see self._interrupted and save.
                            print(f"Step {self.global_step:>7d} | Eval interrupted, skipping BLEU update")
                            self.model.train()
                            continue
                        print(f"Step {self.global_step:>7d} | Valid BLEU: {bleu:.2f} (next eval at {self._next_eval_step})")
                        self.writer.add_scalar("valid/bleu", bleu, self.global_step)
                        self.history["valid_bleu"].append((self.global_step, bleu))

                        if bleu > self.best_bleu:
                            self.best_bleu = bleu
                            self.patience_counter = 0
                            self._save_checkpoint("best.pt")
                            print(f"  New best BLEU: {bleu:.2f}")
                        else:
                            self.patience_counter += 1
                            if (
                                self.early_stopping
                                and self.global_step >= self.min_steps_before_early_stop
                                and self.patience_counter >= self.patience
                            ):
                                print(f"Early stopping at step {self.global_step}")
                                self._save_checkpoint("final.pt")
                                self._generate_report("early_stopping")
                                return

                        self.model.train()

                    if self.global_step % self.save_interval == 0 and self.global_step > 0:
                        self._save_checkpoint(f"step_{self.global_step}.pt")
                        self._cleanup_checkpoints()

                    # Emergency rolling save: cheap, overwrites single file.
                    # Protects against UPS power-off between regular saves.
                    if (
                        self.emergency_save_interval > 0
                        and self.global_step % self.emergency_save_interval == 0
                        and self.global_step > 0
                    ):
                        self._save_checkpoint("emergency.pt")

                # End of epoch (inner for-loop exhausted)
                self._epoch += 1
                elapsed_total = time.time() - self.train_start_time
                h, m = divmod(int(elapsed_total), 3600)
                m = m // 60
                print(f"{'='*60}")
                print(f"  End of epoch {self._epoch} | Step {self.global_step:,} | Elapsed {h}h{m:02d}m")
                print(f"{'='*60}")

            self._save_checkpoint("final.pt")
            self._generate_report("max_steps_reached")
            print(f"Training complete. Best BLEU: {self.best_bleu:.2f}")

        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

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
    def _evaluate(self) -> float | None:
        """Evaluate on validation set, returns BLEU score.

        Returns None if eval was interrupted by Ctrl+C (so the caller can
        proceed to save and exit without a stale BLEU update).
        """
        self.model.eval()

        data_cfg = self.config["data"]
        infer_cfg = self.config["inference"]

        # Read reference and source
        src_lines = Path(data_cfg["valid_src"]).read_text().strip().split("\n")
        ref_lines = Path(data_cfg["valid_tgt"]).read_text().strip().split("\n")

        # Translate (responsive to Ctrl+C between batches)
        try:
            hypotheses = beam_search_translate(
                model=self.model,
                tokenizer=self.tokenizer,
                src_sentences=src_lines,
                beam_size=infer_cfg["beam_size"],
                max_len=infer_cfg["max_decode_len"],
                length_penalty=infer_cfg["length_penalty"],
                device=self.device,
                should_stop=lambda: self._interrupted,
            )
        except TranslationInterrupted as e:
            print(f"  Eval aborted: {e}")
            return None

        bleu = compute_bleu(hypotheses, ref_lines, tgt_lang=data_cfg["tgt_lang"])

        # Print a few samples for debugging
        print("  --- Sample translations ---")
        for i in [0, 100, 500]:
            if i < len(src_lines):
                print(f"  SRC: {src_lines[i][:100]}")
                print(f"  HYP: {hypotheses[i][:100]}")
                print(f"  REF: {ref_lines[i][:100]}")
                print()

        return bleu

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint atomically (write to tmp + rename).

        Atomic save is important for robustness against UPS/power events:
        if the process is killed mid-write, we don't corrupt the main file.
        """
        path = self.ckpt_dir / filename
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "global_step": self.global_step,
                "best_bleu": self.best_bleu,
                "config": self.config,
                # Persist history so training curves span across resumes.
                "history": self.history,
            },
            tmp_path,
        )
        # Atomic rename: POSIX guarantees this is either fully done or not done.
        tmp_path.replace(path)

    def load_checkpoint(self, path: str, reset_optimizer: bool = False):
        """Load a checkpoint to resume training.

        Args:
            path: Path to checkpoint file.
            reset_optimizer: If True, only load model weights and reset
                optimizer/scheduler/scaler to fresh state. Useful when
                LR schedule was changed and Adam momentum is stale.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])

        if reset_optimizer:
            # Keep model weights, but start optimizer fresh.
            # Reset scheduler step based on global_step and accumulate_steps
            # so LR doesn't jump to warmup level.
            self.global_step = ckpt["global_step"]
            sched_step = self.global_step // self.accumulate_steps
            self.scheduler._step = sched_step
            self.optimizer.zero_grad()
            print(f"  Optimizer/scheduler RESET (scheduler step set to {sched_step}, "
                  f"LR = {self.scheduler.current_lr:.2e})")
        else:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])

        self.global_step = ckpt["global_step"]
        self.best_bleu = ckpt.get("best_bleu", 0.0)

        # Restore history so training curves span resumes. Older checkpoints
        # without this field just start fresh (we lose the pre-resume curve
        # but training itself is unaffected).
        saved_history = ckpt.get("history")
        if saved_history and all(k in saved_history for k in ("train_loss", "valid_bleu", "lr")):
            self.history = saved_history
            n_loss = len(self.history["train_loss"])
            n_bleu = len(self.history["valid_bleu"])
            print(f"  Restored history: {n_loss} loss points, {n_bleu} BLEU points")
        else:
            print("  No history in checkpoint — plot will cover only this session.")

        # Schedule next eval relative to resumed step so we don't
        # trigger an immediate eval on the first iteration.
        self._next_eval_step = self.global_step + self._current_eval_interval
        print(f"Resumed from step {self.global_step} (best BLEU: {self.best_bleu:.2f}, next eval at {self._next_eval_step})")

    def _cleanup_checkpoints(self):
        """Keep only the last N step checkpoints."""
        keep = self.config["checkpoint"].get("keep_last", 5)
        step_ckpts = sorted(self.ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
        for ckpt in step_ckpts[:-keep]:
            ckpt.unlink()

    def _generate_report(self, stop_reason: str):
        """Generate a training report as a plain text file."""
        elapsed = time.time() - self.train_start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_str = f"{hours}h {minutes}m {seconds}s"

        model_cfg = self.config["model"]
        train_cfg = self.config["training"]
        data_cfg = self.config["data"]
        n_params = sum(p.numel() for p in self.model.parameters())

        # Build report
        lines = []
        lines.append("=" * 60)
        lines.append("       TRANSFORMER TRAINING REPORT")
        lines.append("=" * 60)
        lines.append("")

        # General info
        lines.append(f"Date:            {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Stop reason:     {stop_reason}")
        lines.append(f"Total steps:     {self.global_step:,}")
        lines.append(f"Training time:   {elapsed_str}")
        lines.append(f"Best BLEU:       {self.best_bleu:.2f}")
        lines.append("")

        # Model config
        lines.append("-" * 60)
        lines.append("MODEL")
        lines.append("-" * 60)
        lines.append(f"Architecture:    Transformer (Vaswani et al. 2017)")
        lines.append(f"Parameters:      {n_params:,}")
        lines.append(f"d_model:         {model_cfg['d_model']}")
        lines.append(f"n_heads:         {model_cfg['n_heads']}")
        lines.append(f"Encoder layers:  {model_cfg['n_encoder_layers']}")
        lines.append(f"Decoder layers:  {model_cfg['n_decoder_layers']}")
        lines.append(f"FFN dim:         {model_cfg['d_ff']}")
        lines.append(f"Dropout:         {model_cfg['dropout']}")
        lines.append(f"Vocab size:      {model_cfg['vocab_size']}")
        lines.append(f"Max seq len:     {model_cfg['max_seq_len']}")
        lines.append(f"Share embeddings:{model_cfg['share_embeddings']}")
        lines.append("")

        # Training config
        lines.append("-" * 60)
        lines.append("TRAINING")
        lines.append("-" * 60)
        lines.append(f"Batch size:      {train_cfg['batch_size']} tokens")
        lines.append(f"Max sentences:   {train_cfg['max_sentences']}")
        lines.append(f"Warmup steps:    {train_cfg['warmup_steps']}")
        lines.append(f"Label smoothing: {train_cfg['label_smoothing']}")
        lines.append(f"FP16:            {train_cfg.get('fp16', False)}")
        lines.append(f"Grad clip norm:  {train_cfg.get('clip_grad_norm', 1.0)}")
        lines.append(f"Seed:            {train_cfg['seed']}")
        lines.append("")

        # Data
        lines.append("-" * 60)
        lines.append("DATA")
        lines.append("-" * 60)
        lines.append(f"Language pair:   {data_cfg['src_lang']} -> {data_cfg['tgt_lang']}")
        lines.append(f"Train src:       {data_cfg['train_src']}")
        lines.append(f"Train tgt:       {data_cfg['train_tgt']}")
        lines.append(f"Valid src:       {data_cfg['valid_src']}")
        lines.append(f"Valid tgt:       {data_cfg['valid_tgt']}")
        lines.append(f"SPM model:       {data_cfg['spm_model']}")
        lines.append("")

        # Device info
        lines.append("-" * 60)
        lines.append("DEVICE")
        lines.append("-" * 60)
        if self.device.type == "cuda":
            lines.append(f"GPU:             {torch.cuda.get_device_name()}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            lines.append(f"VRAM:            {vram:.1f} GB")
        else:
            lines.append(f"Device:          CPU")
        lines.append("")

        # Training loss history
        lines.append("-" * 60)
        lines.append("TRAINING LOSS HISTORY")
        lines.append("-" * 60)
        lines.append(f"{'Step':>10s}  {'Loss':>10s}")
        for step, loss in self.history["train_loss"]:
            lines.append(f"{step:>10d}  {loss:>10.4f}")
        lines.append("")

        # BLEU history
        lines.append("-" * 60)
        lines.append("VALIDATION BLEU HISTORY")
        lines.append("-" * 60)
        lines.append(f"{'Step':>10s}  {'BLEU':>10s}")
        for step, bleu in self.history["valid_bleu"]:
            lines.append(f"{step:>10d}  {bleu:>10.2f}")
        lines.append("")

        # Learning rate history (sampled)
        lines.append("-" * 60)
        lines.append("LEARNING RATE HISTORY (sampled)")
        lines.append("-" * 60)
        lines.append(f"{'Step':>10s}  {'LR':>12s}")
        lr_history = self.history["lr"]
        # Sample at most 30 points to keep report concise
        if len(lr_history) > 30:
            step_size = len(lr_history) // 30
            lr_sampled = lr_history[::step_size] + [lr_history[-1]]
        else:
            lr_sampled = lr_history
        for step, lr in lr_sampled:
            lines.append(f"{step:>10d}  {lr:>12.2e}")
        lines.append("")

        lines.append("=" * 60)
        lines.append("END OF REPORT")
        lines.append("=" * 60)

        report_text = "\n".join(lines) + "\n"

        report_path = self.ckpt_dir / "training_report.txt"
        report_path.write_text(report_text)
        print(f"Training report saved to {report_path}")

        # Generate training curves
        self._plot_curves()

    def _plot_curves(self):
        """Save training curves as PNG to the project root directory."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Transformer Training Curves", fontsize=16, fontweight="bold")

        # 1. Training loss
        ax = axes[0, 0]
        if self.history["train_loss"]:
            steps, losses = zip(*self.history["train_loss"])
            ax.plot(steps, losses, color="#2196F3", linewidth=0.8, alpha=0.6, label="raw")
            # Smoothed line (EMA)
            if len(losses) > 10:
                smoothed = []
                s = losses[0]
                for l in losses:
                    s = 0.95 * s + 0.05 * l
                    smoothed.append(s)
                ax.plot(steps, smoothed, color="#0D47A1", linewidth=1.5, label="smoothed")
                ax.legend(fontsize=8)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)

        # 2. Validation BLEU
        ax = axes[0, 1]
        if self.history["valid_bleu"]:
            steps, bleus = zip(*self.history["valid_bleu"])
            ax.plot(steps, bleus, color="#4CAF50", linewidth=1.5, marker="o", markersize=3)
            ax.axhline(y=self.best_bleu, color="#E91E63", linestyle="--", linewidth=1, alpha=0.7,
                        label=f"best={self.best_bleu:.2f}")
            ax.legend(fontsize=8)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("BLEU")
        ax.set_title("Validation BLEU")
        ax.grid(True, alpha=0.3)

        # 3. Learning rate
        ax = axes[1, 0]
        if self.history["lr"]:
            steps, lrs = zip(*self.history["lr"])
            ax.plot(steps, lrs, color="#FF9800", linewidth=1.2)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))
        ax.grid(True, alpha=0.3)

        # 4. Loss vs BLEU overlay (dual y-axis)
        ax1 = axes[1, 1]
        if self.history["train_loss"]:
            steps, losses = zip(*self.history["train_loss"])
            # Subsample loss for this plot
            if len(losses) > 100:
                step_size = len(losses) // 100
                steps_s = steps[::step_size]
                losses_s = losses[::step_size]
            else:
                steps_s, losses_s = steps, losses
            line1 = ax1.plot(steps_s, losses_s, color="#2196F3", linewidth=1, alpha=0.7, label="Loss")
            ax1.set_xlabel("Global Step")
            ax1.set_ylabel("Loss", color="#2196F3")
            ax1.tick_params(axis="y", labelcolor="#2196F3")

        ax2 = ax1.twinx()
        if self.history["valid_bleu"]:
            steps, bleus = zip(*self.history["valid_bleu"])
            line2 = ax2.plot(steps, bleus, color="#4CAF50", linewidth=1.5, marker="o", markersize=3, label="BLEU")
            ax2.set_ylabel("BLEU", color="#4CAF50")
            ax2.tick_params(axis="y", labelcolor="#4CAF50")

        # Combined legend
        if self.history["train_loss"] and self.history["valid_bleu"]:
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, fontsize=8, loc="center right")
        ax1.set_title("Loss vs BLEU")
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = Path("training_curves.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Training curves saved to {plot_path}")
