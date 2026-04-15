# Machine Translation: Transformer from Scratch

A from-scratch PyTorch implementation of the Transformer model (Vaswani et al., 2017) for Chinese-English machine translation, trained on WMT17 parallel corpus.

**Goal:** Reach BLEU 25+ on WMT test sets without using pretrained models.

## Features

- Pure PyTorch Transformer implementation (no HuggingFace shortcuts)
- Shared SentencePiece BPE tokenizer (32K vocab)
- Token-based dynamic batching for efficient GPU utilization
- Mixed precision training (FP16)
- Label-smoothed cross entropy + Noam learning rate schedule
- Beam search decoding with length penalty
- Graceful Ctrl+C / SIGTERM interrupt (saves checkpoint and resumes)
- Atomic checkpoint writes + rolling emergency save (UPS / power-off safe)
- Automatic training report generation
- TensorBoard logging

## Project Structure

```
Machine_translation/
├── configs/
│   ├── base.yaml              # Transformer Base (65M params)
│   └── big.yaml               # Transformer Big (213M params)
├── scripts/
│   ├── download_data.py       # Download WMT17 zh-en data
│   └── train_tokenizer.py     # Train SentencePiece BPE
├── src/
│   ├── model/                 # Transformer implementation
│   │   ├── attention.py       # Multi-Head Attention
│   │   ├── embeddings.py      # Token + positional encoding
│   │   ├── layers.py          # FFN + residual connections
│   │   ├── encoder.py         # Encoder stack
│   │   ├── decoder.py         # Decoder stack
│   │   └── transformer.py     # Full model
│   ├── data/
│   │   ├── tokenizer.py       # SentencePiece wrapper
│   │   └── dataset.py         # Dataset + token-based batching
│   ├── training/
│   │   ├── loss.py            # Label smoothed cross entropy
│   │   ├── optimizer.py       # Noam LR scheduler
│   │   └── trainer.py         # Main training loop
│   ├── inference/
│   │   └── translate.py       # Beam search decoding
│   └── evaluate.py            # sacrebleu BLEU evaluation
├── train.py                   # Training entry point
├── translate.py               # Inference entry point
└── requirements.txt
```

## Setup

### Requirements
- Python 3.10+
- PyTorch 2.0+ (2.8+ for RTX 5090 / Blackwell GPUs)
- CUDA-capable GPU (16GB+ VRAM recommended)

### Installation
```bash
pip install -r requirements.txt
```

For RTX 5090 (sm_120), install PyTorch nightly with CUDA 12.8:
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Usage

### 1. Download WMT17 zh-en data
```bash
python scripts/download_data.py --output-dir data
```
Downloads ~25M sentence pairs (train / valid / test splits).

### 2. Train BPE tokenizer
```bash
python scripts/train_tokenizer.py \
    --inputs data/train.zh data/train.en \
    --model-prefix data/spm \
    --vocab-size 32000
```
Produces `data/spm.model` and `data/spm.vocab` (shared zh-en BPE, 32K vocab).

### 3. Train the model
```bash
# Transformer Base (~65M params, 1.5-2.5 days on RTX 5090)
python train.py --config configs/base.yaml

# Transformer Big (~213M params, 3-5 days on RTX 5090)
python train.py --config configs/big.yaml
```

**Graceful interrupt:** Press `Ctrl+C` once to save a checkpoint and exit cleanly. Press twice to force-quit without saving.

**Resume from checkpoint:**
```bash
python train.py --config configs/base.yaml --resume checkpoints/interrupted_step_12345.pt
```

**Monitor with TensorBoard:**
```bash
tensorboard --logdir checkpoints/logs
```

### Running training in tmux (recommended for long runs)

Training takes 1–5 days. Using `tmux` lets you detach from the session and safely close your terminal / SSH connection / Jupyter Lab tab without killing the training process.

**Install tmux:**
```bash
# Ubuntu / Debian
sudo apt install tmux
# macOS
brew install tmux
```

**Basic workflow:**
```bash
# 1. Start a new named session
tmux new -s train

# 2. Inside tmux, run training
python train.py --config configs/base.yaml

# 3. Detach (training keeps running):  Ctrl+b  then  d

# 4. Re-attach later from anywhere (new SSH, new terminal, etc.)
tmux attach -t train

# 5. List sessions
tmux ls

# 6. Kill a session when done
tmux kill-session -t train
```

**Useful keybindings (all prefixed with `Ctrl+b`):**

| Keys | Action |
|------|--------|
| `d` | Detach from session |
| `[` | Enter scroll mode (↑/↓/PgUp/PgDn to scroll, `q` to quit) |
| `"` | Split pane horizontally (e.g. to run `nvidia-smi` alongside) |
| `%` | Split pane vertically |
| `o` | Switch between panes |
| `x` | Close current pane |

**Two-pane layout for monitoring:**
```bash
tmux new -s train
# run training in top pane
python train.py --config configs/base.yaml
# Ctrl+b then "  (split horizontally)
# Ctrl+b then o  (switch to bottom pane)
watch -n 2 nvidia-smi
# Ctrl+b then d  (detach — both panes keep running)
```

**Note:** tmux and the graceful-interrupt logic compose naturally. `Ctrl+b d` just detaches; it does NOT send SIGINT. To actually interrupt training cleanly, re-attach first (`tmux attach -t train`), then press `Ctrl+C` inside the session.

### 4. Translate
```bash
# From a file
python translate.py --checkpoint checkpoints/best.pt --input test_input.txt

# From stdin
python translate.py --checkpoint checkpoints/best.pt
```

## Configuration

| Parameter | Base | Big |
|-----------|------|-----|
| d_model | 512 | 1024 |
| n_heads | 8 | 16 |
| n_layers (enc/dec) | 6 / 6 | 6 / 6 |
| d_ff | 2048 | 4096 |
| Dropout | 0.1 | 0.3 |
| Parameters | ~65M | ~213M |
| Vocab | 32K (shared BPE) | 32K (shared BPE) |
| Batch | 32K tokens | 32K tokens |
| Warmup steps | 4000 | 4000 |
| Max steps | 300K | 300K |
| Label smoothing | 0.1 | 0.1 |

## Training Output

- `checkpoints/best.pt` — Best model (by validation BLEU)
- `checkpoints/final.pt` — Final step checkpoint
- `checkpoints/step_*.pt` — Periodic checkpoints (keeps last 5)
- `checkpoints/interrupted_step_*.pt` — Saved on Ctrl+C / SIGTERM
- `checkpoints/emergency.pt` — Rolling save every 500 steps (UPS / power-off fallback)
- `checkpoints/training_report.txt` — Human-readable training summary
- `checkpoints/logs/` — TensorBoard logs

## Results

*To be filled in after training.*

| Config | Valid BLEU | Test BLEU | Training time |
|--------|-----------|-----------|---------------|
| Base   | TBD       | TBD       | TBD           |
| Big    | TBD       | TBD       | TBD           |

## Hardware Used

- GPU: NVIDIA RTX 5090 (32GB VRAM)
- Mixed precision: FP16

## References

- Vaswani et al., *Attention Is All You Need* (2017). [[arxiv]](https://arxiv.org/abs/1706.03762)
- WMT17 Shared Task: Machine Translation. [[link]](https://www.statmt.org/wmt17/)
- SentencePiece: [[github]](https://github.com/google/sentencepiece)
