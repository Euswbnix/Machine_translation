"""Package a trained checkpoint + tokenizer + model card for upload to HF Hub.

Takes a full training checkpoint (the kind `trainer._save_checkpoint` writes,
with optimizer / scheduler / scaler state and training history baked in) and
strips it down to just the model weights needed for inference. Also emits a
clean `config.json`, a model-card `README.md`, and copies the SPM tokenizer.

Usage:
    python scripts/prepare_hf_release.py \\
        --ckpt checkpoints_enfr/averaged.pt \\
        --spm data/spm_enfr_v1.model \\
        --out-dir hf_release/enfr_base_v1 \\
        --valid-bleu 30.00 --test-bleu 34.69

Then to upload (huggingface_hub>=0.25 CLI uses `hf`; older versions use
`huggingface-cli`, which still works but prints a deprecation warning):
    hf auth login   # one-time
    hf upload Euswbnix/transformer-wmt14-enfr-base \\
        hf_release/enfr_base_v1 . --repo-type model
"""

import argparse
import json
import shutil
from pathlib import Path

import torch


MODEL_CARD_TEMPLATE = """---
license: mit
language:
  - en
  - fr
tags:
  - translation
  - transformer
  - from-scratch
  - wmt14
  - pytorch
datasets:
  - wmt14
metrics:
  - sacrebleu
library_name: pytorch
---

# Transformer Base — WMT14 en→fr (from scratch)

A **from-scratch PyTorch implementation** of the Transformer (Vaswani et al.,
2017), trained on **WMT14 English→French** without any pretrained weights.
This is the strongest checkpoint from the parent project and the one worth
sharing externally.

| Metric | Value |
|---|---|
| **Test BLEU (newstest2014)** | **{test_bleu}** |
| Valid BLEU (newstest2013) | {valid_bleu} |
| Tokenizer | sacrebleu `13a` (detokenized) |
| Parameters | {n_params:,} |
| Training compute | single RTX 5090, ~2h 5m |

BLEU is reported as sacrebleu `13a`, the modern detokenized standard.
Vaswani's original paper reports **38.1 in historical tokenized BLEU** for
Base on WMT14 en-fr, which is roughly equivalent to **35-36 sacrebleu** —
so this checkpoint lands ~1-1.5 BLEU below paper Base, attributable to
training on 9.3M strict-filtered pairs (vs the paper's ~36M full corpus).

See the parent repository for full training logs, ablation studies against
larger variants, and why a 9.3M *clean* corpus outperformed a 30M *noisy*
one (data quality > data quantity > capacity).

## Architecture

Standard Transformer Base from the paper, no architectural modifications:

| | |
|---|---|
| d_model | {d_model} |
| n_heads | {n_heads} |
| encoder layers | {n_encoder_layers} |
| decoder layers | {n_decoder_layers} |
| FFN dim | {d_ff} |
| dropout | {dropout} |
| max seq len | {max_seq_len} |
| vocab size | {vocab_size} (shared SentencePiece BPE) |
| shared embeddings | {share_embeddings} |

## Files in this repo

| File | Purpose |
|---|---|
| `pytorch_model.bin` | Model weights only (averaged over the last 5 step-checkpoints, Vaswani trick) |
| `sentencepiece.model` | Shared 32K BPE tokenizer (SentencePiece) |
| `config.json` | Architecture config — sufficient to instantiate the model |
| `example.py` | Minimal self-contained inference script |
| `README.md` | This file |

## Usage

```bash
# 1. Clone the parent repo for model definition + beam search code
git clone https://github.com/Euswbnix/Machine_translation
cd Machine_translation
pip install -r requirements.txt

# 2. Download the weights + tokenizer from this HF repo
pip install huggingface_hub
hf download Euswbnix/transformer-wmt14-enfr-base \\
    pytorch_model.bin sentencepiece.model config.json --local-dir hf_model

# 3. Translate
python examples/load_and_translate.py \\
    --weights hf_model/pytorch_model.bin \\
    --spm hf_model/sentencepiece.model \\
    --config hf_model/config.json \\
    --text "Machine learning is transforming the world."
```

Or use the bundled `example.py`:

```python
import sentencepiece as spm
import torch
from src.model import Transformer

# Load the model (shapes come from config.json)
cfg = json.load(open("config.json"))
model = Transformer(
    vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
    n_heads=cfg["n_heads"], n_encoder_layers=cfg["n_encoder_layers"],
    n_decoder_layers=cfg["n_decoder_layers"], d_ff=cfg["d_ff"],
    dropout=0.0, max_seq_len=cfg["max_seq_len"],
    share_embeddings=cfg["share_embeddings"], pad_idx=0,
)
model.load_state_dict(torch.load("pytorch_model.bin", map_location="cpu"))
model.eval()
```

## Training data

- **Source**: WMT14 en-fr parallel corpus (via HuggingFace `datasets` `wmt14` config)
- **Cleaning**: strict filter — length ratio [0.5, 2.0], min 3 / max 200
  tokens per side, Latin-script ratio ≥ 0.9, no tgt line appearing > 50×
- **Post-clean size**: 9.3M pairs (of 10M subsampled from raw)
- **BPE**: 32K shared vocab, SentencePiece, character coverage 0.9995

## Intended use & limitations

- Translates **English → French** news / general prose
- Trained only on WMT14 (≈2014 news + Europarl + Common Crawl)
- Does not handle code, tables, or long documents
- Output may reflect biases present in WMT14 training data
- Not benchmarked on low-resource domains (medical, legal, etc.)

## Citation

If you use this checkpoint, please cite the original Transformer paper:

```bibtex
@inproceedings{{vaswani2017attention,
  title={{Attention is all you need}},
  author={{Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others}},
  booktitle={{NeurIPS}},
  year={{2017}}
}}
```

And link back to this repo and the GitHub project:
- HF: https://huggingface.co/Euswbnix/transformer-wmt14-enfr-base
- Code: https://github.com/Euswbnix/Machine_translation
"""


EXAMPLE_PY = '''"""Minimal end-to-end example: load the HF-hosted weights and translate.

Run from the parent project directory (so `src` is importable):
    python example.py --text "Hello world, how are you?"
"""

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch

# Requires: src/ from https://github.com/Euswbnix/Machine_translation on the path
from src.model import Transformer
from src.inference.translate import batched_beam_search
from src.data.tokenizer import BOS_ID, EOS_ID, PAD_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="pytorch_model.bin")
    ap.add_argument("--spm", default="sentencepiece.model")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--text", required=True, help="English sentence to translate.")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())

    model = Transformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_encoder_layers=cfg["n_encoder_layers"],
        n_decoder_layers=cfg["n_decoder_layers"],
        d_ff=cfg["d_ff"],
        dropout=0.0,
        max_seq_len=cfg["max_seq_len"],
        share_embeddings=cfg["share_embeddings"],
        pad_idx=PAD_ID,
    ).to(args.device)
    model.load_state_dict(torch.load(args.weights, map_location=args.device))
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm)

    # Encode: wrap with BOS/EOS the same way the trainer does
    ids = [BOS_ID] + sp.encode(args.text, out_type=int) + [EOS_ID]
    src = torch.tensor([ids], dtype=torch.long, device=args.device)

    hyp_ids = batched_beam_search(
        model, src, beam_size=args.beam, max_len=cfg["max_seq_len"], length_penalty=1.0
    )[0]

    # Strip BOS/EOS and decode
    hyp_ids = [t for t in hyp_ids if t not in (BOS_ID, EOS_ID, PAD_ID)]
    print(sp.decode(hyp_ids))


if __name__ == "__main__":
    main()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to averaged.pt (or best.pt)")
    ap.add_argument("--spm", required=True, help="Path to SentencePiece .model")
    ap.add_argument("--out-dir", required=True, help="Output folder (will be created)")
    ap.add_argument("--valid-bleu", type=float, required=True)
    ap.add_argument("--test-bleu", type=float, required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    cfg = ckpt["config"]

    n_params = sum(v.numel() for v in state_dict.values() if v.dtype.is_floating_point)
    print(f"  Model params: {n_params:,}")
    print(f"  Step: {ckpt.get('global_step', 'unknown')}")

    # Save clean weights-only file
    weights_path = out / "pytorch_model.bin"
    torch.save(state_dict, weights_path)
    size_mb = weights_path.stat().st_size / 1e6
    print(f"Wrote {weights_path} ({size_mb:.1f} MB)")

    # Copy SPM tokenizer
    spm_dst = out / "sentencepiece.model"
    shutil.copy(args.spm, spm_dst)
    print(f"Copied {args.spm} → {spm_dst}")

    # Write architecture config
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    config_json = {
        "architecture": "Transformer (Vaswani 2017, from-scratch)",
        "model_type": "transformer-mt",
        "d_model": model_cfg["d_model"],
        "n_heads": model_cfg["n_heads"],
        "n_encoder_layers": model_cfg["n_encoder_layers"],
        "n_decoder_layers": model_cfg["n_decoder_layers"],
        "d_ff": model_cfg["d_ff"],
        "dropout": model_cfg["dropout"],
        "max_seq_len": model_cfg["max_seq_len"],
        "vocab_size": model_cfg["vocab_size"],
        "share_embeddings": model_cfg["share_embeddings"],
        "pad_idx": 0, "unk_idx": 1, "bos_idx": 2, "eos_idx": 3,
        "src_lang": data_cfg["src_lang"],
        "tgt_lang": data_cfg["tgt_lang"],
        "training": {
            "corpus": "WMT14 en-fr (9.3M strict-filtered pairs)",
            "steps": ckpt.get("global_step"),
            "valid_bleu_newstest2013": args.valid_bleu,
            "test_bleu_newstest2014": args.test_bleu,
            "averaging": "last-5 step checkpoints",
        },
    }
    (out / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")
    print(f"Wrote {out / 'config.json'}")

    # Write model card
    card = MODEL_CARD_TEMPLATE.format(
        test_bleu=args.test_bleu,
        valid_bleu=args.valid_bleu,
        n_params=n_params,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_encoder_layers=model_cfg["n_encoder_layers"],
        n_decoder_layers=model_cfg["n_decoder_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        max_seq_len=model_cfg["max_seq_len"],
        vocab_size=model_cfg["vocab_size"],
        share_embeddings=model_cfg["share_embeddings"],
    )
    (out / "README.md").write_text(card)
    print(f"Wrote {out / 'README.md'}")

    # Write the bundled example.py
    (out / "example.py").write_text(EXAMPLE_PY)
    print(f"Wrote {out / 'example.py'}")

    print(f"\nRelease folder ready at: {out}")
    print(f"Total size:")
    for f in sorted(out.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<30s} {size_mb:>8.2f} MB")

    print("\nNext steps:")
    print(f"  hf auth login   # one-time if not done")
    print(f"  hf upload Euswbnix/transformer-wmt14-enfr-base \\")
    print(f"      {out} . --repo-type model")


if __name__ == "__main__":
    main()
