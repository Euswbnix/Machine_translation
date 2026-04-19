"""Quick sanity check: translate a few validation sentences with the current model.

Loads latest checkpoint (or --ckpt), picks N lines from valid set, and prints
SRC / REF / HYP side-by-side. Meant for eyeball inspection between evals to
see whether the model is learning source conditioning or still mode-collapsed.

Usage:
    python scripts/quick_translate_check.py                      # uses default config/big.yaml, latest emergency.pt
    python scripts/quick_translate_check.py --ckpt checkpoints_big/step_70000.pt
    python scripts/quick_translate_check.py --n 10 --indices 0,5,42,100
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.tokenizer import Tokenizer
from src.inference.translate import beam_search_translate
from src.model.transformer import Transformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/big.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="Checkpoint path. Defaults to <ckpt_dir>/emergency.pt, "
                         "falling back to best.pt.")
    ap.add_argument("--n", type=int, default=5, help="Number of sentences to translate.")
    ap.add_argument("--indices", default=None,
                    help="Comma-separated line indices into valid set (overrides --n).")
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_dir = Path(cfg["checkpoint"]["dir"])
        for candidate in ("emergency.pt", "best.pt"):
            if (ckpt_dir / candidate).exists():
                ckpt_path = str(ckpt_dir / candidate)
                break
        if ckpt_path is None:
            raise SystemExit(f"No checkpoint found in {ckpt_dir}. Pass --ckpt.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {ckpt_path}")

    tok = Tokenizer(cfg["data"]["spm_model"])

    model_cfg = cfg["model"]
    model = Transformer(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_encoder_layers=model_cfg["n_encoder_layers"],
        n_decoder_layers=model_cfg["n_decoder_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=0.0,                             # turn off dropout for eval
        max_seq_len=model_cfg["max_seq_len"],
        share_embeddings=model_cfg["share_embeddings"],
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    step = state.get("global_step", "?")
    bleu = state.get("best_bleu", "?")
    print(f"  step={step}, best_bleu={bleu}")

    # Load valid set
    src_path = cfg["data"]["valid_src"]
    tgt_path = cfg["data"]["valid_tgt"]
    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = [ln.strip() for ln in f]
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_lines = [ln.strip() for ln in f]
    assert len(src_lines) == len(tgt_lines), "valid src/tgt length mismatch"
    print(f"Valid set: {len(src_lines)} lines")

    # Pick indices
    if args.indices:
        indices = [int(x) for x in args.indices.split(",")]
    else:
        # First, a few middle, a few long — gives a feel for different lengths
        total = len(src_lines)
        indices = []
        for i in range(args.n):
            indices.append((i * total) // args.n)
    indices = [i for i in indices if 0 <= i < len(src_lines)]

    srcs = [src_lines[i] for i in indices]
    refs = [tgt_lines[i] for i in indices]

    # Translate
    infer_cfg = cfg.get("inference", {})
    hyps = beam_search_translate(
        model=model,
        tokenizer=tok,
        src_sentences=srcs,
        beam_size=args.beam,
        max_len=infer_cfg.get("max_decode_len", 256),
        length_penalty=infer_cfg.get("length_penalty", 1.0),
        device=device,
        batch_size=len(srcs),
    )

    # Print
    print()
    print("=" * 70)
    for idx, src, ref, hyp in zip(indices, srcs, refs, hyps):
        print(f"[line {idx}]")
        print(f"  SRC: {src}")
        print(f"  REF: {ref}")
        print(f"  HYP: {hyp}")
        print()


if __name__ == "__main__":
    main()
