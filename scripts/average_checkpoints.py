"""Average the weights of the last N checkpoints.

Implements the Vaswani et al. (2017) checkpoint-averaging trick: take the last
N saved checkpoints, average their model weights element-wise, write out a new
checkpoint. Usually worth +0.3–0.5 BLEU on top of the best single checkpoint
because it smooths out SGD noise in the late-training oscillation band.

Usage:
    # Average last 5 step_*.pt in checkpoints_enfr/
    python scripts/average_checkpoints.py --ckpt-dir checkpoints_enfr --n 5

    # Explicit list
    python scripts/average_checkpoints.py \\
        --ckpts checkpoints_enfr/step_80000.pt checkpoints_enfr/step_85000.pt ... \\
        --out checkpoints_enfr/averaged.pt
"""

import argparse
import re
from pathlib import Path

import torch


def _find_last_n(ckpt_dir: Path, n: int) -> list[Path]:
    """Return the N newest step_*.pt files, sorted by step number."""
    pat = re.compile(r"step_(\d+)\.pt$")
    files = []
    for p in ckpt_dir.iterdir():
        m = pat.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    if len(files) < n:
        raise SystemExit(f"Only {len(files)} step ckpts in {ckpt_dir}, need {n}.")
    return [p for _, p in files[-n:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints_enfr",
                    help="Directory containing step_*.pt files.")
    ap.add_argument("--n", type=int, default=5, help="Number of last ckpts to average.")
    ap.add_argument("--ckpts", nargs="+", default=None,
                    help="Explicit ckpt paths (overrides --ckpt-dir/--n).")
    ap.add_argument("--out", default=None,
                    help="Output path. Defaults to <ckpt_dir>/averaged.pt.")
    args = ap.parse_args()

    if args.ckpts:
        paths = [Path(p) for p in args.ckpts]
    else:
        paths = _find_last_n(Path(args.ckpt_dir), args.n)

    out_path = Path(args.out) if args.out else Path(args.ckpt_dir) / "averaged.pt"

    print(f"Averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"  {p}")

    # Load first ckpt as template, then accumulate
    print("\nLoading and accumulating...")
    states = []
    template = None
    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if template is None:
            template = ckpt
        states.append(ckpt["model"])

    # Sanity: all keys match
    ref_keys = set(states[0].keys())
    for i, sd in enumerate(states[1:], 1):
        if set(sd.keys()) != ref_keys:
            raise SystemExit(f"Checkpoint {paths[i]} has mismatched keys.")

    # Average
    averaged = {}
    for k in ref_keys:
        stacked = torch.stack([sd[k].float() for sd in states], dim=0)
        averaged[k] = stacked.mean(dim=0).to(states[0][k].dtype)

    # Build output ckpt: reuse the last ckpt's metadata, swap in averaged weights.
    # Keeps step/best_bleu/etc. readable by quick_translate_check.py and train.py
    # resume logic (though you shouldn't resume FROM an averaged ckpt).
    template["model"] = averaged
    template["averaged_from"] = [str(p) for p in paths]

    torch.save(template, out_path)
    print(f"\nWrote {out_path}")
    print(f"Evaluate with:")
    print(f"  python translate.py --ckpt {out_path} --config configs/base_en_fr.yaml \\")
    print(f"    --src data/valid.en --ref data/valid.fr")
    print(f"  python translate.py --ckpt {out_path} --config configs/base_en_fr.yaml \\")
    print(f"    --src data/test.en  --ref data/test.fr")


if __name__ == "__main__":
    main()
