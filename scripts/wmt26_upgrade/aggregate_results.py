"""Aggregate multi-seed results into mean±std tables for the paper.

Reads JSON result files from eval_comet.py and produces:
  1. A console-formatted table (copy-pasteable)
  2. A LaTeX-formatted table
  3. A JSON summary

Usage:
    python scripts/wmt26_upgrade/aggregate_results.py \
        --results-dir results/ \
        --out results/summary.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_cell_name(filename: str) -> tuple[str, str]:
    """Extract (cell, seed) from filename like 'base_enfr_v1_redo_s1.json'."""
    name = Path(filename).stem

    seed = "s42"
    for s in ["_s1", "_s2", "_s3"]:
        if name.endswith(s):
            seed = s.lstrip("_")
            name = name[: -len(s)]
            break

    return name, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/")
    ap.add_argument("--out", default="results/summary.json")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    json_files = sorted(results_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    cells = defaultdict(list)
    for f in json_files:
        if f.name == "summary.json":
            continue
        cell, seed = parse_cell_name(f.name)
        with open(f) as fh:
            data = json.load(fh)
        data["_seed"] = seed
        data["_cell"] = cell
        cells[cell].append(data)

    print(f"\n{'Cell':<30} {'N':>3} {'BLEU mean±std':>16} {'COMET mean±std':>18}",
          file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    summary = {}
    latex_rows = []

    for cell in sorted(cells.keys()):
        runs = cells[cell]
        bleus = [r["bleu"] for r in runs]
        comets = [r["comet"] for r in runs]
        n = len(runs)

        bleu_mean = np.mean(bleus)
        bleu_std = np.std(bleus, ddof=1) if n > 1 else 0.0
        comet_mean = np.mean(comets)
        comet_std = np.std(comets, ddof=1) if n > 1 else 0.0

        summary[cell] = {
            "n_seeds": n,
            "bleu_mean": round(bleu_mean, 2),
            "bleu_std": round(bleu_std, 2),
            "comet_mean": round(comet_mean, 4),
            "comet_std": round(comet_std, 4),
            "bleu_values": bleus,
            "comet_values": comets,
            "seeds": [r["_seed"] for r in runs],
        }

        bleu_str = f"{bleu_mean:.2f}±{bleu_std:.2f}"
        comet_str = f"{comet_mean:.4f}±{comet_std:.4f}"
        print(f"{cell:<30} {n:>3} {bleu_str:>16} {comet_str:>18}",
              file=sys.stderr)

        latex_rows.append(
            f"  {cell.replace('_', ' ')} & {n} & "
            f"${bleu_mean:.2f} \\pm {bleu_std:.2f}$ & "
            f"${comet_mean:.4f} \\pm {comet_std:.4f}$ \\\\"
        )

    print(f"\n--- LaTeX Table Body ---\n", file=sys.stderr)
    print("\\toprule", file=sys.stderr)
    print("\\textbf{Cell} & \\textbf{N} & \\textbf{BLEU} & \\textbf{COMET} \\\\",
          file=sys.stderr)
    print("\\midrule", file=sys.stderr)
    for row in latex_rows:
        print(row, file=sys.stderr)
    print("\\bottomrule", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
