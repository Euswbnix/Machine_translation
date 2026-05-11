"""Evaluate translations with COMET-22 (reference-based) and optionally CometKiwi-22.

Translates test/valid sets from a checkpoint, then scores with COMET.
Outputs a JSON results file with BLEU + COMET scores.

Usage:
    python scripts/wmt26_upgrade/eval_comet.py \
        --checkpoint checkpoints/base_enfr_v1_redo/best.pt \
        --comet-model Unbabel/wmt22-comet-da \
        --out results/base_v1_s42.json

    # Batch mode: evaluate all checkpoints matching a glob
    python scripts/wmt26_upgrade/eval_comet.py \
        --checkpoint-glob "checkpoints/*/best.pt" \
        --comet-model Unbabel/wmt22-comet-da \
        --out-dir results/
"""

import argparse
import glob
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml


def translate_checkpoint(ckpt_path: str, out_path: str,
                         beam_size: int = 5, length_penalty: float = 1.0,
                         max_len: int = 256) -> dict:
    """Run translate.py and return config metadata."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    src_lang = config["data"]["src_lang"]
    tgt_lang = config["data"]["tgt_lang"]
    test_src = config["data"]["test_src"]
    test_tgt = config["data"]["test_tgt"]

    cmd = [
        sys.executable, "translate.py",
        "--checkpoint", ckpt_path,
        "--input", test_src,
        "--beam-size", str(beam_size),
        "--max-len", str(max_len),
        "--length-penalty", str(length_penalty),
    ]
    print(f"  Translating: {' '.join(cmd)}", file=sys.stderr)

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # translate.py outputs "SRC: ...\nTGT: ...\n" pairs; extract TGT lines only
    with open(out_path, "w", encoding="utf-8") as f:
        for line in result.stdout.splitlines():
            if line.startswith("TGT: "):
                f.write(line[5:] + "\n")

    return {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "test_src": test_src,
        "test_ref": test_tgt,
        "seed": config["training"].get("seed", 42),
        "checkpoint": ckpt_path,
    }


def eval_bleu(hyp_path: str, ref_path: str) -> float:
    """Compute sacrebleu 13a score."""
    import sacrebleu
    with open(hyp_path, encoding="utf-8") as fh, \
         open(ref_path, encoding="utf-8") as fr:
        hyps = [l.strip() for l in fh]
        refs = [l.strip() for l in fr]
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return round(bleu.score, 2)


def eval_comet(src_path: str, hyp_path: str, ref_path: str,
               model_name: str) -> float:
    """Score with COMET (reference-based)."""
    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    with open(src_path, encoding="utf-8") as fs, \
         open(hyp_path, encoding="utf-8") as fh, \
         open(ref_path, encoding="utf-8") as fr:
        data = [
            {"src": s.strip(), "mt": h.strip(), "ref": r.strip()}
            for s, h, r in zip(fs, fh, fr)
        ]

    output = model.predict(data, batch_size=64, gpus=1, progress_bar=False)
    return round(output["system_score"], 4)


def process_one(ckpt_path: str, comet_model: str, out_path: str,
                beam_size: int, length_penalty: float):
    """Translate + score one checkpoint."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Checkpoint: {ckpt_path}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        hyp_path = tmp.name

    meta = translate_checkpoint(ckpt_path, hyp_path,
                                beam_size=beam_size,
                                length_penalty=length_penalty)

    bleu = eval_bleu(hyp_path, meta["test_ref"])
    print(f"  BLEU: {bleu}", file=sys.stderr)

    comet = eval_comet(meta["test_src"], hyp_path, meta["test_ref"], comet_model)
    print(f"  COMET: {comet}", file=sys.stderr)

    result = {
        **meta,
        "bleu": bleu,
        "comet": comet,
        "comet_model": comet_model,
        "beam_size": beam_size,
        "length_penalty": length_penalty,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}", file=sys.stderr)

    Path(hyp_path).unlink(missing_ok=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Single checkpoint path")
    group.add_argument("--checkpoint-glob", help="Glob pattern for batch mode")
    ap.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--out", help="Output JSON (single mode)")
    ap.add_argument("--out-dir", default="results/", help="Output dir (batch mode)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    args = ap.parse_args()

    if args.checkpoint:
        out = args.out or f"results/{Path(args.checkpoint).parent.name}.json"
        process_one(args.checkpoint, args.comet_model, out,
                    args.beam_size, args.length_penalty)
    else:
        paths = sorted(glob.glob(args.checkpoint_glob))
        print(f"Found {len(paths)} checkpoints", file=sys.stderr)
        for p in paths:
            name = Path(p).parent.name
            out = str(Path(args.out_dir) / f"{name}.json")
            process_one(p, args.comet_model, out,
                        args.beam_size, args.length_penalty)


if __name__ == "__main__":
    main()
