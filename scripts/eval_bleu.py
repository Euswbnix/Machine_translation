"""Run a checkpoint on a parallel test set and report sacrebleu.

Mirrors what the in-training eval does, but standalone — use after averaging
checkpoints, or to re-score any single ckpt on valid/test.

Usage:
    python scripts/eval_bleu.py \\
        --ckpt checkpoints_enfr/averaged.pt \\
        --config configs/base_en_fr.yaml \\
        --src data/valid.en --ref data/valid.fr

    python scripts/eval_bleu.py \\
        --ckpt checkpoints_enfr/averaged.pt \\
        --config configs/base_en_fr.yaml \\
        --src data/test.en --ref data/test.fr
"""

import argparse
from pathlib import Path

import torch
import yaml

from src.data.tokenizer import Tokenizer, PAD_ID
from src.evaluate import compute_bleu
from src.inference.translate import beam_search_translate
from src.model import Transformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/base_en_fr.yaml")
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default=None, help="Optional: write hypotheses here.")
    ap.add_argument("--beam", type=int, default=None,
                    help="Overrides config inference.beam_size.")
    ap.add_argument("--length-penalty", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-sentences", type=int, default=None,
                    help="Debug: cap #sentences.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model_cfg = cfg["model"]
    model = Transformer(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_encoder_layers=model_cfg["n_encoder_layers"],
        n_decoder_layers=model_cfg["n_decoder_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=0.0,
        max_seq_len=model_cfg["max_seq_len"],
        share_embeddings=model_cfg["share_embeddings"],
        pad_idx=PAD_ID,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.ckpt}"
          f" (step={ckpt.get('global_step', '?')},"
          f" best_bleu={ckpt.get('best_bleu', '?')})")
    if "averaged_from" in ckpt:
        print(f"  averaged from: {ckpt['averaged_from']}")

    # Tokenizer
    tok = Tokenizer(cfg["data"]["spm_model"])

    # Data
    with open(args.src, encoding="utf-8") as f:
        srcs = [ln.strip() for ln in f]
    with open(args.ref, encoding="utf-8") as f:
        refs = [ln.strip() for ln in f]
    assert len(srcs) == len(refs), f"len mismatch: {len(srcs)} vs {len(refs)}"

    if args.max_sentences:
        srcs = srcs[: args.max_sentences]
        refs = refs[: args.max_sentences]
    print(f"Sentences: {len(srcs)}")

    # Translate
    infer_cfg = cfg.get("inference", {})
    beam = args.beam or infer_cfg.get("beam_size", 5)
    lp = args.length_penalty if args.length_penalty is not None \
        else infer_cfg.get("length_penalty", 1.0)
    max_len = infer_cfg.get("max_decode_len", 256)

    print(f"Decoding: beam={beam} length_penalty={lp} max_len={max_len}")
    hyps = beam_search_translate(
        model=model,
        tokenizer=tok,
        src_sentences=srcs,
        beam_size=beam,
        max_len=max_len,
        length_penalty=lp,
        device=device,
        batch_size=args.batch_size,
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for h in hyps:
                f.write(h + "\n")
        print(f"Wrote hypotheses -> {args.out}")

    tgt_lang = cfg["data"].get("tgt_lang", "en")
    bleu = compute_bleu(hyps, refs, tgt_lang=tgt_lang)
    print(f"\nBLEU ({tgt_lang}, sacrebleu 13a): {bleu:.2f}")


if __name__ == "__main__":
    main()
