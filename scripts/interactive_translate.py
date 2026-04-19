"""Interactive translation: load a checkpoint and translate user input on demand.

Loads the model once, then reads lines from stdin (or --input file) and prints
translations. Useful for quick demo / manual inspection after training finishes.

Usage:
    # Interactive REPL — type sentences, get translations, Ctrl+D to quit
    python scripts/interactive_translate.py \\
        --ckpt checkpoints_enfr/averaged.pt \\
        --config configs/base_en_fr.yaml

    # Batch mode — one sentence per line in a file
    python scripts/interactive_translate.py \\
        --ckpt checkpoints_enfr/averaged.pt \\
        --config configs/base_en_fr.yaml \\
        --input my_sentences.txt

    # Different beam / length penalty
    python scripts/interactive_translate.py --ckpt ... --beam 10 --length-penalty 0.6
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

from src.data.tokenizer import Tokenizer, PAD_ID
from src.inference.translate import beam_search_translate
from src.model import Transformer


def load_model(ckpt_path: str, config_path: str, device: torch.device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = Tokenizer(cfg["data"]["spm_model"])

    info = {
        "step": ckpt.get("global_step", "?"),
        "best_bleu": ckpt.get("best_bleu", "?"),
        "averaged_from": ckpt.get("averaged_from"),
        "src_lang": cfg["data"].get("src_lang", "?"),
        "tgt_lang": cfg["data"].get("tgt_lang", "?"),
    }
    return model, tok, cfg, info


def translate_lines(lines: list[str], model, tok, cfg, device, beam, lp):
    lines = [l.strip() for l in lines if l.strip()]
    if not lines:
        return []
    infer_cfg = cfg.get("inference", {})
    return beam_search_translate(
        model=model,
        tokenizer=tok,
        src_sentences=lines,
        beam_size=beam,
        max_len=infer_cfg.get("max_decode_len", 256),
        length_penalty=lp,
        device=device,
        batch_size=min(len(lines), 32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (.pt).")
    ap.add_argument("--config", required=True, help="Matching config yaml.")
    ap.add_argument("--input", default=None,
                    help="Optional: batch-translate a file (one sentence per line). "
                         "If omitted, runs an interactive REPL on stdin.")
    ap.add_argument("--output", default=None,
                    help="Optional: write translations to this file. "
                         "Default: stdout.")
    ap.add_argument("--beam", type=int, default=None,
                    help="Beam size. Defaults to config inference.beam_size.")
    ap.add_argument("--length-penalty", type=float, default=None,
                    help="Length penalty. Defaults to config inference.length_penalty.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", file=sys.stderr)

    model, tok, cfg, info = load_model(args.ckpt, args.config, device)
    print(f"Loaded {args.ckpt}", file=sys.stderr)
    print(f"  step={info['step']}  best_bleu={info['best_bleu']}", file=sys.stderr)
    if info["averaged_from"]:
        print(f"  averaged from: {len(info['averaged_from'])} checkpoints",
              file=sys.stderr)
    print(f"  direction: {info['src_lang']} -> {info['tgt_lang']}", file=sys.stderr)

    infer_cfg = cfg.get("inference", {})
    beam = args.beam or infer_cfg.get("beam_size", 5)
    lp = args.length_penalty if args.length_penalty is not None \
        else infer_cfg.get("length_penalty", 1.0)
    print(f"  beam={beam} length_penalty={lp}", file=sys.stderr)

    # Batch mode
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            lines = f.readlines()
        hyps = translate_lines(lines, model, tok, cfg, device, beam, lp)
        out_stream = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        try:
            for h in hyps:
                print(h, file=out_stream)
        finally:
            if args.output:
                out_stream.close()
        if args.output:
            print(f"Wrote {len(hyps)} translations to {args.output}", file=sys.stderr)
        return

    # Interactive REPL
    print(file=sys.stderr)
    print(f"Interactive mode. Type a sentence and press Enter. Ctrl+D to quit.",
          file=sys.stderr)
    print(f"Prompt: {info['src_lang']}> ", file=sys.stderr)
    try:
        while True:
            try:
                line = input(f"{info['src_lang']}> ")
            except EOFError:
                print(file=sys.stderr)
                break
            if not line.strip():
                continue
            hyps = translate_lines([line], model, tok, cfg, device, beam, lp)
            if hyps:
                print(f"{info['tgt_lang']}: {hyps[0]}")
    except KeyboardInterrupt:
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
