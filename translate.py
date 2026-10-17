"""Translate text using a trained model checkpoint."""

import argparse
import sys

import torch
import yaml

from src.model import Transformer
from src.data.tokenizer import Tokenizer, PAD_ID
from src.inference.translate import beam_search_translate


def main():
    parser = argparse.ArgumentParser(description="Translate with trained Transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--input", type=str, default=None, help="Input file (one sentence per line)")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Build model
    model_cfg = config["model"]
    model = Transformer(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_encoder_layers=model_cfg["n_encoder_layers"],
        n_decoder_layers=model_cfg["n_decoder_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=0.0,  # no dropout at inference
        max_seq_len=model_cfg["max_seq_len"],
        share_embeddings=model_cfg["share_embeddings"],
        pad_idx=PAD_ID,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load tokenizer
    tokenizer = Tokenizer(config["data"]["spm_model"])

    # Read input
    # From a file, keep one output per input line, empty ones included: dropping empty lines
    # made the output shorter than the input, which shifts every later hypothesis against its
    # reference in any script that scores the output line by line (2026-10 fix).
    if args.input:
        with open(args.input, encoding="utf-8", newline="\n") as f:
            all_lines = [line.strip() for line in f]
    else:
        print("Enter text to translate (Ctrl+D to finish):")
        all_lines = [line.strip() for line in sys.stdin if line.strip()]
    keep = [i for i, s in enumerate(all_lines) if s]
    sentences = [all_lines[i] for i in keep]

    # Translate
    translations = beam_search_translate(
        model=model,
        tokenizer=tokenizer,
        src_sentences=sentences,
        beam_size=args.beam_size,
        max_len=args.max_len,
        length_penalty=args.length_penalty,
        device=device,
    )

    by_line = [""] * len(all_lines)
    for i, tgt in zip(keep, translations, strict=True):
        by_line[i] = tgt
    for src, tgt in zip(all_lines, by_line, strict=True):
        print(f"SRC: {src}")
        print(f"TGT: {tgt}")
        print()


if __name__ == "__main__":
    main()
