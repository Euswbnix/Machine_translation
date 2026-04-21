"""Minimal end-to-end example: load a released checkpoint and translate.

Works with weights hosted on HuggingFace
(`euswbnix/transformer-wmt14-enfr-base`) or any local averaged / best
checkpoint. Run from the project root so `src/` is importable.

Usage:
    # Using files downloaded from HF:
    python examples/load_and_translate.py \\
        --weights hf_model/pytorch_model.bin \\
        --spm hf_model/sentencepiece.model \\
        --config hf_model/config.json \\
        --text "Machine learning is transforming the world."

    # Using a local full training checkpoint (with optimizer state):
    python examples/load_and_translate.py \\
        --ckpt checkpoints_enfr/averaged.pt \\
        --spm data/spm_enfr_v1.model \\
        --text "Hello world."
"""

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch

from src.model import Transformer
from src.inference.translate import batched_beam_search
from src.data.tokenizer import BOS_ID, EOS_ID, PAD_ID


def build_model_from_config(cfg: dict, device: str) -> Transformer:
    return Transformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_encoder_layers=cfg["n_encoder_layers"],
        n_decoder_layers=cfg["n_decoder_layers"],
        d_ff=cfg["d_ff"],
        dropout=0.0,  # inference: disable dropout
        max_seq_len=cfg["max_seq_len"],
        share_embeddings=cfg["share_embeddings"],
        pad_idx=PAD_ID,
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="pytorch_model.bin (HF-style weights-only)")
    ap.add_argument("--config", help="config.json (HF-style architecture config)")
    ap.add_argument("--ckpt", help="Full training checkpoint (averaged.pt / best.pt)")
    ap.add_argument("--spm", required=True, help="SentencePiece .model file")
    ap.add_argument("--text", required=True, help="English sentence to translate")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        model = build_model_from_config(ckpt["config"]["model"], args.device)
        model.load_state_dict(ckpt["model"])
        max_len = ckpt["config"]["model"]["max_seq_len"]
    else:
        if not (args.weights and args.config):
            ap.error("Provide either --ckpt, or both --weights and --config")
        cfg = json.loads(Path(args.config).read_text())
        model = build_model_from_config(cfg, args.device)
        model.load_state_dict(torch.load(args.weights, map_location=args.device))
        max_len = cfg["max_seq_len"]
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm)

    ids = [BOS_ID] + sp.encode(args.text, out_type=int) + [EOS_ID]
    src = torch.tensor([ids], dtype=torch.long, device=args.device)
    hyp_ids = batched_beam_search(
        model, src, beam_size=args.beam, max_len=max_len, length_penalty=1.0
    )[0]
    hyp_ids = [t for t in hyp_ids if t not in (BOS_ID, EOS_ID, PAD_ID)]
    print(sp.decode(hyp_ids))


if __name__ == "__main__":
    main()
