"""Diagnose whether cross-attention is actually using the source.

Loads the current best checkpoint, feeds 3 wildly different source sentences
through the model with the same decoder input, and checks:
  1. Are encoder outputs different for different sources?
  2. Are decoder outputs (logits) different for different sources?
  3. What does cross-attention actually attend to?

If encoder outputs are ~identical across inputs, the encoder is collapsed.
If encoder outputs differ but decoder outputs don't, cross-attention is dead.
"""

import sys
import torch
import yaml

sys.path.insert(0, ".")
from src.data.tokenizer import Tokenizer, BOS_ID
from src.model.transformer import Transformer


CKPT = "checkpoints/best.pt"
CONFIG = "configs/base.yaml"
SOURCES = [
    "加利福尼亚州水务工程的新问题",
    "蹦床于2000年首次成为奥运会比赛项目。",
    "今天天气真好，我们去公园散步吧。",
]


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    tok = Tokenizer(cfg["data"]["spm_model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_encoder_layers=cfg["model"]["n_encoder_layers"],
        n_decoder_layers=cfg["model"]["n_decoder_layers"],
        d_ff=cfg["model"]["d_ff"],
        dropout=0.0,
        max_seq_len=cfg["model"]["max_seq_len"],
        share_embeddings=cfg["model"]["share_embeddings"],
    ).to(device)

    state = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {CKPT} (step {state.get('global_step', '?')})")

    # Encode each source
    src_tensors = []
    for s in SOURCES:
        ids = tok.encode(s)
        src_tensors.append(torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0))

    # Pad to same length for comparison
    max_len = max(t.size(1) for t in src_tensors)
    padded = torch.zeros(len(SOURCES), max_len, dtype=torch.long, device=device)
    for i, t in enumerate(src_tensors):
        padded[i, : t.size(1)] = t.squeeze(0)

    with torch.no_grad():
        src_mask = model.make_src_mask(padded)
        enc_out = model.encode(padded, src_mask)  # (3, src_len, d_model)

        print("\n=== Encoder output stats ===")
        for i, s in enumerate(SOURCES):
            eo = enc_out[i]
            print(f"[{i}] '{s[:20]}...' -> mean={eo.mean():.4f} "
                  f"std={eo.std():.4f} norm={eo.norm():.2f}")

        print("\n=== Pairwise encoder-output L2 distance ===")
        for i in range(len(SOURCES)):
            for j in range(i + 1, len(SOURCES)):
                # Only compare the first min-length positions (ignore padding)
                minlen = min(src_tensors[i].size(1), src_tensors[j].size(1))
                d = (enc_out[i, :minlen] - enc_out[j, :minlen]).norm().item()
                print(f"  enc[{i}] vs enc[{j}]: {d:.2f}")

        # Now: same decoder input (BOS), for all 3 sources. See if decoder output differs.
        bos = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
        dec_in = bos.repeat(len(SOURCES), 1)  # (3, 1)
        tgt_mask = model.make_tgt_mask(dec_in)
        dec_out = model.decode(dec_in, enc_out, tgt_mask, src_mask)  # (3, 1, d_model)
        logits = model.output_proj(dec_out).squeeze(1)  # (3, vocab)

        print("\n=== Decoder-output (logits) stats after BOS ===")
        for i, s in enumerate(SOURCES):
            top_ids = logits[i].topk(5).indices.tolist()
            top_toks = [tok.sp.IdToPiece(t) for t in top_ids]
            print(f"[{i}] '{s[:20]}...' top5={top_toks}")

        print("\n=== Pairwise logits L2 distance ===")
        for i in range(len(SOURCES)):
            for j in range(i + 1, len(SOURCES)):
                d = (logits[i] - logits[j]).norm().item()
                print(f"  logits[{i}] vs logits[{j}]: {d:.2f}")

    print("\nInterpretation:")
    print("  - If encoder distances are small (<10): encoder collapsed on input.")
    print("  - If encoder distances are fine but logits distances are tiny: "
          "cross-attention is dead.")
    print("  - If logits top-5 are identical across sources: confirmed mode collapse.")


if __name__ == "__main__":
    main()
