"""Inference-side glue: load checkpoint, segment text, translate, reassemble.

Sentence segmentation prevents truncation when input exceeds the model's
trained 256-token max length. Per-sentence translation matches the
training distribution and is what production MT systems do.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import sentencepiece as spm
import torch

from src.model import Transformer
from src.inference.translate import batched_beam_search
from src.data.tokenizer import BOS_ID, EOS_ID, PAD_ID


# pysbd is small (~2MB) and handles English abbreviations / numbers well.
# Lazy-imported so the desktop binary doesn't pay the cost on launch
# if the user never translates.
def _segment(text: str) -> List[str]:
    try:
        import pysbd
        seg = pysbd.Segmenter(language="en", clean=False)
        return [s for s in seg.segment(text) if s.strip()]
    except ImportError:
        # Fallback: simple regex split on sentence-final punctuation
        # followed by whitespace. Coarser than pysbd but works.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
        return [p for p in parts if p.strip()]


def _is_translatable(s: str) -> bool:
    """Skip empty / pure-punctuation / URL fragments."""
    s = s.strip()
    if len(s) < 2:
        return False
    if not re.search(r"[a-zA-Z]", s):
        return False
    if re.match(r"^https?://", s):
        return False
    return True


def _looks_hallucinated(src: str, tgt: str) -> bool:
    """Length-ratio heuristic for likely hallucinations."""
    src_w = len(src.split())
    tgt_w = len(tgt.split())
    if src_w < 3:
        return False
    ratio = tgt_w / max(src_w, 1)
    return ratio > 2.5 or ratio < 0.3


@dataclass
class TranslationResult:
    sentences_src: List[str]
    sentences_tgt: List[str]
    flagged: List[bool]              # parallel to sentences_*; True = possibly hallucinated
    output_text: str                 # reassembled with original line/paragraph structure


class TransformerMTRuntime:
    """Loads a checkpoint + SPM and exposes translate(text)."""

    def __init__(self, model_dir: Path, device: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.model_dir / "config.json") as f:
            cfg = json.load(f)
        self.cfg = cfg
        self.max_len = int(cfg.get("max_seq_len", 256))

        self.model = Transformer(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_encoder_layers=cfg["n_encoder_layers"],
            n_decoder_layers=cfg["n_decoder_layers"],
            d_ff=cfg["d_ff"],
            dropout=0.0,
            max_seq_len=self.max_len,
            share_embeddings=cfg.get("share_embeddings", True),
            pad_idx=PAD_ID,
        ).to(self.device)
        state = torch.load(self.model_dir / "pytorch_model.bin",
                           map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_dir / "sentencepiece.model"))

    # ------------------------------------------------------------ helpers

    def _encode(self, text: str) -> torch.Tensor:
        ids = [BOS_ID] + self.sp.encode(text, out_type=int) + [EOS_ID]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _decode(self, ids: List[int]) -> str:
        ids = [t for t in ids if t not in (BOS_ID, EOS_ID, PAD_ID)]
        return self.sp.decode(ids)

    def _translate_one(self, sent: str, beam: int = 5,
                       length_penalty: float = 1.0) -> str:
        src = self._encode(sent)
        # Hard guard: even after sentence segmentation, a single sentence may
        # exceed max_seq_len (e.g. legal text without commas). Truncate
        # quietly to avoid crashes; the length-ratio check downstream will
        # flag it if the result looks pathological.
        if src.size(1) > self.max_len:
            src = src[:, : self.max_len]
        hyp = batched_beam_search(
            self.model, src,
            beam_size=beam,
            max_len=self.max_len,
            length_penalty=length_penalty,
        )[0]
        return self._decode(hyp)

    # ------------------------------------------------------------ public API

    def translate(
        self,
        text: str,
        beam: int = 5,
        length_penalty: Optional[float] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> TranslationResult:
        """Translate arbitrary-length text via sentence segmentation."""
        # en-de in our setup uses lp=0.6 by default; en-fr uses 1.0
        if length_penalty is None:
            tgt_lang = self.cfg.get("tgt_lang", "fr")
            length_penalty = 0.6 if tgt_lang == "de" else 1.0

        sentences = _segment(text)
        outputs: List[str] = []
        flagged: List[bool] = []

        for i, sent in enumerate(sentences):
            if progress_cb is not None:
                progress_cb(i, len(sentences), sent[:60])
            if not _is_translatable(sent):
                outputs.append(sent)         # preserve URLs, pure punctuation, etc.
                flagged.append(False)
                continue
            tgt = self._translate_one(sent, beam=beam,
                                      length_penalty=length_penalty)
            outputs.append(tgt)
            flagged.append(_looks_hallucinated(sent, tgt))
        if progress_cb is not None:
            progress_cb(len(sentences), len(sentences), "done")

        # Re-assemble. We don't try to reconstruct exact whitespace from the
        # input — we join translated sentences with single spaces, but
        # preserve newline groupings by re-detecting them from the input.
        output_text = _reassemble(text, sentences, outputs)

        return TranslationResult(
            sentences_src=sentences,
            sentences_tgt=outputs,
            flagged=flagged,
            output_text=output_text,
        )


def _reassemble(original: str, src_sents: List[str], tgt_sents: List[str]) -> str:
    """Re-join translated sentences while preserving paragraph breaks.

    Strategy: split original into paragraphs (delimited by blank lines),
    translate sentences, regroup translated sentences into matching
    paragraphs by counting how many src sentences came from each paragraph.
    """
    if not src_sents:
        return ""
    paragraphs = original.split("\n\n")
    # Match each sentence to its paragraph by re-segmenting the paragraph.
    # (Cheaper alternative: just concatenate with space, lose paragraph breaks.)
    out_paragraphs = []
    sent_idx = 0
    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            out_paragraphs.append("")
            continue
        para_sents = _segment(para)
        n = len(para_sents)
        if n == 0:
            out_paragraphs.append(para)  # nothing to translate (e.g. blank)
            continue
        translated = tgt_sents[sent_idx : sent_idx + n]
        sent_idx += n
        out_paragraphs.append(" ".join(translated))
    # If any leftover (shouldn't happen) just append
    if sent_idx < len(tgt_sents):
        out_paragraphs.append(" ".join(tgt_sents[sent_idx:]))
    return "\n\n".join(out_paragraphs)
