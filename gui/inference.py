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


class CancelledError(Exception):
    """Raised when a translation run is cancelled by the caller."""


# pysbd is small (~2MB) and handles English abbreviations / numbers well.
# Lazy-imported so the desktop binary doesn't pay the cost on launch
# if the user never translates.
def _segment(text: str) -> List[str]:
    text = text.replace("\u2029", "\n").replace("\u2028", "\n")
    try:
        import pysbd
        seg = pysbd.Segmenter(language="en", clean=False)
        return [s for s in seg.segment(text) if s.strip()]
    except ImportError:
        # Fallback: split on newlines first, and also on sentence-final
        # punctuation followed by whitespace. Coarser than pysbd but keeps
        # email/paragraph formatting much better.
        parts = re.split(r"(?:\r?\n+|(?<=[.!?])\s+)", text)
        return [p for p in parts if p.strip()]

def _build_sentence_skeleton(text: str) -> Tuple[List[Tuple[bool, str]], List[str]]:
    """Return (parts, sentences) where parts preserve original separators.

    parts is a list of (is_sentence, chunk). Non-sentence chunks keep the
    original whitespace/newlines/punctuation between sentence chunks so we
    can reconstruct output with input formatting preserved.
    """
    text = text.replace("\u2029", "\n").replace("\u2028", "\n")
    sents = _segment(text)
    if not sents:
        return ([(False, text)] if text else []), []

    parts: List[Tuple[bool, str]] = []
    idx = 0
    for sent in sents:
        pos = text.find(sent, idx)
        if pos < 0:
            # Fallback: if exact substring match fails, keep behavior safe.
            return [], sents
        if pos > idx:
            parts.append((False, text[idx:pos]))
        parts.append((True, sent))
        idx = pos + len(sent)
    if idx < len(text):
        parts.append((False, text[idx:]))
    return parts, sents


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


def _fix_missing_space_after_sentence_punct(text: str) -> str:
    """Insert a space after sentence punctuation when it's missing.

    Example: "phares.Auparavant" -> "phares. Auparavant"
    """
    # Keep newlines intact by only touching same-line runs (spaces/tabs).
    # Handles:
    #   "phares.Auparavant" -> "phares. Auparavant"
    #   ").J'ai"            -> "). J'ai"
    #   "»Aujourd'hui"      -> "» Aujourd'hui"
    pattern = r"([.!?…][\"')\]»”]*)([ \t]*)(?=[A-Za-zÀ-ÖØ-öø-ÿ])"
    return re.sub(pattern, r"\1 ", text)


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
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

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
        return _fix_missing_space_after_sentence_punct(self._decode(hyp))

    # ------------------------------------------------------------ public API

    def translate(
        self,
        text: str,
        beam: int = 5,
        length_penalty: Optional[float] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        sentence_cache: Optional[dict] = None,
    ) -> TranslationResult:
        """Translate arbitrary-length text via sentence segmentation.

        ``cancel_check`` is polled between sentences; returning True aborts
        the run early and raises CancelledError. ``sentence_cache`` is an
        optional dict[str, str] that memoizes per-sentence translations
        across calls (live-typing mode reuses it so re-translating only
        hits the model on changed sentences).
        """
        text = text.replace("\u2029", "\n").replace("\u2028", "\n")

        # en-de in our setup uses lp=0.6 by default; en-fr uses 1.0
        if length_penalty is None:
            tgt_lang = self.cfg.get("tgt_lang", "fr")
            length_penalty = 0.6 if tgt_lang == "de" else 1.0

        # Strict formatting preservation: split by physical lines while keeping
        # line endings exactly, translate per line body, then stitch back.
        lines = text.splitlines(keepends=True)
        if not lines and text:
            lines = [text]
        all_sentences_src: List[str] = []
        per_line_sentences: List[Tuple[int, str, str, List[Tuple[bool, str]], List[str]]] = []
        for idx, line in enumerate(lines):
            body = line.rstrip("\r\n")
            ending = line[len(body):]
            parts, sents = _build_sentence_skeleton(body)
            per_line_sentences.append((idx, body, ending, parts, sents))
            all_sentences_src.extend(sents)

        outputs: List[str] = []
        flagged: List[bool] = []
        total = len(all_sentences_src)
        for i, sent in enumerate(all_sentences_src):
            if cancel_check is not None and cancel_check():
                raise CancelledError()
            if progress_cb is not None:
                progress_cb(i, total, sent[:60])
            if not _is_translatable(sent):
                outputs.append(sent)
                flagged.append(False)
                continue
            cache_key = (sent, beam, length_penalty)
            if sentence_cache is not None and cache_key in sentence_cache:
                tgt = sentence_cache[cache_key]
            else:
                tgt = self._translate_one(sent, beam=beam,
                                          length_penalty=length_penalty)
                if sentence_cache is not None:
                    sentence_cache[cache_key] = tgt
            # Normalize punctuation spacing even for cache hits from older runs.
            tgt = _fix_missing_space_after_sentence_punct(tgt)
            outputs.append(tgt)
            flagged.append(_looks_hallucinated(sent, tgt))
        if progress_cb is not None:
            progress_cb(total, total, "done")

        # Rebuild each line body with translated sentence slice, then re-attach
        # the exact original line ending.
        out_lines = list(lines)
        sent_idx = 0
        for line_idx, body, ending, parts, sents in per_line_sentences:
            n = len(sents)
            translated_body = _reassemble(
                body,
                parts,
                outputs[sent_idx : sent_idx + n],
            )
            out_lines[line_idx] = translated_body + ending
            sent_idx += n
        output_text = _fix_missing_space_after_sentence_punct("".join(out_lines))

        return TranslationResult(
            sentences_src=all_sentences_src,
            sentences_tgt=outputs,
            flagged=flagged,
            output_text=output_text,
        )


def _reassemble(
    original: str,
    parts: List[Tuple[bool, str]],
    tgt_sents: List[str],
) -> str:
    """Rebuild text by replacing only sentence chunks.

    If sentence-to-text alignment fails, fallback to previous paragraph logic.
    """
    if not parts:
        # Fallback for alignment failure: preserve line breaks exactly.
        src_sents = _segment(original)
        if not src_sents:
            return original
        line_chunks = re.split(r"(\r?\n)", original)  # keep separators
        out_chunks: List[str] = []
        sent_idx = 0
        for chunk in line_chunks:
            if re.fullmatch(r"\r?\n", chunk):
                out_chunks.append(chunk)
                continue
            segs = _segment(chunk)
            n = len(segs)
            if n == 0:
                out_chunks.append(chunk)
                continue
            translated = tgt_sents[sent_idx : sent_idx + n]
            sent_idx += n
            out_chunks.append(" ".join(translated))
        if sent_idx < len(tgt_sents):
            out_chunks.append(" ".join(tgt_sents[sent_idx:]))
        return "".join(out_chunks)

    out_chunks: List[str] = []
    sent_idx = 0
    for is_sentence, chunk in parts:
        if not is_sentence:
            out_chunks.append(chunk)
            continue
        if sent_idx < len(tgt_sents):
            out_chunks.append(tgt_sents[sent_idx])
            sent_idx += 1
        else:
            out_chunks.append(chunk)
    return "".join(out_chunks)
