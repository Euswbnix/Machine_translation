"""SentencePiece BPE tokenizer wrapper for training and encoding."""

import os
import threading
import time
from pathlib import Path

import sentencepiece as spm
from tqdm import tqdm


# Special token IDs (SentencePiece defaults)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def train_tokenizer(
    input_files: list[str],
    model_prefix: str,
    vocab_size: int = 32000,
    character_coverage: float = 1.0,
    input_sentence_size: int = 10_000_000,
    num_threads: int = 8,
):
    """Train a shared SentencePiece BPE model on source + target data.

    Args:
        input_files: List of text file paths (one sentence per line).
        model_prefix: Output path prefix (produces .model and .vocab files).
        vocab_size: Target vocabulary size.
        character_coverage: Fraction of characters to cover. 1.0 = no <unk>
            from rare characters (recommended; 0.9995 caused 4.4% of French
            valid sentences to hit <unk> on accented chars like Israël).
        input_sentence_size: Max sentences to sample for training (caps RAM
            when corpus is huge).
        num_threads: BPE training threads.
    """
    # Pre-scan: line counts per file (fast, gives user a feel for corpus size)
    total_lines = 0
    for path in input_files:
        n = _count_lines(path)
        total_lines += n
        print(f"  {path}: {n:,} lines")
    print(f"Total: {total_lines:,} lines across {len(input_files)} files "
          f"(SPM will sample up to {input_sentence_size:,}).")

    # SentencePiece writes ~100 INFO lines to stderr per training run, which
    # burns terminal I/O time and clutters logs. Suppress them and show an
    # elapsed-time bar from a background thread instead.
    stop_event = threading.Event()
    pbar = tqdm(
        desc="Training SPM",
        bar_format="{desc}: elapsed {elapsed}, {postfix}",
        leave=True,
    )
    pbar.set_postfix_str(f"vocab={vocab_size:,}, coverage={character_coverage}")

    def _tick():
        while not stop_event.is_set():
            pbar.refresh()
            time.sleep(1.0)

    ticker = threading.Thread(target=_tick, daemon=True)
    ticker.start()
    try:
        spm.SentencePieceTrainer.train(
            input=",".join(input_files),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=PAD_ID,
            unk_id=UNK_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            character_coverage=character_coverage,
            num_threads=num_threads,
            shuffle_input_sentence=True,
            input_sentence_size=input_sentence_size,
            minloglevel=2,  # ERROR only — suppress INFO/WARNING spam
        )
    finally:
        stop_event.set()
        ticker.join(timeout=2.0)
        pbar.close()

    print(f"Tokenizer trained: {model_prefix}.model")


class Tokenizer:
    """Wraps a trained SentencePiece model for encoding/decoding."""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Encode text to token IDs, optionally adding BOS/EOS."""
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text, stripping special tokens."""
        # Filter out special tokens
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self.sp.DecodeIds(ids)

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = True,
        add_eos: bool = True,
        num_threads: int | None = None,
    ) -> list[list[int]]:
        # SPM thread count is tunable; default to all cores for C++-side parallelism.
        if num_threads is None:
            num_threads = os.cpu_count() or 1
        return self.sp.encode(
            texts,
            add_bos=add_bos,
            add_eos=add_eos,
            num_threads=num_threads,
            out_type=int,
        )

    def decode_batch(self, batch_ids: list[list[int]]) -> list[str]:
        return [self.decode(ids) for ids in batch_ids]
