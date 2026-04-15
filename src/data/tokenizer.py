"""SentencePiece BPE tokenizer wrapper for training and encoding."""

from pathlib import Path

import sentencepiece as spm


# Special token IDs (SentencePiece defaults)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def train_tokenizer(
    input_files: list[str],
    model_prefix: str,
    vocab_size: int = 32000,
):
    """Train a shared SentencePiece BPE model on source + target data.

    Args:
        input_files: List of text file paths (one sentence per line).
        model_prefix: Output path prefix (produces .model and .vocab files).
        vocab_size: Target vocabulary size.
    """
    spm.SentencePieceTrainer.train(
        input=",".join(input_files),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        character_coverage=0.9995,  # high coverage for CJK
        num_threads=8,
        shuffle_input_sentence=True,
        input_sentence_size=10_000_000,  # subsample for speed if corpus is huge
    )
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

    def encode_batch(self, texts: list[str], add_bos: bool = True, add_eos: bool = True) -> list[list[int]]:
        return [self.encode(t, add_bos, add_eos) for t in texts]

    def decode_batch(self, batch_ids: list[list[int]]) -> list[str]:
        return [self.decode(ids) for ids in batch_ids]
