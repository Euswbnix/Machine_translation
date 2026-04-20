"""Translation dataset with dynamic batching by token count.

Memory-efficient storage: tokens are stored as flat numpy uint16 arrays with
offset indices, and cached to disk after the first tokenization pass.
"""

import random
import signal
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from .tokenizer import Tokenizer, PAD_ID


def _worker_init_ignore_signals(worker_id: int):
    """Make DataLoader workers ignore SIGINT/SIGTERM.

    Without this, Ctrl+C in the terminal sends SIGINT to the whole process
    group: main + every worker each run the parent's signal handler, causing
    the "Interrupt received..." message to print once per worker. We want
    only the main process to handle the signal.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


class TranslationDataset(Dataset):
    """Loads parallel corpus, stores tokens as flat numpy arrays for memory efficiency.

    Caches tokenized output to disk so subsequent runs skip tokenization.
    """

    def __init__(
        self,
        src_path: str,
        tgt_path: str,
        tokenizer: Tokenizer,
        max_tokens: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        cache_path = Path(str(src_path) + f".cached_{max_tokens}.npz")

        if cache_path.exists():
            print(f"Loading cached tokenized data from {cache_path}...")
            data = np.load(cache_path)
            self.src_tokens = data["src_tokens"]
            self.src_offsets = data["src_offsets"]
            self.tgt_tokens = data["tgt_tokens"]
            self.tgt_offsets = data["tgt_offsets"]
            print(f"Loaded {len(self.src_offsets) - 1} pairs from cache")
        else:
            self._build_from_text(src_path, tgt_path, max_tokens)
            print(f"Saving tokenized cache to {cache_path}...")
            np.savez(
                cache_path,
                src_tokens=self.src_tokens,
                src_offsets=self.src_offsets,
                tgt_tokens=self.tgt_tokens,
                tgt_offsets=self.tgt_offsets,
            )

        # Pre-compute lengths for the batch sampler (fast access)
        self.src_lens = np.diff(self.src_offsets).astype(np.int32)
        self.tgt_lens = np.diff(self.tgt_offsets).astype(np.int32)

    def _build_from_text(self, src_path: str, tgt_path: str, max_tokens: int, chunk_size: int = 100_000):
        """Tokenize raw text files and store as flat numpy arrays.

        Reads both files in chunks and batch-encodes each chunk via
        SentencePiece's multi-threaded C++ API (see Tokenizer.encode_batch).
        """
        # Count lines for progress bar
        with open(src_path, "r") as f:
            n_lines = sum(1 for _ in f)

        src_flat: list[int] = []
        tgt_flat: list[int] = []
        src_offsets: list[int] = [0]
        tgt_offsets: list[int] = [0]

        total = 0

        def _flush(src_chunk: list[str], tgt_chunk: list[str]):
            if not src_chunk:
                return
            src_batch = self.tokenizer.encode_batch(src_chunk)
            tgt_batch = self.tokenizer.encode_batch(tgt_chunk)
            for src_ids, tgt_ids in zip(src_batch, tgt_batch):
                if len(src_ids) > max_tokens or len(tgt_ids) > max_tokens:
                    continue
                src_flat.extend(src_ids)
                tgt_flat.extend(tgt_ids)
                src_offsets.append(len(src_flat))
                tgt_offsets.append(len(tgt_flat))

        src_chunk: list[str] = []
        tgt_chunk: list[str] = []
        with open(src_path, "r") as f_src, open(tgt_path, "r") as f_tgt:
            pbar = tqdm(total=n_lines, desc="Tokenizing")
            for src_line, tgt_line in zip(f_src, f_tgt):
                total += 1
                pbar.update(1)
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if not src_line or not tgt_line:
                    continue
                src_chunk.append(src_line)
                tgt_chunk.append(tgt_line)
                if len(src_chunk) >= chunk_size:
                    _flush(src_chunk, tgt_chunk)
                    src_chunk.clear()
                    tgt_chunk.clear()
            _flush(src_chunk, tgt_chunk)
            pbar.close()

        self.src_tokens = np.array(src_flat, dtype=np.uint16)
        self.tgt_tokens = np.array(tgt_flat, dtype=np.uint16)
        self.src_offsets = np.array(src_offsets, dtype=np.int64)
        self.tgt_offsets = np.array(tgt_offsets, dtype=np.int64)

        n_pairs = len(self.src_offsets) - 1
        print(f"Loaded {n_pairs} pairs (filtered from {total})")

    def __len__(self) -> int:
        return len(self.src_offsets) - 1

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        src = self.src_tokens[self.src_offsets[idx] : self.src_offsets[idx + 1]]
        tgt = self.tgt_tokens[self.tgt_offsets[idx] : self.tgt_offsets[idx + 1]]
        return src, tgt


class TokenBatchSampler(Sampler):
    """Groups sentences into batches by total token count (not sentence count)."""

    def __init__(
        self,
        dataset: TranslationDataset,
        max_tokens: int = 32768,
        max_sentences: int = 256,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.shuffle = shuffle

    def __iter__(self):
        src_lens = self.dataset.src_lens
        tgt_lens = self.dataset.tgt_lens
        n = len(self.dataset)

        # Shuffle within buckets, then sort within chunks for efficient batching
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)
            chunk_size = self.max_sentences * 100
            for i in range(0, n, chunk_size):
                chunk = indices[i : i + chunk_size]
                chunk_sorted = chunk[np.argsort(src_lens[chunk])]
                indices[i : i + chunk_size] = chunk_sorted

        # Create batches based on token count
        batches = []
        batch = []
        max_len = 0

        for idx in indices:
            seq_len = max(int(src_lens[idx]), int(tgt_lens[idx]))
            new_max = max(max_len, seq_len)
            num_tokens = (len(batch) + 1) * new_max

            if num_tokens > self.max_tokens or len(batch) >= self.max_sentences:
                if batch:
                    batches.append(batch)
                batch = [int(idx)]
                max_len = seq_len
            else:
                batch.append(int(idx))
                max_len = new_max

        if batch:
            batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        return len(self.dataset) // self.max_sentences + 1


def collate_fn(batch: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Pad and collate a batch of (src_ids, tgt_ids) pairs."""
    src_seqs, tgt_seqs = zip(*batch)

    src_max_len = max(len(s) for s in src_seqs)
    tgt_max_len = max(len(t) for t in tgt_seqs)

    batch_size = len(batch)
    src_padded = np.full((batch_size, src_max_len), PAD_ID, dtype=np.int64)
    tgt_padded = np.full((batch_size, tgt_max_len), PAD_ID, dtype=np.int64)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_padded[i, : len(s)] = s
        tgt_padded[i, : len(t)] = t

    return {
        "src": torch.from_numpy(src_padded),
        "tgt": torch.from_numpy(tgt_padded),
    }


def create_dataloader(
    src_path: str,
    tgt_path: str,
    tokenizer: Tokenizer,
    max_tokens_per_batch: int = 32768,
    max_sentences: int = 256,
    max_seq_len: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with dynamic token-based batching."""
    dataset = TranslationDataset(src_path, tgt_path, tokenizer, max_seq_len)
    sampler = TokenBatchSampler(dataset, max_tokens_per_batch, max_sentences, shuffle)

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_ignore_signals if num_workers > 0 else None,
    )
