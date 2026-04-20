"""Translation dataset with dynamic batching by token count.

Memory-efficient storage: tokens are stored as flat numpy uint16 arrays with
offset indices, and cached to disk after the first tokenization pass.
"""

import multiprocessing as mp
import os
import random
import signal
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from .tokenizer import Tokenizer, PAD_ID


_WORKER_SP: "spm.SentencePieceProcessor | None" = None


def _tokenize_worker_init(model_proto: bytes):
    global _WORKER_SP
    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(model_proto)
    _WORKER_SP = sp


def _tokenize_chunk_worker(args):
    """Tokenize a chunk of (src, tgt) line pairs and apply max_tokens filter.

    Returns pre-packed numpy arrays (uint16 tokens + int32 per-sample lengths)
    instead of Python lists. Keeps main-process memory flat: 2 bytes/token
    vs ~28 bytes per Python int, and avoids 10^9-element list.extend() which
    swaps the machine late in a 30M-pair run.
    """
    chunk_idx, src_lines, tgt_lines, max_tokens = args
    sp = _WORKER_SP
    src_batch = sp.encode(src_lines, add_bos=True, add_eos=True, out_type=int)
    tgt_batch = sp.encode(tgt_lines, add_bos=True, add_eos=True, out_type=int)

    src_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    src_lens: list[int] = []
    tgt_lens: list[int] = []
    for src_ids, tgt_ids in zip(src_batch, tgt_batch):
        if len(src_ids) > max_tokens or len(tgt_ids) > max_tokens:
            continue
        src_parts.append(np.asarray(src_ids, dtype=np.uint16))
        tgt_parts.append(np.asarray(tgt_ids, dtype=np.uint16))
        src_lens.append(len(src_ids))
        tgt_lens.append(len(tgt_ids))

    if src_parts:
        src_arr = np.concatenate(src_parts)
        tgt_arr = np.concatenate(tgt_parts)
    else:
        src_arr = np.empty(0, dtype=np.uint16)
        tgt_arr = np.empty(0, dtype=np.uint16)
    return (
        chunk_idx,
        src_arr,
        tgt_arr,
        np.asarray(src_lens, dtype=np.int32),
        np.asarray(tgt_lens, dtype=np.int32),
    )


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

    def _build_from_text(self, src_path: str, tgt_path: str, max_tokens: int, chunk_size: int = 50_000):
        """Tokenize raw text files and store as flat numpy arrays.

        Fans chunks of line pairs out to a multiprocessing.Pool where each
        worker owns its own SentencePieceProcessor. Bypasses the GIL so
        SPM's internal C++ threads stay fed. Chunks are consumed in order
        (imap) so output offsets are deterministic and byte-for-byte match
        the previous single-process implementation.
        """
        with open(src_path, "r") as f:
            n_lines = sum(1 for _ in f)

        # Accumulate per-chunk numpy arrays; concatenate once at the end.
        # Avoids building two Python lists of ~10^9 int objects, which would
        # take ~50 GB and tank throughput via GC + swap near the tail.
        src_arrays: list[np.ndarray] = []
        tgt_arrays: list[np.ndarray] = []
        src_lens_parts: list[np.ndarray] = []
        tgt_lens_parts: list[np.ndarray] = []
        total = 0

        # Serialize SPM model once in the parent so workers can rebuild a
        # processor locally — SentencePieceProcessor itself isn't picklable.
        model_proto = self.tokenizer.sp.serialized_model_proto()

        def _chunk_iter():
            nonlocal total
            src_chunk: list[str] = []
            tgt_chunk: list[str] = []
            chunk_idx = 0
            with open(src_path, "r") as f_src, open(tgt_path, "r") as f_tgt:
                for src_line, tgt_line in zip(f_src, f_tgt):
                    total += 1
                    src_line = src_line.strip()
                    tgt_line = tgt_line.strip()
                    if not src_line or not tgt_line:
                        continue
                    src_chunk.append(src_line)
                    tgt_chunk.append(tgt_line)
                    if len(src_chunk) >= chunk_size:
                        yield (chunk_idx, src_chunk, tgt_chunk, max_tokens)
                        chunk_idx += 1
                        src_chunk = []
                        tgt_chunk = []
                if src_chunk:
                    yield (chunk_idx, src_chunk, tgt_chunk, max_tokens)

        n_procs = max(1, (os.cpu_count() or 2) // 2)
        ctx = mp.get_context("spawn")
        pbar = tqdm(total=n_lines, desc="Tokenizing")
        last_total = 0
        with ctx.Pool(
            processes=n_procs,
            initializer=_tokenize_worker_init,
            initargs=(model_proto,),
        ) as pool:
            for _idx, src_arr, tgt_arr, s_lens, t_lens in pool.imap(
                _tokenize_chunk_worker, _chunk_iter(), chunksize=1
            ):
                if s_lens.size:
                    src_arrays.append(src_arr)
                    tgt_arrays.append(tgt_arr)
                    src_lens_parts.append(s_lens)
                    tgt_lens_parts.append(t_lens)
                # total grows as the generator is consumed; advance pbar to match.
                pbar.update(total - last_total)
                last_total = total
        pbar.update(total - last_total)
        pbar.close()

        self.src_tokens = (
            np.concatenate(src_arrays) if src_arrays else np.empty(0, dtype=np.uint16)
        )
        self.tgt_tokens = (
            np.concatenate(tgt_arrays) if tgt_arrays else np.empty(0, dtype=np.uint16)
        )
        # Build offsets from cumulative per-sample lengths.
        src_lens_all = (
            np.concatenate(src_lens_parts) if src_lens_parts else np.empty(0, dtype=np.int32)
        )
        tgt_lens_all = (
            np.concatenate(tgt_lens_parts) if tgt_lens_parts else np.empty(0, dtype=np.int32)
        )
        self.src_offsets = np.empty(len(src_lens_all) + 1, dtype=np.int64)
        self.tgt_offsets = np.empty(len(tgt_lens_all) + 1, dtype=np.int64)
        self.src_offsets[0] = 0
        self.tgt_offsets[0] = 0
        np.cumsum(src_lens_all, dtype=np.int64, out=self.src_offsets[1:])
        np.cumsum(tgt_lens_all, dtype=np.int64, out=self.tgt_offsets[1:])

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
