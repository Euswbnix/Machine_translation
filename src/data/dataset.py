"""Translation dataset with dynamic batching by token count."""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .tokenizer import Tokenizer, PAD_ID


class TranslationDataset(Dataset):
    """Loads pre-tokenized parallel corpus from text files."""

    def __init__(
        self,
        src_path: str,
        tgt_path: str,
        tokenizer: Tokenizer,
        max_tokens: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.pairs = []

        total = 0
        with open(src_path, "r") as f_src, open(tgt_path, "r") as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                total += 1
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if not src_line or not tgt_line:
                    continue
                src_ids = tokenizer.encode(src_line)
                tgt_ids = tokenizer.encode(tgt_line)
                # Filter out sequences that are too long
                if len(src_ids) <= max_tokens and len(tgt_ids) <= max_tokens:
                    self.pairs.append((src_ids, tgt_ids))

        print(f"Loaded {len(self.pairs)} pairs (filtered from {total})")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.pairs[idx]


class TokenBatchSampler(Sampler):
    """Groups sentences into batches by total token count (not sentence count).

    This maximizes GPU utilization by ensuring each batch has roughly the same
    number of tokens, regardless of sentence length.
    """

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
        # Sort by source length for efficient batching
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            # Shuffle within buckets for some randomness
            random.shuffle(indices)
            # Then sort by length within chunks for batching efficiency
            chunk_size = self.max_sentences * 100
            chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
            indices = []
            for chunk in chunks:
                chunk.sort(key=lambda i: len(self.dataset.pairs[i][0]))
                indices.extend(chunk)

        # Create batches based on token count
        batches = []
        batch = []
        max_src_len = 0
        max_tgt_len = 0

        for idx in indices:
            src_len = len(self.dataset.pairs[idx][0])
            tgt_len = len(self.dataset.pairs[idx][1])
            new_max_src = max(max_src_len, src_len)
            new_max_tgt = max(max_tgt_len, tgt_len)
            num_tokens = (len(batch) + 1) * max(new_max_src, new_max_tgt)

            if num_tokens > self.max_tokens or len(batch) >= self.max_sentences:
                if batch:
                    batches.append(batch)
                batch = [idx]
                max_src_len = src_len
                max_tgt_len = tgt_len
            else:
                batch.append(idx)
                max_src_len = new_max_src
                max_tgt_len = new_max_tgt

        if batch:
            batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        # Approximate
        return len(self.dataset) // self.max_sentences + 1


def collate_fn(batch: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
    """Pad and collate a batch of (src_ids, tgt_ids) pairs."""
    src_seqs, tgt_seqs = zip(*batch)

    src_max_len = max(len(s) for s in src_seqs)
    tgt_max_len = max(len(t) for t in tgt_seqs)

    src_padded = [s + [PAD_ID] * (src_max_len - len(s)) for s in src_seqs]
    tgt_padded = [t + [PAD_ID] * (tgt_max_len - len(t)) for t in tgt_seqs]

    return {
        "src": torch.tensor(src_padded, dtype=torch.long),
        "tgt": torch.tensor(tgt_padded, dtype=torch.long),
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
    )
