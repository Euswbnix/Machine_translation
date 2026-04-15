"""Beam search decoding for translation inference.

Provides both single-sentence and batched beam search. Batched version
dramatically speeds up evaluation (~10-20x) by processing multiple
sentences simultaneously on the GPU.
"""

from typing import Callable, Optional

import torch
from tqdm import tqdm

from src.model import Transformer
from src.data.tokenizer import Tokenizer, BOS_ID, EOS_ID, PAD_ID


class TranslationInterrupted(Exception):
    """Raised when beam search is aborted via should_stop() callback."""
    pass


@torch.no_grad()
def batched_beam_search(
    model: Transformer,
    src_batch: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 256,
    length_penalty: float = 1.0,
) -> list[list[int]]:
    """Batched beam search decoding.

    Args:
        model: Transformer model (eval mode).
        src_batch: (batch_size, src_len) source token IDs, padded.
        beam_size: Beam width.
        max_len: Max decoding length.
        length_penalty: Length normalization factor alpha.

    Returns:
        List of best hypotheses (one per source sentence), each as a list of token IDs.
    """
    device = src_batch.device
    B = src_batch.size(0)
    K = beam_size

    # Encode once (not per beam)
    src_mask = model.make_src_mask(src_batch)          # (B, 1, 1, L)
    enc = model.encode(src_batch, src_mask)            # (B, L, D)
    L = enc.size(1)
    D = enc.size(-1)

    # Expand encoder outputs for each beam: (B*K, L, D)
    enc = enc.unsqueeze(1).expand(B, K, L, D).reshape(B * K, L, D)
    src_mask = src_mask.unsqueeze(1).expand(B, K, 1, 1, L).reshape(B * K, 1, 1, L)

    # Initialize beams with BOS
    beams = torch.full((B * K, 1), BOS_ID, dtype=torch.long, device=device)

    # Scores: (B, K), only first beam active initially
    scores = torch.zeros(B, K, device=device)
    scores[:, 1:] = float("-inf")

    # Finished hypotheses per sentence
    finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(B)]
    active = torch.ones(B, dtype=torch.bool, device=device)

    for step in range(max_len):
        if not active.any():
            break

        tgt_mask = model.make_tgt_mask(beams)
        dec = model.decode(beams, enc, tgt_mask, src_mask)   # (B*K, cur_len, D)
        logits = model.output_proj(dec[:, -1, :])            # (B*K, V)
        logp = torch.log_softmax(logits, dim=-1)
        V = logp.size(-1)

        # Candidate scores: (B, K, V) -> (B, K*V)
        cand = scores.unsqueeze(-1) + logp.view(B, K, V)
        cand = cand.view(B, K * V)

        # Take top 2K per sentence (extra room for EOS candidates)
        top_scores, top_idx = cand.topk(2 * K, dim=-1)       # (B, 2K)
        top_beam = top_idx // V
        top_tok = top_idx % V

        new_beams = torch.zeros(B * K, beams.size(1) + 1, dtype=torch.long, device=device)
        new_scores = torch.full((B, K), float("-inf"), device=device)

        # Move to CPU once for the Python loop (faster than .item() per element)
        top_scores_cpu = top_scores.cpu().tolist()
        top_beam_cpu = top_beam.cpu().tolist()
        top_tok_cpu = top_tok.cpu().tolist()
        active_cpu = active.cpu().tolist()

        for b in range(B):
            if not active_cpu[b]:
                # Freeze this sentence's beams (scores stay -inf)
                new_beams[b * K : (b + 1) * K, :-1] = beams[b * K : (b + 1) * K]
                continue

            n_added = 0
            for i in range(2 * K):
                s = top_scores_cpu[b][i]
                beam_i = top_beam_cpu[b][i]
                tok = top_tok_cpu[b][i]

                if tok == EOS_ID:
                    seq = beams[b * K + beam_i].tolist() + [EOS_ID]
                    lp = ((5.0 + len(seq)) / 6.0) ** length_penalty
                    finished[b].append((s / lp, seq))
                elif n_added < K:
                    new_beams[b * K + n_added, :-1] = beams[b * K + beam_i]
                    new_beams[b * K + n_added, -1] = tok
                    new_scores[b, n_added] = s
                    n_added += 1

            # Stop early if we have K finished and the best alive can't beat them
            if len(finished[b]) >= K and n_added > 0:
                best_fin = max(f[0] for f in finished[b])
                # Alive beam's normalized score can only decrease as sequence grows,
                # so if best alive score / min_length_penalty < best_fin, stop.
                best_alive_norm = new_scores[b, 0].item() / (((5.0 + beams.size(1) + 1) / 6.0) ** length_penalty)
                if best_alive_norm < best_fin:
                    active[b] = False

        beams = new_beams
        scores = new_scores

    # Flush remaining alive beams to finished
    for b in range(B):
        for i in range(K):
            s = scores[b, i].item()
            if s == float("-inf"):
                continue
            seq = beams[b * K + i].tolist()
            lp = ((5.0 + len(seq)) / 6.0) ** length_penalty
            finished[b].append((s / lp, seq))

    # Pick best hypothesis per sentence
    results: list[list[int]] = []
    for b in range(B):
        if not finished[b]:
            results.append([BOS_ID, EOS_ID])
        else:
            finished[b].sort(key=lambda x: x[0], reverse=True)
            results.append(finished[b][0][1])

    return results


def beam_search_translate(
    model: Transformer,
    tokenizer: Tokenizer,
    src_sentences: list[str],
    beam_size: int = 5,
    max_len: int = 256,
    length_penalty: float = 1.0,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 32,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[str]:
    """Translate a list of source sentences using batched beam search.

    Sentences are sorted by length for efficient batching, then restored
    to original order before returning.

    Args:
        model: Trained Transformer model.
        tokenizer: SentencePiece tokenizer.
        src_sentences: Source sentences (raw text).
        beam_size: Beam width.
        max_len: Max decoding length.
        length_penalty: Length penalty alpha.
        device: Computation device.
        batch_size: Number of sentences per batch.
        should_stop: Optional callable; if it returns True between batches,
            aborts decoding by raising TranslationInterrupted. Used to
            respond promptly to Ctrl+C during long evals.

    Returns:
        List of translated sentences in the original order.

    Raises:
        TranslationInterrupted: If should_stop() returned True mid-decode.
    """
    model.eval()
    n = len(src_sentences)

    # Tokenize all sentences
    src_ids_list = [tokenizer.encode(s) for s in src_sentences]

    # Sort by length (descending) for efficient batching - group similar lengths
    order = sorted(range(n), key=lambda i: len(src_ids_list[i]), reverse=True)
    sorted_ids = [src_ids_list[i] for i in order]

    # Decode in batches
    translations_sorted: list[str] = [""] * n
    for start in tqdm(range(0, n, batch_size), desc="Translating", leave=False):
        # Check for interrupt before starting each batch — keeps eval
        # responsive to Ctrl+C without leaving the GPU mid-step.
        if should_stop is not None and should_stop():
            raise TranslationInterrupted(
                f"Aborted at batch {start // batch_size + 1}/{(n + batch_size - 1) // batch_size}"
            )
        end = min(start + batch_size, n)
        batch = sorted_ids[start:end]

        # Pad to max length in this batch
        max_src = max(len(s) for s in batch)
        padded = [s + [PAD_ID] * (max_src - len(s)) for s in batch]
        src_tensor = torch.tensor(padded, dtype=torch.long, device=device)

        # Batched beam search
        output_ids_batch = batched_beam_search(
            model, src_tensor, beam_size, max_len, length_penalty
        )

        for local_i, output_ids in enumerate(output_ids_batch):
            original_idx = order[start + local_i]
            translations_sorted[original_idx] = tokenizer.decode(output_ids)

    return translations_sorted


@torch.no_grad()
def beam_search_decode(
    model: Transformer,
    src: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 256,
    length_penalty: float = 1.0,
) -> list[int]:
    """Convenience wrapper: beam search for a single source sequence.

    Args:
        src: (1, src_len) source token IDs.
    """
    results = batched_beam_search(model, src, beam_size, max_len, length_penalty)
    return results[0]
