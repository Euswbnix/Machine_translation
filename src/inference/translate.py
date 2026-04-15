"""Beam search decoding for translation inference."""

import torch
from tqdm import tqdm

from src.model import Transformer
from src.data.tokenizer import Tokenizer, BOS_ID, EOS_ID, PAD_ID


def beam_search_decode(
    model: Transformer,
    src: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 256,
    length_penalty: float = 1.0,
) -> list[int]:
    """Beam search decoding for a single source sequence.

    Args:
        model: Trained Transformer model (in eval mode).
        src: (1, src_len) source token IDs.
        beam_size: Number of beams to keep.
        max_len: Maximum decoding length.
        length_penalty: Length normalization factor (alpha).

    Returns:
        Best hypothesis as a list of token IDs.
    """
    device = src.device

    # Encode source once
    src_mask = model.make_src_mask(src)
    enc_output = model.encode(src, src_mask)

    # Expand for beam search
    enc_output = enc_output.repeat(beam_size, 1, 1)
    src_mask = src_mask.repeat(beam_size, 1, 1, 1)

    # Initialize beams: (beam_size, 1) starting with BOS
    beams = torch.full((beam_size, 1), BOS_ID, dtype=torch.long, device=device)
    beam_scores = torch.zeros(beam_size, device=device)
    beam_scores[1:] = float("-inf")  # only first beam active initially

    finished = []

    for step in range(max_len):
        tgt_mask = model.make_tgt_mask(beams)
        dec_output = model.decode(beams, enc_output, tgt_mask, src_mask)

        # Get logits for last position only
        logits = model.output_proj(dec_output[:, -1, :])  # (beam_size, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Score all possible next tokens
        vocab_size = log_probs.size(-1)
        next_scores = beam_scores.unsqueeze(-1) + log_probs  # (beam, vocab)
        next_scores = next_scores.view(-1)  # (beam * vocab,)

        # Select top-k
        topk_scores, topk_indices = next_scores.topk(beam_size * 2)
        beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size

        # Build new beams
        new_beams = []
        new_scores = []

        for score, beam_idx, token_idx in zip(topk_scores, beam_indices, token_indices):
            beam_idx = beam_idx.item()
            token_idx = token_idx.item()

            if token_idx == EOS_ID:
                # Finished beam - apply length penalty
                seq_len = beams.size(1)
                lp = ((5.0 + seq_len) / 6.0) ** length_penalty
                finished.append((score.item() / lp, beams[beam_idx].tolist() + [EOS_ID]))
            else:
                new_beams.append(
                    torch.cat([beams[beam_idx], torch.tensor([token_idx], device=device)])
                )
                new_scores.append(score)

            if len(new_beams) == beam_size:
                break

        if not new_beams:
            break

        beams = torch.stack(new_beams)
        beam_scores = torch.stack(new_scores)

    # Add remaining active beams to finished
    for i in range(beams.size(0)):
        seq_len = beams.size(1)
        lp = ((5.0 + seq_len) / 6.0) ** length_penalty
        finished.append((beam_scores[i].item() / lp, beams[i].tolist()))

    # Return best
    finished.sort(key=lambda x: x[0], reverse=True)
    return finished[0][1]


def beam_search_translate(
    model: Transformer,
    tokenizer: Tokenizer,
    src_sentences: list[str],
    beam_size: int = 5,
    max_len: int = 256,
    length_penalty: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> list[str]:
    """Translate a list of source sentences using beam search.

    Args:
        model: Trained Transformer model.
        tokenizer: SentencePiece tokenizer.
        src_sentences: Source sentences (raw text).
        beam_size: Beam width.
        max_len: Max decoding length.
        length_penalty: Length penalty alpha.
        device: Device for computation.

    Returns:
        List of translated sentences.
    """
    model.eval()
    translations = []

    for sentence in tqdm(src_sentences, desc="Translating", leave=False):
        src_ids = tokenizer.encode(sentence)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

        output_ids = beam_search_decode(
            model, src_tensor, beam_size, max_len, length_penalty
        )

        translation = tokenizer.decode(output_ids)
        translations.append(translation)

    return translations
