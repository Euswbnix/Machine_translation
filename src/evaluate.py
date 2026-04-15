"""BLEU evaluation using sacrebleu."""

import sacrebleu


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Compute BLEU score using sacrebleu (detokenized).

    Args:
        hypotheses: List of translated sentences.
        references: List of reference sentences.

    Returns:
        BLEU score (0-100).
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="zh")
    return bleu.score
