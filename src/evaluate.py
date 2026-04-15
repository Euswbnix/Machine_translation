"""BLEU evaluation using sacrebleu."""

import sacrebleu


def compute_bleu(
    hypotheses: list[str],
    references: list[str],
    tgt_lang: str = "en",
) -> float:
    """Compute BLEU score using sacrebleu.

    Args:
        hypotheses: List of translated sentences.
        references: List of reference sentences.
        tgt_lang: Target language code. Determines tokenization:
            - "zh": character-level tokenization for Chinese references
            - anything else: sacrebleu default "13a" (for en, de, fr, etc.)

    Returns:
        BLEU score (0-100).
    """
    tokenize = "zh" if tgt_lang == "zh" else "13a"
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize)
    return bleu.score
