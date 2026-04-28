"""Registry of available models, mapped to their HuggingFace repos.

Default catalogue uses base (60M) FP32 weights — small and fast to download.
FT (fine-tuned) entries are populated by the Machine-Translation-FT
extension at startup if its package is importable; otherwise they remain
hidden in the dropdown.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelEntry:
    key: str                       # internal id, e.g. "enfr_base_v11"
    label: str                     # shown in dropdown
    hf_repo: str                   # HuggingFace repo id
    src_lang: str                  # "en"
    tgt_lang: str                  # "fr" or "de"
    arch: str                      # "base" or "big"
    size_mb_estimate: int          # for download dialog text
    is_ft: bool = False            # Fine-tuned variant?
    files: List[str] = field(default_factory=lambda: [
        "pytorch_model.bin",
        "sentencepiece.model",
        "config.json",
    ])


# Default catalogue — the four pretrained checkpoints we published.
# Order matters: first entry is the default selection on first launch.
BASE_MODELS: List[ModelEntry] = [
    ModelEntry(
        key="enfr_base_v11",
        label="English → French (Base v1.1, 60M)",
        hf_repo="euswbnix/transformer-wmt14-enfr-base",
        src_lang="en", tgt_lang="fr", arch="base",
        size_mb_estimate=240,
    ),
    ModelEntry(
        key="ende_base",
        label="English → German (Base, 60M)",
        hf_repo="euswbnix/transformer-wmt14-ende-base",
        src_lang="en", tgt_lang="de", arch="base",
        size_mb_estimate=240,
    ),
    ModelEntry(
        key="enfr_big_v11",
        label="English → French (Big v1.1, 209M)",
        hf_repo="euswbnix/transformer-wmt14-enfr-big",
        src_lang="en", tgt_lang="fr", arch="big",
        size_mb_estimate=836,
    ),
    ModelEntry(
        key="ende_big",
        label="English → German (Big, 209M)",
        hf_repo="euswbnix/transformer-wmt14-ende-big",
        src_lang="en", tgt_lang="de", arch="big",
        size_mb_estimate=836,
    ),
]


def get_models(include_ft: bool = False) -> List[ModelEntry]:
    """Return the available models. FT models are appended only if requested
    AND the FT extension package is importable (lazy)."""
    models = list(BASE_MODELS)
    if include_ft:
        try:
            from gui_ft_extension.ft_models import FT_MODELS  # type: ignore
            models.extend(FT_MODELS)
        except ImportError:
            pass
    return models


def find_by_key(key: str, include_ft: bool = True) -> Optional[ModelEntry]:
    for m in get_models(include_ft=include_ft):
        if m.key == key:
            return m
    return None


DEFAULT_MODEL_KEY = "enfr_base_v11"
