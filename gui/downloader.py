"""HuggingFace model file downloader with progress reporting.

Files are cached under ~/.cache/transformer-mt/<repo-id>/.
Subsequent launches with the same repo reuse the cache.
"""

from pathlib import Path
from typing import Callable, List, Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


CACHE_ROOT = Path.home() / ".cache" / "transformer-mt"


def cache_dir_for(repo_id: str) -> Path:
    """Return per-repo cache directory."""
    safe = repo_id.replace("/", "__")
    return CACHE_ROOT / safe


def is_cached(repo_id: str, files: List[str]) -> bool:
    """All required files present locally?"""
    d = cache_dir_for(repo_id)
    return d.exists() and all((d / f).exists() for f in files)


def download_model(
    repo_id: str,
    files: List[str],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Path:
    """Download all required files for a model. Returns local cache dir.

    progress_cb(file_index, file_total, filename) lets the UI show
    "downloading 2/3: pytorch_model.bin" style updates.
    """
    out = cache_dir_for(repo_id)
    out.mkdir(parents=True, exist_ok=True)

    for i, fname in enumerate(files):
        if progress_cb is not None:
            progress_cb(i, len(files), fname)
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=str(out),
                local_dir_use_symlinks=False,
            )
        except HfHubHTTPError as e:
            raise RuntimeError(
                f"Failed to download {fname} from {repo_id}: {e}\n"
                "Check your internet connection or HuggingFace status."
            ) from e
        # huggingface_hub caches under .huggingface/ subdir if local_dir_use_symlinks
        # is True; we forced False so file is at out/fname directly.
    if progress_cb is not None:
        progress_cb(len(files), len(files), "done")
    return out
