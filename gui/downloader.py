"""HuggingFace model file downloader with byte-level progress reporting.

Files are cached under ~/.cache/transformer-mt/<repo-id>/.
We bypass hf_hub_download's tqdm-only progress and stream files via
requests so the UI can show megabytes downloaded vs total in real time.
"""

from pathlib import Path
from typing import Callable, List, Optional

import requests
from huggingface_hub import hf_hub_url


CACHE_ROOT = Path.home() / ".cache" / "transformer-mt"
CHUNK_SIZE = 64 * 1024              # 64 KB; ~3700 chunks for a 240 MB file


def cache_dir_for(repo_id: str) -> Path:
    """Return per-repo cache directory."""
    safe = repo_id.replace("/", "__")
    return CACHE_ROOT / safe


def is_cached(repo_id: str, files: List[str]) -> bool:
    """All required files present locally?"""
    d = cache_dir_for(repo_id)
    return d.exists() and all((d / f).exists() and (d / f).stat().st_size > 0
                              for f in files)


def _stream_one_file(
    repo_id: str,
    filename: str,
    dest: Path,
    progress_cb: Optional[Callable[[int, int, str], None]],
) -> None:
    """Stream a single file with byte-level progress."""
    url = hf_hub_url(repo_id=repo_id, filename=filename, revision="main")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, allow_redirects=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        progress_cb(downloaded, total, filename)
    tmp.replace(dest)                # atomic move on success


def download_model(
    repo_id: str,
    files: List[str],
    progress_cb: Optional[Callable[[int, int, int, int, str], None]] = None,
) -> Path:
    """Download all required files for a model. Returns local cache dir.

    progress_cb signature:
        (file_idx, file_total, bytes_done, bytes_total, filename)
    bytes_done / bytes_total are for the CURRENT file (resets per file).
    """
    out = cache_dir_for(repo_id)
    out.mkdir(parents=True, exist_ok=True)

    for i, fname in enumerate(files):
        dest = out / fname
        if dest.exists() and dest.stat().st_size > 0:
            if progress_cb is not None:
                size = dest.stat().st_size
                progress_cb(i, len(files), size, size, fname)
            continue

        def _byte_cb(done: int, total: int, name: str) -> None:
            if progress_cb is not None:
                progress_cb(i, len(files), done, total, name)

        try:
            _stream_one_file(repo_id, fname, dest, _byte_cb)
        except requests.HTTPError as e:
            raise RuntimeError(
                f"HTTP error downloading {fname} from {repo_id}: {e}"
            ) from e
        except requests.RequestException as e:
            raise RuntimeError(
                f"Network error downloading {fname} from {repo_id}: {e}\n"
                "Check your internet connection or try again."
            ) from e
    return out
