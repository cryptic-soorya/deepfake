"""Shared helper for lazily downloading pretrained checkpoints that have no
pip-installable package (MiniFASNet, SyncNet) — mirrors the pattern
insightface already uses for its own model packs: fetch on first run, cache
under ~/.cache, skip the download on subsequent loads.
"""
import os
import urllib.request
from pathlib import Path

CACHE_DIR = Path(os.environ.get("MORPHEUS_WEIGHTS_CACHE", Path.home() / ".cache" / "morpheus" / "weights"))


def fetch(url: str, filename: str | None = None) -> Path:
    """Download `url` into the weights cache if not already present, return the local path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    name = filename or url.rsplit("/", 1)[-1]
    dest = CACHE_DIR / name
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def fetch_gdrive(file_id: str, filename: str) -> Path:
    """Download a Google Drive share-link file into the same weights cache.

    Google Drive share links need gdown (it handles the interstitial "can't
    scan this file for viruses" confirmation-token page that larger files
    trigger) rather than a plain urllib GET like `fetch()` above uses for
    raw-githubusercontent URLs.
    """
    import gdown

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / filename
    if not dest.exists():
        gdown.download(id=file_id, output=str(dest), quiet=False)
    return dest
