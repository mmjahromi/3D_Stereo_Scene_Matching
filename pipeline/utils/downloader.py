"""Pretrained weight download helpers."""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

WEIGHT_REGISTRY = {
    "fcgf": {
        "url": "http://node2.chrischoy.org/data/projects/FCGF/2019-09-18_09-21-23.pth",
        "filename": "fcgf_3dmatch.pth",
        "md5": None,  # not published
        "description": "FCGF 3DMatch checkpoint (ResUNetBN2C, D=32)",
    },
    "geotransformer": {
        "url": (
            "https://github.com/qinzheng93/GeoTransformer/releases/download/"
            "v1.0.0/geotransformer-3dmatch.pth.tar"
        ),
        "filename": "geotransformer-3dmatch.pth.tar",
        "md5": None,
        "description": "GeoTransformer 3DMatch checkpoint",
    },
    "predator": {
        "url": None,  # Must be fetched from prs-eth/OverlapPredator repo
        "filename": "predator_3dmatch.pth",
        "md5": None,
        "description": (
            "OverlapPredator 3DMatch checkpoint. "
            "Download manually from https://github.com/prs-eth/OverlapPredator/releases"
        ),
    },
}


def download_weights(
    models: Optional[list] = None,
    weights_dir: str = "weights",
) -> None:
    """Download pretrained weights for the requested models.

    Args:
        models:      List of model names, e.g. ["fcgf", "geotransformer"].
                     If None, downloads all available weights.
        weights_dir: Target directory.
    """
    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)

    targets = models if models is not None else list(WEIGHT_REGISTRY.keys())

    for model_name in targets:
        if model_name not in WEIGHT_REGISTRY:
            log.warning(f"No weight registry entry for '{model_name}'. Skipping.")
            continue

        entry = WEIGHT_REGISTRY[model_name]
        dest = weights_path / entry["filename"]

        if dest.exists():
            log.info(f"[{model_name}] Already downloaded: {dest}")
            continue

        if entry["url"] is None:
            print(
                f"[{model_name}] Manual download required:\n"
                f"  {entry['description']}\n"
                f"  Place file at: {dest}\n"
            )
            continue

        print(f"[{model_name}] Downloading {entry['filename']} …")
        _download_file(entry["url"], str(dest), md5=entry["md5"])
        print(f"[{model_name}] Saved to {dest}")


def _download_file(url: str, dest: str, md5: Optional[str] = None) -> None:
    """Download url → dest with a progress hook, optionally verify MD5."""
    tmp = dest + ".tmp"

    def _reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 / total_size)
            print(f"\r  {pct:5.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
        print()  # newline after progress
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc

    if md5 is not None:
        actual = _md5(tmp)
        if actual != md5:
            os.remove(tmp)
            raise ValueError(
                f"MD5 mismatch for {dest}: expected {md5}, got {actual}"
            )

    os.rename(tmp, dest)


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
