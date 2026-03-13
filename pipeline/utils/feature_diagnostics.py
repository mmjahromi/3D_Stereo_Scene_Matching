"""Feature discriminability diagnostics using nearest-neighbour ratio analysis."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from pipeline.models.base import FeatureOutput

log = logging.getLogger(__name__)

# Thresholds for the mean NN-ratio rating
_RATING_THRESHOLDS = [
    (0.70, "good", "Features are discriminative — RANSAC should work well."),
    (0.80, "ok",   "Moderate discriminability — RANSAC may produce some false positives."),
    (0.90, "poor", "Low discriminability — consider color-assisted filtering or learned features (FCGF/GeoTransformer)."),
    (1.01, "bad",  "Very ambiguous features — RANSAC will likely fail. Strongly recommend learned features."),
]


def check_discriminability(
    feat: FeatureOutput,
    label: str = "",
    max_sample: int = 3000,
) -> dict:
    """Compute NN-ratio distribution to measure descriptor discriminability.

    For each descriptor d_i, computes ratio = dist(d_nn1) / dist(d_nn2).
    A ratio near 1 means many similar descriptors → ambiguous → RANSAC-hostile.
    A ratio near 0 means a clear nearest neighbour → discriminative.

    Args:
        feat:       FeatureOutput from extract_features().
        label:      Display label (e.g. "src" or "tgt").
        max_sample: Subsample to this many descriptors for speed on large clouds.

    Returns:
        dict with keys: mean_ratio, pct_ambiguous, rating, suggestion.
    """
    from scipy.spatial import cKDTree

    descs = feat.descriptors  # (N, D)
    N = len(descs)

    if N < 3:
        return {"mean_ratio": float("nan"), "pct_ambiguous": float("nan"),
                "rating": "N/A", "suggestion": "Too few points."}

    # Subsample for speed without losing statistical accuracy
    if N > max_sample:
        idx = np.random.choice(N, max_sample, replace=False)
        descs_sample = descs[idx]
    else:
        descs_sample = descs

    # Build tree over all descriptors; query sample points for 2 nearest neighbours.
    # When sample IS the full set, each query point is in the tree (self-match at d=0),
    # so we need k=3 and skip index 0. When subsampled, no self-match → k=2 suffices.
    tree = cKDTree(descs)
    in_tree = descs_sample is descs
    dists, _ = tree.query(descs_sample, k=3 if in_tree else 2)

    if in_tree:
        d1 = dists[:, 1]  # skip self at index 0
        d2 = dists[:, 2]
    else:
        d1 = dists[:, 0]
        d2 = dists[:, 1]

    valid = d2 > 1e-10
    ratio = np.where(valid, d1 / np.where(valid, d2, 1.0), 1.0)

    mean_ratio = float(ratio.mean())
    pct_ambiguous = float((ratio > 0.85).mean() * 100)

    rating = "bad"
    suggestion = _RATING_THRESHOLDS[-1][2]
    for threshold, r, s in _RATING_THRESHOLDS:
        if mean_ratio < threshold:
            rating = r
            suggestion = s
            break

    result = {
        "mean_ratio": mean_ratio,
        "pct_ambiguous": pct_ambiguous,
        "rating": rating,
        "suggestion": suggestion,
    }

    _log_result(label, N, result)
    return result


def _log_result(label: str, n_points: int, r: dict) -> None:
    prefix = f"  [{label}]" if label else " "
    rating_symbols = {"good": "✓", "ok": "~", "poor": "!", "bad": "✗", "N/A": "?"}
    sym = rating_symbols.get(r["rating"], "?")

    line = (
        f"{prefix} discriminability: mean_NN_ratio={r['mean_ratio']:.3f}  "
        f"pct_ambiguous={r['pct_ambiguous']:.1f}%  "
        f"rating={sym} {r['rating'].upper()}  n={n_points}"
    )

    if r["rating"] in ("poor", "bad"):
        log.warning(line)
        log.warning(f"  {prefix}   → {r['suggestion']}")
    else:
        log.info(line)
