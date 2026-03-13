"""Z-translation correction for outdoor scenes with vertical facades.

FPFH correspondences on building walls are geometrically consistent at any
height (wall normals are horizontal → no Z gradient in point-to-plane ICP).
This module corrects the Z component of T_est after RANSAC/ICP.

Two strategies:
  - 'ground_percentile': align low-percentile Z (≈ ground level).  Fast,
    works when ground points are visible in both clouds.
  - 'histogram_xcorr': 1-D height histogram cross-correlation over a search
    window.  More robust when one cloud has fewer ground points or when the
    height offset is large.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def correct_z(
    src_xyz: np.ndarray,
    tgt_xyz: np.ndarray,
    T_est: np.ndarray,
    method: str = "ground_percentile",
    z_axis: int = 2,
    percentile: float = 5.0,
    bin_size: float = 0.1,
    search_range: float = 5.0,
) -> np.ndarray:
    """Correct the Z-translation component of T_est.

    Args:
        src_xyz:      Source point cloud XYZ (N, 3).
        tgt_xyz:      Target point cloud XYZ (M, 3).
        T_est:        Current 4×4 transform estimate (src → tgt).
        method:       'ground_percentile' or 'histogram_xcorr'.
        z_axis:       Which axis is vertical (0=X, 1=Y, 2=Z).
        percentile:   Percentile used as ground proxy (ground_percentile only).
        bin_size:     Histogram bin size in metres (histogram_xcorr only).
        search_range: ±metres to search for Z offset (histogram_xcorr only).

    Returns:
        Corrected 4×4 transform (copy of T_est with updated Z translation).
    """
    # Transform source points with current estimate
    ones = np.ones((len(src_xyz), 1), dtype=np.float64)
    src_hom = np.hstack([src_xyz.astype(np.float64), ones])
    src_t = (T_est @ src_hom.T).T[:, :3]

    src_z = src_t[:, z_axis]
    tgt_z = tgt_xyz[:, z_axis].astype(np.float64)

    if method == "ground_percentile":
        dz = _ground_percentile(src_z, tgt_z, percentile)
    elif method == "histogram_xcorr":
        dz = _histogram_xcorr(src_z, tgt_z, bin_size, search_range)
    else:
        raise ValueError(f"Unknown z_correction method: '{method}'")

    if abs(dz) < 1e-4:
        log.info("Z correction: Δz=%.4f m (negligible — skipping)", dz)
        return T_est

    T_corrected = T_est.copy()
    T_corrected[z_axis, 3] += dz
    log.info(
        "Z correction (%s): Δz=%+.3f m  (z_before=%.3f → z_after=%.3f)",
        method, dz, float(T_est[z_axis, 3]), float(T_corrected[z_axis, 3]),
    )
    return T_corrected


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ground_percentile(src_z: np.ndarray, tgt_z: np.ndarray, pct: float) -> float:
    """Align the pct-th percentile of Z (proxy for ground level)."""
    z_src_ground = float(np.percentile(src_z, pct))
    z_tgt_ground = float(np.percentile(tgt_z, pct))
    dz = z_tgt_ground - z_src_ground
    log.debug(
        "  Z ground_percentile p%.0f: src_ground=%.3f  tgt_ground=%.3f  Δz=%.3f",
        pct, z_src_ground, z_tgt_ground, dz,
    )
    return dz


def _histogram_xcorr(
    src_z: np.ndarray,
    tgt_z: np.ndarray,
    bin_size: float,
    search_range: float,
) -> float:
    """Find Z-shift by cross-correlating 1-D height histograms."""
    z_min = min(src_z.min(), tgt_z.min()) - search_range
    z_max = max(src_z.max(), tgt_z.max()) + search_range
    bins = np.arange(z_min, z_max + bin_size, bin_size)

    h_src, _ = np.histogram(src_z, bins=bins)
    h_tgt, _ = np.histogram(tgt_z, bins=bins)

    corr = np.correlate(h_tgt.astype(float), h_src.astype(float), mode="full")
    best_lag = int(np.argmax(corr)) - (len(h_src) - 1)
    dz = best_lag * bin_size
    log.debug(
        "  Z histogram_xcorr: best_lag=%d bins  Δz=%.3f m  peak_corr=%.0f",
        best_lag, dz, float(corr[np.argmax(corr)]),
    )
    return dz
