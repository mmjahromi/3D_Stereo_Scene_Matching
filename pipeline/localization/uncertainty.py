"""Multi-run RANSAC uncertainty estimation for North/East position."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from pipeline.models.base import FeatureOutput

log = logging.getLogger(__name__)

_AXIS_MAP = {"x": 0, "y": 1, "z": 2}


def estimate_ne_uncertainty(
    src_feat: FeatureOutput,
    tgt_feat: FeatureOutput,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    voxel_size: float,
    n_runs: int = 10,
    ransac_cfg: Optional[dict] = None,
    east_axis: str = "x",
    north_axis: str = "y",
) -> Tuple[np.ndarray, float, float, float]:
    """Run RANSAC n_runs times; return (T_best, sigma_east, sigma_north, best_inlier_ratio).

    Args:
        src_feat:    Source features.
        tgt_feat:    Target features.
        src_idx:     Correspondence indices into src_feat.keypoints.
        tgt_idx:     Correspondence indices into tgt_feat.keypoints.
        voxel_size:  Voxel size (metres).
        n_runs:      Number of independent RANSAC trials.
        ransac_cfg:  Dict with keys max_iterations, confidence, ransac_n,
                     distance_threshold_factor.
        east_axis:   Which xyz axis maps to East ("x" | "y" | "z").
        north_axis:  Which xyz axis maps to North ("x" | "y" | "z").

    Returns:
        T_best            4×4 transform with most inliers.
        sigma_east        Std-dev of translation East component (metres).
        sigma_north       Std-dev of translation North component (metres).
        best_inlier_ratio Fraction of correspondences that are inliers for T_best.
    """
    import open3d as o3d

    from pipeline.matching.matcher import ransac_registration

    if ransac_cfg is None:
        ransac_cfg = {}

    east_idx = _AXIS_MAP[east_axis]
    north_idx = _AXIS_MAP[north_axis]

    dist_thr = voxel_size * ransac_cfg.get("distance_threshold_factor", 1.5)
    max_iter = ransac_cfg.get("max_iterations", 1_000_000)
    confidence = ransac_cfg.get("confidence", 0.999)
    ransac_n = ransac_cfg.get("ransac_n", 3)

    if len(src_idx) < ransac_n:
        log.warning(
            f"Too few correspondences ({len(src_idx)}) for uncertainty estimation. "
            "Returning identity with voxel_size uncertainty."
        )
        return np.eye(4), voxel_size, voxel_size, 0.0

    # Build O3D point clouds once for evaluation
    src_pcd_o3d = o3d.geometry.PointCloud()
    src_pcd_o3d.points = o3d.utility.Vector3dVector(
        src_feat.keypoints.astype(np.float64)
    )
    tgt_pcd_o3d = o3d.geometry.PointCloud()
    tgt_pcd_o3d.points = o3d.utility.Vector3dVector(
        tgt_feat.keypoints.astype(np.float64)
    )

    translations = []
    inlier_counts = []
    transforms = []

    for run_seed in range(n_runs):
        T = ransac_registration(
            src_feat.keypoints,
            tgt_feat.keypoints,
            src_idx,
            tgt_idx,
            voxel_size=voxel_size,
            max_iterations=max_iter,
            confidence=confidence,
            ransac_n=ransac_n,
            distance_threshold_factor=ransac_cfg.get("distance_threshold_factor", 1.5),
            seed=run_seed,
        )

        # Count inliers via ICP evaluate (cheap)
        eval_result = o3d.pipelines.registration.evaluate_registration(
            src_pcd_o3d, tgt_pcd_o3d, dist_thr, T
        )
        inliers = int(eval_result.correspondence_set.__len__())
        translations.append(T[:3, 3].copy())
        inlier_counts.append(inliers)
        transforms.append(T)

    inlier_counts = np.array(inlier_counts)
    best_run = int(np.argmax(inlier_counts))
    max_inliers = inlier_counts[best_run]

    log.info(
        "RANSAC uncertainty runs: max_inliers=%d, counts=%s",
        max_inliers,
        inlier_counts.tolist(),
    )

    if max_inliers == 0:
        log.warning(
            "All %d RANSAC runs found 0 inliers — correspondences have no "
            "geometric consistency. Check voxel_size / distance_threshold_factor.",
            n_runs,
        )
        return np.eye(4), voxel_size, voxel_size, 0.0

    # Keep runs with inlier_count >= max_inliers * 0.5
    threshold = max_inliers * 0.5
    valid_mask = inlier_counts >= threshold
    valid_translations = np.array(translations)[valid_mask]

    # Robust sigma: discard runs whose translation is far from the best run.
    # Failed RANSAC runs can produce translations millions of meters away
    # (random correspondences that coincidentally satisfy the threshold count).
    best_t = np.array(translations[best_run])
    dists_from_best = np.linalg.norm(valid_translations - best_t, axis=1)
    # Accept runs within 50 m (or 200× voxel_size if larger) of the best run
    search_radius = max(50.0, voxel_size * 200)
    close_mask = dists_from_best < search_radius
    consistent_translations = valid_translations[close_mask]

    log.info(
        "Sigma estimation: %d/%d valid runs within %.1f m of best translation",
        int(close_mask.sum()), len(valid_translations), search_radius,
    )

    if len(consistent_translations) >= 2:
        sigma_east = float(np.std(consistent_translations[:, east_idx]))
        sigma_north = float(np.std(consistent_translations[:, north_idx]))
    else:
        # Only one consistent run — can't compute spread; use voxel_size as proxy
        sigma_east = voxel_size
        sigma_north = voxel_size

    T_best = transforms[best_run]
    # Inlier ratio relative to the number of source keypoints
    n_src_pts = len(src_feat.keypoints)
    best_inlier_ratio = min(max_inliers / max(n_src_pts, 1), 1.0)

    if best_inlier_ratio < 0.05:
        log.warning(
            "Low inlier ratio: %.1f%% (%d/%d). Registration likely unreliable. "
            "Consider increasing voxel_size or checking point cloud overlap.",
            best_inlier_ratio * 100, max_inliers, n_src_pts,
        )

    log.info(
        "Best RANSAC: inliers=%d (%.1f%%)  T_translation=[%.3f, %.3f, %.3f]  "
        "σ_E=%.3f m  σ_N=%.3f m",
        max_inliers, best_inlier_ratio * 100,
        T_best[0, 3], T_best[1, 3], T_best[2, 3],
        sigma_east, sigma_north,
    )

    return T_best, sigma_east, sigma_north, best_inlier_ratio
