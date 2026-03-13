"""Evaluation metrics for point cloud registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EvalResult:
    rre: float                                # Relative Rotation Error (degrees)
    rte: float                                # Relative Translation Error (metres)
    success: bool                             # RRE < rre_thr AND RTE < rte_thr
    inlier_ratio: Optional[float] = None
    chamfer_distance: Optional[float] = None


def compute_rre(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Relative Rotation Error in degrees.

    RRE = arccos( (trace(R_est^T @ R_gt) - 1) / 2 )
    """
    R = R_est.T @ R_gt
    trace = np.trace(R)
    # Clamp to [-1, 1] to guard against floating-point drift
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_rte(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Relative Translation Error in metres (L2 norm)."""
    return float(np.linalg.norm(t_est - t_gt))


def compute_inlier_ratio(
    src_kps: np.ndarray,
    tgt_kps: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    T_gt: np.ndarray,
    voxel_size: float,
    distance_threshold_factor: float = 1.5,
) -> float:
    """Fraction of correspondences within threshold under ground-truth transform."""
    if len(src_idx) == 0:
        return 0.0

    threshold = voxel_size * distance_threshold_factor

    src_corr = src_kps[src_idx]    # (M, 3)
    tgt_corr = tgt_kps[tgt_idx]    # (M, 3)

    # Apply GT transform to source correspondences
    src_corr_hom = np.hstack([src_corr, np.ones((len(src_corr), 1))])  # (M, 4)
    src_transformed = (T_gt @ src_corr_hom.T).T[:, :3]                  # (M, 3)

    dists = np.linalg.norm(src_transformed - tgt_corr, axis=1)
    return float(np.mean(dists < threshold))


def compute_chamfer(src_xyz: np.ndarray, tgt_xyz: np.ndarray, T_est: np.ndarray) -> float:
    """Mean symmetric nearest-neighbour distance (Chamfer Distance) in metres.

    Applies T_est to src before computing distances.
    """
    from scipy.spatial import cKDTree

    # Apply estimated transform to source
    src_hom = np.hstack([src_xyz, np.ones((len(src_xyz), 1))])
    src_t = (T_est @ src_hom.T).T[:, :3]

    tree_tgt = cKDTree(tgt_xyz)
    tree_src = cKDTree(src_t)

    d_s2t, _ = tree_tgt.query(src_t, k=1, workers=-1)
    d_t2s, _ = tree_src.query(tgt_xyz, k=1, workers=-1)

    return float((d_s2t.mean() + d_t2s.mean()) / 2.0)


def evaluate_pair(
    T_est: np.ndarray,
    T_gt: np.ndarray,
    rre_threshold: float = 15.0,
    rte_threshold: float = 0.30,
    src_kps: Optional[np.ndarray] = None,
    tgt_kps: Optional[np.ndarray] = None,
    src_idx: Optional[np.ndarray] = None,
    tgt_idx: Optional[np.ndarray] = None,
    src_xyz: Optional[np.ndarray] = None,
    tgt_xyz: Optional[np.ndarray] = None,
    voxel_size: float = 0.05,
    compute_chamfer_dist: bool = False,
) -> EvalResult:
    """Compute all metrics for one estimated vs ground-truth transform pair."""
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]

    rre = compute_rre(R_est, R_gt)
    rte = compute_rte(t_est, t_gt)
    success = rre < rre_threshold and rte < rte_threshold

    ir = None
    if (
        src_kps is not None
        and tgt_kps is not None
        and src_idx is not None
        and tgt_idx is not None
        and len(src_idx) > 0
    ):
        ir = compute_inlier_ratio(src_kps, tgt_kps, src_idx, tgt_idx, T_gt, voxel_size)

    cd = None
    if compute_chamfer_dist and src_xyz is not None and tgt_xyz is not None:
        cd = compute_chamfer(src_xyz, tgt_xyz, T_est)

    return EvalResult(rre=rre, rte=rte, success=success, inlier_ratio=ir, chamfer_distance=cd)
