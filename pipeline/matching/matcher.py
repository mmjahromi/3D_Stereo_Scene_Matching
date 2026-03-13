"""Feature matching and RANSAC/ICP registration."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from pipeline.models.base import FeatureOutput, PointCloud

log = logging.getLogger(__name__)


def match_features(
    src_feat: FeatureOutput,
    tgt_feat: FeatureOutput,
    method: str = "mutual_nn",
    ratio_threshold: float = 0.80,
    overlap_threshold: Optional[float] = None,
    use_faiss: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find correspondences between two sets of descriptors.

    Args:
        src_feat:          Source features.
        tgt_feat:          Target features.
        method:            "mutual_nn" | "ratio_test" | "combined"
        ratio_threshold:   Lowe ratio threshold (used with "ratio_test" / "combined").
        overlap_threshold: If set, filter src by src_feat.scores > threshold.
        use_faiss:         Use faiss for NN search (falls back to scipy cKDTree).

    Returns:
        src_idx, tgt_idx: (M,) correspondence index arrays.
    """
    src_desc = src_feat.descriptors  # (N, D)
    tgt_desc = tgt_feat.descriptors  # (M, D)

    # Optional: filter source by overlap/matchability scores (Predator)
    src_mask = np.ones(len(src_desc), dtype=bool)
    if overlap_threshold is not None and src_feat.scores is not None:
        src_mask = src_feat.scores > overlap_threshold

    src_desc_filtered = src_desc[src_mask]

    if len(src_desc_filtered) == 0 or len(tgt_desc) == 0:
        log.warning("Empty descriptor arrays — returning no correspondences.")
        return np.array([], dtype=int), np.array([], dtype=int)

    nn_fn = _faiss_nn if use_faiss else _scipy_nn

    if method == "mutual_nn":
        si, ti = _mutual_nn(src_desc_filtered, tgt_desc, nn_fn)
    elif method == "ratio_test":
        si, ti = _ratio_test(src_desc_filtered, tgt_desc, nn_fn, ratio_threshold)
    elif method == "combined":
        si1, ti1 = _mutual_nn(src_desc_filtered, tgt_desc, nn_fn)
        si2, ti2 = _ratio_test(src_desc_filtered, tgt_desc, nn_fn, ratio_threshold)
        # Intersection
        pairs1 = set(zip(si1.tolist(), ti1.tolist()))
        pairs2 = set(zip(si2.tolist(), ti2.tolist()))
        common = np.array(list(pairs1 & pairs2), dtype=int)
        if len(common) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        si, ti = common[:, 0], common[:, 1]
    else:
        raise ValueError(f"Unknown matching method: '{method}'")

    # Map filtered indices back to original src indices
    orig_indices = np.where(src_mask)[0]
    si_orig = orig_indices[si]
    return si_orig, ti


# ---------------------------------------------------------------------------
# Low-level NN helpers
# ---------------------------------------------------------------------------

def _scipy_nn(query: np.ndarray, ref: np.ndarray, k: int = 2):
    """Return (indices, distances) of k nearest neighbours using cKDTree."""
    from scipy.spatial import cKDTree

    tree = cKDTree(ref)
    dists, indices = tree.query(query, k=k, workers=-1)
    return indices, dists


def _faiss_nn(query: np.ndarray, ref: np.ndarray, k: int = 2):
    """Return (indices, distances) using faiss (L2). Falls back to scipy."""
    try:
        import faiss

        d = ref.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(ref.astype(np.float32))
        dists, indices = index.search(query.astype(np.float32), k)
        return indices, np.sqrt(dists)  # faiss returns squared L2
    except ImportError:
        log.warning("faiss not installed — falling back to scipy cKDTree.")
        return _scipy_nn(query, ref, k=k)


def _mutual_nn(src: np.ndarray, tgt: np.ndarray, nn_fn) -> Tuple[np.ndarray, np.ndarray]:
    """Mutual nearest-neighbour matching."""
    src_to_tgt, _ = nn_fn(src, tgt, k=1)
    tgt_to_src, _ = nn_fn(tgt, src, k=1)

    src_to_tgt = src_to_tgt.flatten()
    tgt_to_src = tgt_to_src.flatten()

    si = np.arange(len(src))
    mutual = tgt_to_src[src_to_tgt[si]] == si
    si = si[mutual]
    ti = src_to_tgt[si]
    return si, ti


def _ratio_test(
    src: np.ndarray, tgt: np.ndarray, nn_fn, ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Lowe ratio test: d1/d2 < ratio."""
    indices, dists = nn_fn(src, tgt, k=2)
    if indices.ndim == 1:
        # Only 1 point in target
        return np.arange(len(src)), indices.flatten()
    d1 = dists[:, 0]
    d2 = dists[:, 1]
    mask = d1 < ratio * d2
    si = np.where(mask)[0]
    ti = indices[si, 0]
    return si, ti


# ---------------------------------------------------------------------------
# FGR registration (robust alternative to RANSAC for noisy/sparse clouds)
# ---------------------------------------------------------------------------

def fgr_registration(
    src_kps: np.ndarray,
    tgt_kps: np.ndarray,
    src_desc: np.ndarray,
    tgt_desc: np.ndarray,
    voxel_size: float,
    distance_threshold_factor: float = 1.5,
) -> np.ndarray:
    """Fast Global Registration using full descriptor arrays.

    Unlike RANSAC, FGR uses all correspondences jointly with a Geman-McClure
    robust loss — no hard inlier/outlier threshold, more robust to the depth
    noise typical of stereo-derived sparse point clouds.

    Args:
        src_kps:   Source keypoints (N, 3).
        tgt_kps:   Target keypoints (M, 3).
        src_desc:  Source FPFH descriptors (N, D).
        tgt_desc:  Target FPFH descriptors (M, D).
        voxel_size:                  Voxel size (metres).
        distance_threshold_factor:   Correspondence cutoff = voxel_size * factor.

    Returns:
        4×4 rigid transform (float64).
    """
    import open3d as o3d

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_kps.astype(np.float64))
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_kps.astype(np.float64))

    # O3D Feature stores descriptors as (D, N) — transpose from (N, D)
    src_feat = o3d.pipelines.registration.Feature()
    src_feat.data = src_desc.T.astype(np.float64)
    tgt_feat = o3d.pipelines.registration.Feature()
    tgt_feat.data = tgt_desc.T.astype(np.float64)

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_pcd,
        tgt_pcd,
        src_feat,
        tgt_feat,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel_size * distance_threshold_factor,
        ),
    )
    log.debug(
        "FGR: fitness=%.4f  inlier_rmse=%.4f",
        result.fitness,
        result.inlier_rmse,
    )
    return np.asarray(result.transformation, dtype=np.float64)


# ---------------------------------------------------------------------------
# RANSAC registration
# ---------------------------------------------------------------------------

def ransac_registration(
    src_kps: np.ndarray,
    tgt_kps: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    voxel_size: float,
    max_iterations: int = 1_000_000,
    confidence: float = 0.999,
    ransac_n: int = 3,
    distance_threshold_factor: float = 1.5,
    seed: int = 42,
) -> np.ndarray:
    """RANSAC rigid registration from point correspondences.

    Returns estimated 4×4 transform T_est (float64).
    """
    import open3d as o3d

    np.random.seed(seed)
    o3d.utility.random.seed(seed)

    if len(src_idx) < ransac_n:
        log.warning(
            f"Too few correspondences ({len(src_idx)}) for RANSAC — returning identity."
        )
        return np.eye(4)

    corr = np.column_stack([src_idx, tgt_idx]).astype(np.int32)
    corr_o3d = o3d.utility.Vector2iVector(corr)

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_kps.astype(np.float64))

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_kps.astype(np.float64))

    dist_thr = voxel_size * distance_threshold_factor

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd,
        tgt_pcd,
        corr_o3d,
        max_correspondence_distance=dist_thr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thr),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=max_iterations, confidence=confidence
        ),
    )
    log.debug(
        "RANSAC: fitness=%.4f  inlier_rmse=%.4f  n_corr_in=%d",
        result.fitness,
        result.inlier_rmse,
        len(result.correspondence_set),
    )

    return np.array(result.transformation, dtype=np.float64)


# ---------------------------------------------------------------------------
# Multi-scale GICP (for same-CRS data: UTM local_scan + UTM global_map)
# ---------------------------------------------------------------------------

def multiscale_gicp(
    src: PointCloud,
    tgt: PointCloud,
    T_init: np.ndarray,
    coarse_dist: float = 5.0,
    fine_dist: float = 0.5,
    n_scales: int = 4,
    max_iter: int = 300,
    translation_only: bool = False,
) -> np.ndarray:
    """Multi-scale GICP for same-coordinate-system clouds (e.g. both in UTM).

    Runs GICP at decreasing correspondence distances, starting from T_init.
    Internally shifts both clouds to local frame (src centroid = origin) to
    avoid floating-point precision issues with large absolute coordinates
    (UTM, ECEF, …).  The returned transform is in the original coordinate frame.

    Args:
        src:               Source PointCloud (must have normals).
        tgt:               Target PointCloud (must have normals).
        T_init:            Initial 4×4 transform (e.g. centroid alignment or identity).
        coarse_dist:       Starting correspondence distance (metres).
        fine_dist:         Final correspondence distance (metres).
        n_scales:          Number of distance steps between coarse and fine.
        max_iter:          GICP iterations per scale.
        translation_only:  If True, strip GICP rotation and keep only the
                           translation. Use for same-CRS data (e.g. UTM) where
                           rotation should be near-identity. A spurious GICP
                           rotation, when multiplied by large UTM coordinates,
                           causes km-scale errors in the recovered transform.

    Returns:
        Refined 4×4 transform (float64) in the original coordinate frame.
    """
    import open3d as o3d

    # Shift to local frame: subtract src centroid from both clouds.
    # This removes large absolute coordinates (e.g. UTM 705K / 5659K) that
    # can cause numerical drift in GICP's covariance and KDTree computations.
    c = src.xyz.mean(axis=0).astype(np.float64)

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector((src.xyz.astype(np.float64) - c))
    if src.normals is not None:
        src_pcd.normals = o3d.utility.Vector3dVector(src.normals.astype(np.float64))

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector((tgt.xyz.astype(np.float64) - c))
    if tgt.normals is not None:
        tgt_pcd.normals = o3d.utility.Vector3dVector(tgt.normals.astype(np.float64))

    # Adjust T_init to the local frame:
    #   T_local @ (x - c) = T_orig @ x - c
    #   T_local_t = T_orig_R @ c + T_orig_t - c
    R_init = T_init[:3, :3]
    t_init = T_init[:3, 3]
    T_local = np.eye(4)
    T_local[:3, :3] = R_init
    T_local[:3, 3] = R_init @ c + t_init - c

    distances = np.geomspace(coarse_dist, fine_dist, n_scales)
    T = T_local.copy()

    for dist in distances:
        result = o3d.pipelines.registration.registration_generalized_icp(
            src_pcd,
            tgt_pcd,
            max_correspondence_distance=dist,
            init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )
        T = np.array(result.transformation, dtype=np.float64)
        log.debug(
            "multiscale_gicp dist=%.2f  fitness=%.4f  rmse=%.4f  t=[%.2f, %.2f, %.2f]",
            dist, result.fitness, result.inlier_rmse, T[0, 3], T[1, 3], T[2, 3],
        )

    # Recover transform in original frame.
    # For same-CRS data (translation_only=True), strip any rotation GICP found
    # and keep only the translation.  A spurious rotation from GICP, when
    # applied to large UTM coordinates (e.g. 5.6 M), creates km-scale errors
    # in the recovered translation.  For same-CRS localization, rotation ≈ I.
    if translation_only:
        T_out = np.eye(4)
        T_out[:3, 3] = T[:3, 3]   # t_local IS the position offset (R=I assumed)
    else:
        T_out = T.copy()
        T_out[:3, 3] = T[:3, 3] - T[:3, :3] @ c + c
    return T_out


# ---------------------------------------------------------------------------
# ICP refinement
# ---------------------------------------------------------------------------

def icp_refinement(
    src: PointCloud,
    tgt: PointCloud,
    T_init: np.ndarray,
    voxel_size: float,
    point_to_plane: bool = True,
) -> np.ndarray:
    """Refine a registration estimate using ICP.

    Uses point-to-plane if normals are available and point_to_plane=True,
    otherwise falls back to point-to-point.

    Returns refined 4×4 transform (float64).
    """
    import open3d as o3d

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src.xyz.astype(np.float64))

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt.xyz.astype(np.float64))

    if tgt.normals is not None:
        tgt_pcd.normals = o3d.utility.Vector3dVector(tgt.normals.astype(np.float64))

    dist_thr = voxel_size * 2.0   # 2× voxel gives ICP room to converge from RANSAC quality init

    use_p2plane = (
        point_to_plane and tgt.normals is not None and len(tgt.normals) == len(tgt.xyz)
    )

    if use_p2plane:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    result = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        dist_thr,
        T_init,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )

    return np.array(result.transformation, dtype=np.float64)


def gicp_refinement(
    src: PointCloud,
    tgt: PointCloud,
    T_init: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Refine registration using Generalized ICP (covariance-weighted).

    More robust to noise than point-to-plane ICP — recommended for stereo
    point clouds where depth uncertainty is high.

    Returns refined 4×4 transform (float64).
    """
    import open3d as o3d

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src.xyz.astype(np.float64))

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt.xyz.astype(np.float64))

    result = o3d.pipelines.registration.registration_generalized_icp(
        src_pcd,
        tgt_pcd,
        max_correspondence_distance=voxel_size * 1.0,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )

    return np.array(result.transformation, dtype=np.float64)


def color_icp_refinement(
    src: PointCloud,
    tgt: PointCloud,
    T_init: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Refine registration using Colored ICP (geometry + RGB jointly).

    Significantly more accurate than plain ICP on textured outdoor scenes.
    Falls back to GICP if either cloud lacks color.

    Returns refined 4×4 transform (float64).
    """
    import open3d as o3d

    if src.colors is None or tgt.colors is None:
        log.warning("ColorICP: one or both clouds lack colors — falling back to GICP.")
        return gicp_refinement(src, tgt, T_init, voxel_size)

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src.xyz.astype(np.float64))
    src_pcd.colors = o3d.utility.Vector3dVector(src.colors.astype(np.float64))
    if src.normals is not None:
        src_pcd.normals = o3d.utility.Vector3dVector(src.normals.astype(np.float64))

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt.xyz.astype(np.float64))
    tgt_pcd.colors = o3d.utility.Vector3dVector(tgt.colors.astype(np.float64))
    if tgt.normals is not None:
        tgt_pcd.normals = o3d.utility.Vector3dVector(tgt.normals.astype(np.float64))

    result = o3d.pipelines.registration.registration_colored_icp(
        src_pcd,
        tgt_pcd,
        voxel_size * 1.0,
        T_init,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )

    return np.array(result.transformation, dtype=np.float64)
