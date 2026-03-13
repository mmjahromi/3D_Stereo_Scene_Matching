"""SHOT (Signature of Histograms of OrienTations) descriptor.

Classical 352-D descriptor with a robust Local Reference Frame.  Better
noise tolerance than FPFH for stereo-derived outdoor point clouds.

Algorithm (Tombari et al., 2010):
    1. ISS keypoint detection (Open3D).
    2. For each keypoint: find neighbours within radius R.
    3. Compute weighted-PCA Local Reference Frame (LRF).
    4. Divide neighbourhood sphere into 32 spatial cells
       (8 azimuth × 2 elevation × 2 radial).
    5. Per cell: accumulate an 11-bin cosine histogram of
       neighbour-normal dot-products with the LRF z-axis.
    6. Concatenate → 352-D, L2-normalise.

No pretrained weights required.  Pure numpy + scipy + Open3D.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline.models.base import BaseModel, FeatureOutput, PointCloud

# SHOT spatial layout
_N_AZIMUTH   = 8
_N_ELEVATION = 2
_N_RADIAL    = 2
_N_CELLS     = _N_AZIMUTH * _N_ELEVATION * _N_RADIAL  # 32
_N_HIST_BINS = 11
_DESC_DIM    = _N_CELLS * _N_HIST_BINS                # 352


class SHOTModel(BaseModel):
    """SHOT feature extractor via ISS keypoints + numpy SHOT computation.

    Uses ISS to detect a sparse, repeatable set of keypoints, then computes
    the full 352-D SHOT descriptor on each.  This is significantly faster
    than computing SHOT on every voxel-downsampled point while retaining
    competitive matching performance.
    """

    def __init__(
        self,
        voxel_size: float = 0.5,
        shot_radius: Optional[float] = None,
        shot_radius_factor: float = 5.0,
        max_keypoints: int = 4000,
    ) -> None:
        self.voxel_size     = voxel_size
        self.shot_radius    = shot_radius          # explicit radius; overrides factor
        self.shot_radius_factor = shot_radius_factor
        self.max_keypoints  = max_keypoints

    def load_weights(self, weights_path: str) -> None:
        """SHOT has no learnable weights — no-op."""

    def extract_features(self, pcd: PointCloud) -> FeatureOutput:
        """Detect ISS keypoints and compute SHOT descriptors.

        Args:
            pcd: Preprocessed point cloud (must have normals).

        Returns:
            FeatureOutput with (K, 352) descriptors at ISS keypoint positions.
        """
        if pcd.normals is None:
            raise ValueError(
                "SHOT requires normals. Run preprocessor with estimate_normals=True."
            )

        radius = self.shot_radius or (self.voxel_size * self.shot_radius_factor)

        kp_xyz = self._detect_iss_keypoints(pcd, radius)

        if len(kp_xyz) == 0:
            # Fallback: subsample uniformly when ISS finds nothing
            idx = np.random.choice(len(pcd.xyz), min(self.max_keypoints, len(pcd.xyz)),
                                   replace=False)
            kp_xyz = pcd.xyz[idx]

        descriptors = _compute_shot(pcd.xyz, pcd.normals, kp_xyz, radius)

        # L2-normalise
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        descriptors = (descriptors / norms).astype(np.float32)

        return FeatureOutput(keypoints=kp_xyz.astype(np.float32),
                             descriptors=descriptors)

    # ------------------------------------------------------------------

    def _detect_iss_keypoints(self, pcd: PointCloud, radius: float) -> np.ndarray:
        """Return (K, 3) ISS keypoints, capped at max_keypoints."""
        import open3d as o3d

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd.xyz.astype(np.float64))

        kp = o3d.geometry.keypoint.compute_iss_keypoints(
            o3d_pcd,
            salient_radius=radius * 1.2,
            non_max_radius=radius * 0.8,
            gamma_21=0.975,
            gamma_32=0.975,
            min_neighbors=5,
        )
        kp_xyz = np.asarray(kp.points, dtype=np.float32)

        if len(kp_xyz) > self.max_keypoints:
            idx = np.random.choice(len(kp_xyz), self.max_keypoints, replace=False)
            kp_xyz = kp_xyz[idx]

        return kp_xyz


# ---------------------------------------------------------------------------
# Core SHOT computation (numpy + scipy)
# ---------------------------------------------------------------------------

def _compute_lrf(rel: np.ndarray, dists: np.ndarray, radius: float) -> np.ndarray:
    """Weighted-PCA Local Reference Frame.

    Args:
        rel:    (M, 3) neighbour positions relative to keypoint.
        dists:  (M,)  Euclidean distances.
        radius: search radius (used as weight denominator).

    Returns:
        (3, 3) orthonormal rotation matrix [x | y | z] where z is the
        least-dominant axis (surface normal direction).
    """
    weights = np.maximum(radius - dists, 0.0)
    w_sum = weights.sum()
    if w_sum < 1e-10:
        return np.eye(3, dtype=np.float64)

    # Weighted covariance
    w_rel = rel * weights[:, None]
    M = (rel.T @ w_rel) / w_sum  # (3, 3)

    eigenvalues, eigenvectors = np.linalg.eigh(M)  # ascending order
    # x = dominant, z = least dominant (surface normal proxy)
    x_axis = eigenvectors[:, 2].copy()
    z_axis = eigenvectors[:, 0].copy()

    # Disambiguate sign: majority of neighbours should project positively
    if np.sum(rel @ x_axis >= 0) < len(rel) / 2:
        x_axis = -x_axis
    if np.sum(rel @ z_axis >= 0) < len(rel) / 2:
        z_axis = -z_axis

    y_axis = np.cross(z_axis, x_axis)
    n = np.linalg.norm(y_axis)
    if n < 1e-10:
        return np.eye(3, dtype=np.float64)
    y_axis /= n
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    return np.column_stack([x_axis, y_axis, z_axis])  # (3, 3)


def _compute_shot(
    all_xyz: np.ndarray,
    all_normals: np.ndarray,
    kp_xyz: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Compute SHOT descriptors for every keypoint.

    Args:
        all_xyz:     (N, 3) full point positions (float32 or float64).
        all_normals: (N, 3) corresponding normals.
        kp_xyz:      (K, 3) keypoint positions.
        radius:      SHOT neighbourhood radius.

    Returns:
        (K, 352) float32 descriptors (not yet L2-normalised).
    """
    from scipy.spatial import cKDTree

    xyz64     = all_xyz.astype(np.float64)
    normals64 = all_normals.astype(np.float64)
    kp64      = kp_xyz.astype(np.float64)

    tree      = cKDTree(xyz64)
    nn_lists  = tree.query_ball_point(kp64, radius, workers=-1)

    descriptors = np.zeros((len(kp_xyz), _DESC_DIM), dtype=np.float32)

    for i, nn_idx in enumerate(nn_lists):
        nn_idx = np.asarray(nn_idx, dtype=np.int64)
        if len(nn_idx) < 5:
            continue

        center = kp64[i]
        rel    = xyz64[nn_idx] - center          # (M, 3)
        dists  = np.linalg.norm(rel, axis=1)    # (M,)

        valid = dists > 1e-6
        if valid.sum() < 5:
            continue

        rel       = rel[valid]
        dists     = dists[valid]
        nn_norms  = normals64[nn_idx[valid]]

        # Local Reference Frame
        R = _compute_lrf(rel, dists, radius)     # (3, 3)

        # Project into LRF
        local_pts  = rel      @ R                # (M, 3)
        local_nrms = nn_norms @ R                # (M, 3)

        # ---- spatial bins ------------------------------------------------
        # Elevation: 0 = upper (z >= 0), 1 = lower
        cos_elev  = local_pts[:, 2] / (dists + 1e-10)
        elev_bins = (cos_elev < 0).astype(np.int32)

        # Azimuth: [−π, π] → [0, 8)
        azimuth  = np.arctan2(local_pts[:, 1], local_pts[:, 0])
        az_bins  = np.floor((azimuth + np.pi) / (2.0 * np.pi) * _N_AZIMUTH
                            ).astype(np.int32) % _N_AZIMUTH

        # Radial: inner half=0, outer half=1
        rad_bins = (dists > radius * 0.5).astype(np.int32)

        cell_idx = (elev_bins * _N_AZIMUTH + az_bins) * _N_RADIAL + rad_bins  # (M,)

        # ---- histogram bins ----------------------------------------------
        cos_normal = np.clip(local_nrms[:, 2], -1.0, 1.0)
        hist_bins  = np.floor((cos_normal + 1.0) / 2.0 * _N_HIST_BINS
                              ).astype(np.int32)
        hist_bins  = np.clip(hist_bins, 0, _N_HIST_BINS - 1)

        flat_idx   = cell_idx * _N_HIST_BINS + hist_bins                      # (M,)
        desc       = np.bincount(flat_idx, minlength=_DESC_DIM).astype(np.float32)

        descriptors[i] = desc

    return descriptors
