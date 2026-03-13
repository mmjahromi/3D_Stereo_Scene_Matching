"""Generalized ICP (GICP) registration using Open3D (CPU, no weights needed)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline.models.base import PairModel, PointCloud


class GICPModel(PairModel):
    """FPFH global alignment followed by Generalized ICP refinement.

    GICP models per-point covariance from local surface geometry, making it
    more robust to noise than standard point-to-plane ICP — beneficial for
    stereo-derived point clouds where depth noise is high.

    Pipeline:
        1. FPFH features → RANSAC global alignment
        2. GICP refinement with covariance-weighted error metric
    """

    def __init__(
        self,
        voxel_size: float = 0.5,
        feature_radius: Optional[float] = None,
        feature_radius_factor: float = 5.0,
        feature_max_nn: int = 100,
    ) -> None:
        self.voxel_size = voxel_size
        self.feature_radius = feature_radius
        self.feature_radius_factor = feature_radius_factor
        self.feature_max_nn = feature_max_nn

    def load_weights(self, weights_path: str) -> None:
        """GICP has no learnable weights — no-op."""

    def register_pair(self, src: PointCloud, tgt: PointCloud) -> np.ndarray:
        """Register src into tgt.

        Args:
            src: Source point cloud (must have normals pre-computed).
            tgt: Target point cloud (must have normals pre-computed).

        Returns:
            4×4 rigid transform (float64) mapping src → tgt.
        """
        import open3d as o3d

        src_o3d = _to_o3d(src)
        tgt_o3d = _to_o3d(tgt)

        radius = self.feature_radius or (self.voxel_size * self.feature_radius_factor)
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=self.feature_max_nn
        )

        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(src_o3d, search_param)
        tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(tgt_o3d, search_param)

        # FGR replaces RANSAC: Geman-McClure robust loss handles noisy stereo
        # correspondences without a hard inlier/outlier threshold.
        src_desc = np.asarray(src_fpfh.data).T  # (N, D)
        tgt_desc = np.asarray(tgt_fpfh.data).T  # (M, D)
        from pipeline.matching.matcher import fgr_registration
        T_init = fgr_registration(
            np.asarray(src_o3d.points),
            np.asarray(tgt_o3d.points),
            src_desc,
            tgt_desc,
            voxel_size=self.voxel_size,
        )

        from pipeline.matching.matcher import gicp_refinement
        return gicp_refinement(src, tgt, T_init, self.voxel_size)


def _to_o3d(pcd: PointCloud):
    import open3d as o3d

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.xyz.astype(np.float64))
    if pcd.normals is not None:
        o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.normals.astype(np.float64))
    return o3d_pcd
