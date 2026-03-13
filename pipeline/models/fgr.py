"""Fast Global Registration (FGR) using Open3D (CPU, no weights needed)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline.models.base import PairModel, PointCloud


class FGRModel(PairModel):
    """Fast Global Registration via Open3D's FGR algorithm.

    Deterministic (unlike RANSAC) and typically faster. No pretrained weights.
    Internally computes FPFH features then calls O3D's FGR solver.
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        feature_radius: Optional[float] = None,
        feature_radius_factor: float = 5.0,
        feature_max_nn: int = 100,
        icp_refinement: bool = True,
    ) -> None:
        self.voxel_size = voxel_size
        self.feature_radius = feature_radius
        self.feature_radius_factor = feature_radius_factor
        self.feature_max_nn = feature_max_nn
        self._icp_refinement = icp_refinement

    def load_weights(self, weights_path: str) -> None:
        """FGR has no learnable weights — this is a no-op."""

    def register_pair(self, src: PointCloud, tgt: PointCloud) -> np.ndarray:
        """Register src into tgt using FGR + optional ICP.

        Args:
            src: Source point cloud (must have normals for FPFH).
            tgt: Target point cloud (must have normals for FPFH).

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

        # maximum_correspondence_distance is O3D FGR's inlier/outlier cutoff in 3D space.
        # Recommended: voxel_size * 1.5 (NOT the feature descriptor radius).
        # Using radius * 0.5 would be ~1 m for outdoor scenes — far too large and causes divergence.
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_o3d,
            tgt_o3d,
            src_fpfh,
            tgt_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=self.voxel_size * 1.5,
            ),
        )

        T_est = np.asarray(result.transformation, dtype=np.float64)

        if self._icp_refinement:
            from pipeline.matching.matcher import gicp_refinement
            T_est = gicp_refinement(src, tgt, T_est, voxel_size=self.voxel_size)

        return T_est


def _to_o3d(pcd: PointCloud):
    import open3d as o3d

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.xyz.astype(np.float64))
    if pcd.normals is not None:
        o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.normals.astype(np.float64))
    return o3d_pcd
