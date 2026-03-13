"""Colored ICP registration using Open3D (CPU, no weights needed)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline.models.base import PairModel, PointCloud


class ColorICPModel(PairModel):
    """FPFH global alignment followed by Colored ICP refinement.

    ColorICP jointly minimises geometric distance and photometric (RGB)
    error, making it significantly more accurate than plain ICP on textured
    outdoor scenes captured by stereo cameras.  Falls back to GICP if either
    cloud lacks color information.

    Pipeline:
        1. FPFH features → RANSAC global alignment
        2. Colored ICP refinement (geometry + RGB)
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
        """ColorICP has no learnable weights — no-op."""

    def register_pair(self, src: PointCloud, tgt: PointCloud) -> np.ndarray:
        """Register src into tgt using color-assisted ICP.

        Args:
            src: Source point cloud (normals + colors recommended).
            tgt: Target point cloud (normals + colors recommended).

        Returns:
            4×4 rigid transform (float64) mapping src → tgt.
        """
        import open3d as o3d

        src_o3d = _to_o3d_geom(src)
        tgt_o3d = _to_o3d_geom(tgt)

        radius = self.feature_radius or (self.voxel_size * self.feature_radius_factor)
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=self.feature_max_nn
        )

        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(src_o3d, search_param)
        tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(tgt_o3d, search_param)

        dist_thr_coarse = self.voxel_size * 4.0
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_o3d,
            tgt_o3d,
            src_fpfh,
            tgt_fpfh,
            mutual_filter=True,
            max_correspondence_distance=dist_thr_coarse,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thr_coarse),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1_000_000, 0.999),
        )
        T_init = np.asarray(result_ransac.transformation, dtype=np.float64)

        from pipeline.matching.matcher import color_icp_refinement
        return color_icp_refinement(src, tgt, T_init, self.voxel_size)


def _to_o3d_geom(pcd: PointCloud):
    """Convert PointCloud to Open3D with xyz + normals only (for FPFH)."""
    import open3d as o3d

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.xyz.astype(np.float64))
    if pcd.normals is not None:
        o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.normals.astype(np.float64))
    return o3d_pcd
