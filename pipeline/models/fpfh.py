"""FPFH feature extractor using Open3D (CPU, no GPU needed)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline.models.base import BaseModel, FeatureOutput, PointCloud


class FPFHModel(BaseModel):
    """Fast Point Feature Histograms via Open3D.

    No pretrained weights required. All parameters come from config.
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        feature_radius_factor: float = 5.0,
        feature_max_nn: int = 100,
        normal_radius_factor: float = 2.0,
        normal_max_nn: int = 30,
        feature_radius: Optional[float] = None,
    ) -> None:
        self.voxel_size = voxel_size
        self.feature_radius_factor = feature_radius_factor
        self.feature_max_nn = feature_max_nn
        self.normal_radius_factor = normal_radius_factor
        self.normal_max_nn = normal_max_nn
        self.feature_radius = feature_radius

    def load_weights(self, weights_path: str) -> None:
        """FPFH has no learnable weights — this is a no-op."""

    def extract_features(self, pcd: PointCloud) -> FeatureOutput:
        """Compute FPFH descriptors for every point in pcd.

        The input cloud should already be voxel-downsampled and have normals.
        """
        import open3d as o3d

        if pcd.normals is None:
            raise ValueError(
                "FPFH requires normals. Run preprocessor with estimate_normals=True."
            )

        o3d_pcd = _to_o3d(pcd)

        feature_radius = self.feature_radius or (self.voxel_size * self.feature_radius_factor)
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            o3d_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=feature_radius, max_nn=self.feature_max_nn
            ),
        )

        # fpfh.data is (33, N) — transpose to (N, 33)
        descriptors = np.asarray(fpfh.data, dtype=np.float32).T
        keypoints = pcd.xyz.copy()

        # L2-normalise per descriptor
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        descriptors = descriptors / norms

        return FeatureOutput(keypoints=keypoints, descriptors=descriptors)


def _to_o3d(pcd: PointCloud):
    import open3d as o3d

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.xyz.astype(np.float64))
    if pcd.normals is not None:
        o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.normals.astype(np.float64))
    return o3d_pcd
