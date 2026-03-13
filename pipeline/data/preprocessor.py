"""Voxel downsampling and normal estimation."""

from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline.models.base import PointCloud


def preprocess(
    pcd: PointCloud,
    voxel_size: float,
    estimate_normals: bool = True,
    normal_radius_factor: float = 2.0,
    normal_max_nn: int = 30,
    skip_downsample: bool = False,
    normal_radius: Optional[float] = None,
    remove_outliers: bool = False,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
) -> PointCloud:
    """Voxel-downsample and (optionally) estimate normals.

    Args:
        pcd:                  Input point cloud.
        voxel_size:           Voxel grid size in metres.
        estimate_normals:     Whether to compute/orient normals.
        normal_radius_factor: normal search radius = voxel_size * factor (used when
                              normal_radius is None).
        normal_max_nn:        Max neighbours for normal estimation.
        skip_downsample:      When True, skip voxel_down_sample() entirely (useful
                              when the cloud is already sparse).
        normal_radius:        Explicit normal search radius in metres.  If None, falls
                              back to voxel_size * normal_radius_factor.  If
                              skip_downsample is True and voxel_size <= 0, defaults
                              to 1.0 m.

    Returns:
        Preprocessed PointCloud.
    """
    import open3d as o3d

    pcd_o3d = _to_o3d(pcd)

    # Voxel downsample (optional)
    if skip_downsample:
        pcd_down = pcd_o3d
    else:
        pcd_down = pcd_o3d.voxel_down_sample(voxel_size)

    # Statistical outlier removal — removes isolated noise points.
    # Applied after downsampling so the neighbour statistics are computed
    # on the same density as the feature extraction step.
    if remove_outliers:
        pcd_down, _ = pcd_down.remove_statistical_outlier(
            nb_neighbors=outlier_nb_neighbors,
            std_ratio=outlier_std_ratio,
        )

    if estimate_normals:
        if normal_radius is not None:
            radius = normal_radius
        elif voxel_size > 0:
            radius = voxel_size * normal_radius_factor
        else:
            radius = 1.0  # safe fallback when voxel_size is not meaningful

        # Center before normal estimation to avoid qhull precision failures
        # with large absolute coordinates (e.g. UTM: X~700000, Y~5600000).
        pts = np.asarray(pcd_down.points)
        centroid = pts.mean(axis=0)
        pcd_down.points = o3d.utility.Vector3dVector(pts - centroid)

        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=normal_max_nn
            )
        )
        # Step 1: globally consistent orientation (arbitrary sign).
        pcd_down.orient_normals_consistent_tangent_plane(k=15)

        # Step 2: fix global sign ambiguity by orienting toward the centroid.
        # orient_normals_consistent_tangent_plane gives consistent RELATIVE
        # orientation but the overall sign (all-inward vs all-outward) is
        # arbitrary and differs between clouds — which flips FPFH features.
        # Orienting toward the centroid (camera at [0,0,0] after centering)
        # locks the global convention to "normals point inward" for both clouds.
        pcd_down.orient_normals_towards_camera_location(
            camera_location=np.array([0.0, 0.0, 0.0])
        )

        # Restore original coordinates
        pcd_down.points = o3d.utility.Vector3dVector(
            np.asarray(pcd_down.points) + centroid
        )

    return _from_o3d(pcd_down)


def _to_o3d(pcd: PointCloud):
    """Convert PointCloud to open3d.geometry.PointCloud."""
    import open3d as o3d

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.xyz.astype(np.float64))
    if pcd.normals is not None:
        o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.normals.astype(np.float64))
    if pcd.colors is not None:
        o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors.astype(np.float64))
    return o3d_pcd


def _from_o3d(o3d_pcd) -> PointCloud:
    """Convert open3d.geometry.PointCloud back to PointCloud."""
    xyz = np.asarray(o3d_pcd.points, dtype=np.float64)
    normals = (
        np.asarray(o3d_pcd.normals, dtype=np.float32)
        if o3d_pcd.has_normals()
        else None
    )
    colors = (
        np.asarray(o3d_pcd.colors, dtype=np.float32)
        if o3d_pcd.has_colors()
        else None
    )
    return PointCloud(xyz=xyz, normals=normals, colors=colors)
