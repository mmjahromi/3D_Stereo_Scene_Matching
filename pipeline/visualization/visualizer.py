"""Open3D-based before/after registration visualizer."""

from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline.models.base import PointCloud


def visualize_registration(
    src: PointCloud,
    tgt: PointCloud,
    T_est: np.ndarray,
    T_gt: Optional[np.ndarray] = None,
    window_title: str = "Registration Result",
    point_size: float = 2.0,
) -> None:
    """Show source and target clouds before and after applying T_est.

    Colors:
        Source (original)   → red
        Source (transformed) → yellow
        Target              → blue
        Source GT-aligned   → green  (only if T_gt provided)

    Opens an Open3D visualisation window (blocking).
    """
    import open3d as o3d

    def _to_o3d(xyz: np.ndarray, color) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.paint_uniform_color(color)
        return pcd

    src_orig = _to_o3d(src.xyz, [1, 0, 0])          # red
    tgt_pcd = _to_o3d(tgt.xyz, [0, 0.651, 0.929])   # blue

    src_transformed = src_orig.transform(T_est)
    src_est = _to_o3d(np.asarray(src_transformed.points), [1, 0.706, 0])  # yellow

    geometries = [_to_o3d(src.xyz, [1, 0, 0]), tgt_pcd, src_est]

    if T_gt is not None:
        src_gt_xyz = (T_gt @ np.hstack([src.xyz, np.ones((len(src), 1))]).T).T[:, :3]
        geometries.append(_to_o3d(src_gt_xyz, [0, 1, 0]))  # green

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title)
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()


def save_registration_screenshot(
    src: PointCloud,
    tgt: PointCloud,
    T_est: np.ndarray,
    output_path: str,
) -> None:
    """Save an off-screen screenshot of the registration result."""
    import open3d as o3d

    def _to_o3d(xyz: np.ndarray, color) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.paint_uniform_color(color)
        return pcd

    tgt_pcd = _to_o3d(tgt.xyz, [0, 0.651, 0.929])
    src_xyz_t = (T_est @ np.hstack([src.xyz, np.ones((len(src), 1))]).T).T[:, :3]
    src_t_pcd = _to_o3d(src_xyz_t, [1, 0.706, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(tgt_pcd)
    vis.add_geometry(src_t_pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path)
    vis.destroy_window()
