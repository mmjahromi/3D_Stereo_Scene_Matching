"""Load PLY / PCD / NPZ files into PointCloud objects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np

from pipeline.models.base import PointCloud


def load_point_cloud(path: Union[str, Path]) -> PointCloud:
    """Load a point cloud file and return a PointCloud.

    Supported formats:
        .ply, .pcd  — via Open3D
        .npz        — expects arrays 'xyz'; optionally 'normals', 'colors'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    ext = path.suffix.lower()

    if ext in (".ply", ".pcd"):
        return _load_o3d(path)
    elif ext == ".npz":
        return _load_npz(path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported: .ply, .pcd, .npz"
        )


def _load_o3d(path: Path) -> PointCloud:
    import open3d as o3d  # lazy import — not needed for NPZ path

    pcd_o3d = o3d.io.read_point_cloud(str(path))
    if not pcd_o3d.has_points():
        raise ValueError(f"No points loaded from {path}")

    xyz = np.asarray(pcd_o3d.points, dtype=np.float64)

    normals = (
        np.asarray(pcd_o3d.normals, dtype=np.float32)
        if pcd_o3d.has_normals()
        else None
    )
    colors = (
        np.asarray(pcd_o3d.colors, dtype=np.float32)
        if pcd_o3d.has_colors()
        else None
    )
    return PointCloud(xyz=xyz, normals=normals, colors=colors)


def _load_npz(path: Path) -> PointCloud:
    data = np.load(path)

    if "xyz" not in data:
        raise KeyError(
            f"NPZ file {path} must contain an 'xyz' array. "
            f"Found keys: {list(data.keys())}"
        )

    xyz = data["xyz"].astype(np.float64)
    normals = data["normals"].astype(np.float32) if "normals" in data else None
    colors = data["colors"].astype(np.float32) if "colors" in data else None

    return PointCloud(xyz=xyz, normals=normals, colors=colors)


def load_transform(path: Union[str, Path]) -> np.ndarray:
    """Load a 4×4 ground-truth transform from a .npy or .txt file."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".npy":
        T = np.load(str(path))
    elif ext in (".txt", ".csv"):
        T = np.loadtxt(str(path))
    else:
        raise ValueError(f"Unsupported transform format '{ext}'. Supported: .npy, .txt")

    T = T.astype(np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected (4,4) transform, got shape {T.shape}")
    return T
