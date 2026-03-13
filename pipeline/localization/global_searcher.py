"""Tile a large global map and pre-screen candidates for localization."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from pipeline.models.base import PointCloud

log = logging.getLogger(__name__)


class GlobalSearcher:
    """Tile a global point cloud and find candidate tiles matching a local scan.

    Args:
        tile_size:         Side length of each tile in metres.
        tile_overlap:      Fractional overlap between adjacent tiles (0–1).
        max_direct_points: If global cloud has ≤ this many points, skip tiling.
        top_k_tiles:       Return at most this many candidate tiles.
        min_tile_points:   Tiles with fewer points are discarded.
    """

    def __init__(
        self,
        tile_size: float = 20.0,
        tile_overlap: float = 0.3,
        max_direct_points: int = 100_000,
        top_k_tiles: int = 3,
        min_tile_points: int = 200,
    ) -> None:
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.max_direct_points = max_direct_points
        self.top_k_tiles = top_k_tiles
        self.min_tile_points = min_tile_points

    def find_candidates(
        self,
        local_pcd: PointCloud,
        global_pcd: PointCloud,
        voxel_size: float,
    ) -> List[Tuple[PointCloud, np.ndarray]]:
        """Return top-K (tile_pcd, tile_origin_xyz) pairs sorted by similarity score.

        If the global cloud is small enough, returns [(global_pcd, [0,0,0])].
        """
        if len(global_pcd.xyz) <= self.max_direct_points:
            log.info(
                f"Global map has {len(global_pcd.xyz)} points (≤ {self.max_direct_points}); "
                "skipping tiling — using full map directly."
            )
            return [(global_pcd, np.zeros(3))]

        tiles = self._create_tiles(global_pcd)
        if not tiles:
            log.warning("No valid tiles found; falling back to full global map.")
            return [(global_pcd, np.zeros(3))]

        log.info(f"Screening {len(tiles)} tiles for local scan match …")
        scored: List[Tuple[float, PointCloud, np.ndarray]] = []
        for tile_pcd, origin in tiles:
            score = self._screen_tile(local_pcd, tile_pcd)
            scored.append((score, tile_pcd, origin))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.top_k_tiles]

        log.info(
            "Top-%d tile scores: %s",
            len(top),
            [f"{s:.3f}" for s, _, _ in top],
        )
        return [(pcd, origin) for _, pcd, origin in top]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_tiles(
        self, global_pcd: PointCloud
    ) -> List[Tuple[PointCloud, np.ndarray]]:
        """Slide a tile window over the XY bounding box of the global cloud."""
        xyz = global_pcd.xyz
        x_min, y_min = float(xyz[:, 0].min()), float(xyz[:, 1].min())
        x_max, y_max = float(xyz[:, 0].max()), float(xyz[:, 1].max())

        stride = self.tile_size * (1.0 - self.tile_overlap)
        half = self.tile_size / 2.0

        tiles: List[Tuple[PointCloud, np.ndarray]] = []

        x = x_min + half
        while x < x_max + half:
            y = y_min + half
            while y < y_max + half:
                mask = (
                    (xyz[:, 0] >= x - half)
                    & (xyz[:, 0] < x + half)
                    & (xyz[:, 1] >= y - half)
                    & (xyz[:, 1] < y + half)
                )
                pts = xyz[mask]
                if len(pts) >= self.min_tile_points:
                    normals = global_pcd.normals[mask] if global_pcd.normals is not None else None
                    colors = global_pcd.colors[mask] if global_pcd.colors is not None else None
                    tile = PointCloud(xyz=pts, normals=normals, colors=colors)
                    origin = np.array([x - half, y - half, 0.0])
                    tiles.append((tile, origin))
                y += stride
            x += stride

        log.debug("Created %d tiles from global map.", len(tiles))
        return tiles

    def _screen_tile(self, local_pcd: PointCloud, tile_pcd: PointCloud) -> float:
        """Fast pre-ranking: cosine similarity of Z-height histograms.

        Z-histograms are rotation-invariant (independent of horizontal heading),
        so they work even when the local scan is in sensor frame and the tile is
        in the ENU world frame.  The histograms are computed over a common Z-range
        after mean-Z normalisation to cancel out the sensor height offset.
        """
        local_z = local_pcd.xyz[:, 2].astype(np.float32)
        tile_z  = tile_pcd.xyz[:,  2].astype(np.float32)

        # Normalise each cloud's Z to zero-mean so sensor height offset cancels
        local_z = local_z - local_z.mean()
        tile_z  = tile_z  - tile_z.mean()

        # Common range covering both distributions
        z_min = float(min(local_z.min(), tile_z.min()))
        z_max = float(max(local_z.max(), tile_z.max()))
        if z_max - z_min < 1e-6:
            return 0.0

        bins = 30
        local_hist, _ = np.histogram(local_z, bins=bins, range=(z_min, z_max))
        tile_hist,  _ = np.histogram(tile_z,  bins=bins, range=(z_min, z_max))

        local_hist = local_hist.astype(np.float32)
        tile_hist  = tile_hist.astype(np.float32)

        norm_l = np.linalg.norm(local_hist)
        norm_t = np.linalg.norm(tile_hist)
        if norm_l < 1e-12 or norm_t < 1e-12:
            return 0.0

        return float(np.dot(local_hist, tile_hist) / (norm_l * norm_t))
