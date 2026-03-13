"""Core data structures and abstract base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PointCloud:
    """Unified point cloud representation."""

    xyz: np.ndarray                        # (N, 3) float32
    normals: Optional[np.ndarray] = None   # (N, 3) float32
    colors: Optional[np.ndarray] = None    # (N, 3) float32 in [0, 1]

    def __len__(self) -> int:
        return len(self.xyz)

    def __repr__(self) -> str:
        has_n = self.normals is not None
        has_c = self.colors is not None
        return f"PointCloud(N={len(self)}, normals={has_n}, colors={has_c})"


@dataclass
class FeatureOutput:
    """Keypoints + descriptors extracted by a per-cloud model."""

    keypoints: np.ndarray               # (K, 3) float32
    descriptors: np.ndarray             # (K, D) float32, L2-normalised
    scores: Optional[np.ndarray] = None  # (K,) overlap/matchability (Predator only)

    def __len__(self) -> int:
        return len(self.keypoints)

    def __repr__(self) -> str:
        K, D = self.descriptors.shape
        return f"FeatureOutput(K={K}, D={D}, scores={self.scores is not None})"


@dataclass
class PairResult:
    """Full result for one source-target pair under one model."""

    model_name: str
    T_est: np.ndarray                          # (4, 4) estimated rigid transform
    T_gt: Optional[np.ndarray] = None          # (4, 4) ground-truth transform
    rre: Optional[float] = None                # degrees
    rte: Optional[float] = None                # metres
    inlier_ratio: Optional[float] = None
    chamfer_distance: Optional[float] = None
    num_correspondences: int = 0
    time_s: float = 0.0

    @property
    def success(self) -> Optional[bool]:
        """True if RRE < threshold AND RTE < threshold (set externally)."""
        return self._success

    @success.setter
    def success(self, value: bool) -> None:
        self._success = value

    def __post_init__(self) -> None:
        self._success: Optional[bool] = None


class BaseModel(ABC):
    """Abstract base for per-cloud feature extractors (FPFH, FCGF)."""

    voxel_size: float = 0.05

    @abstractmethod
    def load_weights(self, weights_path: str) -> None:
        """Load pretrained weights (lazy heavy imports go here)."""

    @abstractmethod
    def extract_features(self, pcd: PointCloud) -> FeatureOutput:
        """Extract keypoints + L2-normalised descriptors from a single cloud."""


class PairModel(ABC):
    """Abstract base for pair models that compute T directly (GeoTransformer, Predator)."""

    voxel_size: float = 0.025

    @abstractmethod
    def load_weights(self, weights_path: str) -> None:
        """Load pretrained weights."""

    @abstractmethod
    def register_pair(self, src: PointCloud, tgt: PointCloud) -> np.ndarray:
        """Return estimated 4×4 rigid transform T_est that maps src → tgt."""
