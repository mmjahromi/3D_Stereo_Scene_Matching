"""LocalizationResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LocalizationResult:
    """Output of the localization pipeline for one model."""

    T_est: np.ndarray           # 4×4 best estimated transform (local → global tile)
    east_m: float               # estimated East position in global frame
    north_m: float              # estimated North position in global frame
    sigma_east_m: float         # 1-sigma uncertainty in East (from RANSAC spread)
    sigma_north_m: float        # 1-sigma uncertainty in North
    match_score: float          # best inlier ratio across all RANSAC runs
    model_name: str = ""
    time_s: float = 0.0
    n_correspondences: int = 0
    tile_origin: Optional[np.ndarray] = None  # XY offset of matched tile in global frame
