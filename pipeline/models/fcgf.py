"""FCGF (Fully Convolutional Geometric Features) wrapper.

Requires MinkowskiEngine:
    Build from source: https://github.com/NVIDIA/MinkowskiEngine
    Not installable via pip for PyTorch 2.x.

On macOS (no CUDA), FCGF runs on CPU — very slow.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from pipeline.models.base import BaseModel, FeatureOutput, PointCloud

log = logging.getLogger(__name__)


class FCGFModel(BaseModel):
    """FCGF feature extractor (MinkowskiEngine backbone)."""

    def __init__(
        self,
        voxel_size: float = 0.025,
        feature_dim: int = 32,
        weights_path: Optional[str] = None,
    ) -> None:
        self.voxel_size = voxel_size
        self.feature_dim = feature_dim
        self.weights_path = weights_path
        self._model = None
        self._device = None

    def load_weights(self, weights_path: str) -> None:
        """Load FCGF weights; lazily imports MinkowskiEngine and torch."""
        self.weights_path = weights_path
        self._load()

    def _load(self) -> None:
        try:
            import MinkowskiEngine as ME  # noqa: N813
        except ImportError as e:
            raise ImportError(
                "MinkowskiEngine is not installed. "
                "Build from source: https://github.com/NVIDIA/MinkowskiEngine"
            ) from e

        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device.type == "cpu":
            log.warning("FCGF: CUDA not available — running on CPU (very slow).")

        from pipeline.models._fcgf_arch import ResUNetBN2C

        model = ResUNetBN2C(
            in_channels=1,
            out_channels=self.feature_dim,
            normalize_feature=True,
            conv1_kernel_size=7,
            D=3,
        ).to(self._device)

        checkpoint = torch.load(self.weights_path, map_location=self._device)
        # Checkpoints may be stored under 'state_dict' key
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model
        log.info(f"FCGF weights loaded from {self.weights_path} on {self._device}")

    def extract_features(self, pcd: PointCloud) -> FeatureOutput:
        if self._model is None:
            if self.weights_path is None:
                raise RuntimeError("Call load_weights() before extract_features().")
            self._load()

        import MinkowskiEngine as ME  # noqa: N813
        import torch

        xyz = pcd.xyz.astype(np.float32)

        # Quantise coordinates to voxel grid and deduplicate
        coords = np.floor(xyz / self.voxel_size).astype(np.int32)
        coords, unique_idx = np.unique(coords, axis=0, return_index=True)
        xyz_unique = xyz[unique_idx]

        # Batch index = 0 (single scene)
        coords_with_batch = np.hstack(
            [np.zeros((len(coords), 1), dtype=np.int32), coords]
        )
        coords_t = torch.IntTensor(coords_with_batch)
        feats_t = torch.ones((len(coords), 1), dtype=torch.float32)

        sinput = ME.SparseTensor(
            features=feats_t,
            coordinates=coords_t,
            device=self._device,
        )

        with torch.no_grad():
            soutput = self._model(sinput)

        descriptors = soutput.F.cpu().numpy().astype(np.float32)

        return FeatureOutput(keypoints=xyz_unique, descriptors=descriptors)
