"""GeoTransformer pair-model wrapper.

Installation:
    pip install geotransformer
    # or clone https://github.com/qinzheng93/GeoTransformer
    #    cd GeoTransformer && pip install -e .

Weights:
    Download geotransformer-3dmatch.pth.tar from GitHub releases.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from pipeline.models.base import PairModel, PointCloud

log = logging.getLogger(__name__)


class GeoTransformerModel(PairModel):
    """GeoTransformer: Robust Point Cloud Registration with Geometric Transformers."""

    def __init__(
        self,
        voxel_size: float = 0.025,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.voxel_size = voxel_size
        self.weights_path = weights_path
        self.config_path = config_path
        self._model = None
        self._cfg = None

    def load_weights(self, weights_path: str) -> None:
        self.weights_path = weights_path
        self._load()

    def _load(self) -> None:
        try:
            from geotransformer.utils.open3d import (  # noqa: F401
                make_open3d_point_cloud,
            )
        except ImportError as e:
            raise ImportError(
                "GeoTransformer package not found. "
                "Install from https://github.com/qinzheng93/GeoTransformer"
            ) from e

        import torch
        from geotransformer.config import make_cfg
        from geotransformer.engine import SingleTester
        from geotransformer.modules.geotransformer import GeoTransformer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build config — allow override via config_path
        if self.config_path is not None:
            self._cfg = make_cfg(self.config_path)
        else:
            self._cfg = _default_3dmatch_config()

        self._model = GeoTransformer(self._cfg).to(self._device)

        checkpoint = torch.load(self.weights_path, map_location=self._device)
        state_dict = checkpoint.get("model", checkpoint)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        log.info(f"GeoTransformer weights loaded from {self.weights_path}")

    def register_pair(self, src: PointCloud, tgt: PointCloud) -> np.ndarray:
        """Run GeoTransformer on a source-target pair and return 4×4 transform."""
        if self._model is None:
            if self.weights_path is None:
                raise RuntimeError("Call load_weights() before register_pair().")
            self._load()

        import torch
        from geotransformer.utils.registration import (
            compute_registration_error,
            get_correspondences,
        )

        def _to_tensor(arr):
            return torch.from_numpy(arr.astype(np.float32)).to(self._device)

        with torch.no_grad():
            data = {
                "src_points": _to_tensor(src.xyz).unsqueeze(0),
                "tgt_points": _to_tensor(tgt.xyz).unsqueeze(0),
                "src_feats": torch.ones(1, len(src), 1, device=self._device),
                "tgt_feats": torch.ones(1, len(tgt), 1, device=self._device),
            }
            output = self._model(data)

        T_est = output["estimated_transform"].squeeze(0).cpu().numpy().astype(np.float64)
        return T_est


def _default_3dmatch_config():
    """Minimal stand-in config matching the 3DMatch checkpoint defaults."""
    from types import SimpleNamespace

    cfg = SimpleNamespace()
    cfg.backbone = SimpleNamespace(
        init_voxel_size=0.025,
        kernel_size=15,
        init_radius=0.0375,
        init_sigma=0.0125,
        group_norm=32,
        input_dim=1,
        init_dim=64,
        output_dim=256,
    )
    cfg.model = SimpleNamespace(
        ground_truth_matching_radius=0.05,
        num_points_in_patch=64,
        num_sinkhorn_iterations=100,
    )
    cfg.geotransformer = SimpleNamespace(
        input_dim=2048,
        hidden_dim=256,
        output_dim=256,
        num_heads=4,
        blocks=["self", "cross", "self", "cross", "self", "cross"],
        sigma_d=0.2,
        sigma_a=15,
        angle_k=3,
        reduction_a="max",
    )
    cfg.coarse_matching = SimpleNamespace(
        num_targets=128,
        overlap_threshold=0.1,
        num_correspondences=256,
        dual_normalization=True,
    )
    cfg.fine_matching = SimpleNamespace(
        acceptance_radius=0.1,
        mutual=True,
        confidence_threshold=0.05,
        use_dustbin=False,
        use_global_score=False,
        correspondences_from_superpatch=False,
        topk=1,
        estimated_overlap=0.2,
    )
    return cfg
