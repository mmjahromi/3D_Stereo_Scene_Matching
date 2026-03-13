"""Predator (OverlapPredator) pair-model wrapper.

Installation:
    git clone https://github.com/prs-eth/OverlapPredator
    cd OverlapPredator && pip install -e .

Weights:
    Download predator_3dmatch.pth from the repo releases.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from pipeline.models.base import PairModel, PointCloud

log = logging.getLogger(__name__)


class PredatorModel(PairModel):
    """Overlap-Predator: Registration of 3D Point Clouds with Low Overlap."""

    def __init__(
        self,
        voxel_size: float = 0.025,
        weights_path: Optional[str] = None,
        max_keypoints: int = 1000,
        overlap_threshold: float = 0.5,
    ) -> None:
        self.voxel_size = voxel_size
        self.weights_path = weights_path
        self.max_keypoints = max_keypoints
        self.overlap_threshold = overlap_threshold
        self._model = None
        self._device = None

    def load_weights(self, weights_path: str) -> None:
        self.weights_path = weights_path
        self._load()

    def _load(self) -> None:
        try:
            from models.architectures import KPFCNN  # noqa: F401 — OverlapPredator package
        except ImportError as e:
            raise ImportError(
                "OverlapPredator is not installed. "
                "Clone https://github.com/prs-eth/OverlapPredator and pip install -e ."
            ) from e

        import torch
        from models.architectures import KPFCNN

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = _default_predator_config()
        self._model = KPFCNN(config, [32, 64, 128, 256], [0.2, 0.4, 0.8]).to(
            self._device
        )

        checkpoint = torch.load(self.weights_path, map_location=self._device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        log.info(f"Predator weights loaded from {self.weights_path}")

    def register_pair(self, src: PointCloud, tgt: PointCloud) -> np.ndarray:
        """Run Predator on a source-target pair and return 4×4 transform."""
        if self._model is None:
            if self.weights_path is None:
                raise RuntimeError("Call load_weights() before register_pair().")
            self._load()

        import torch
        from datasets.dataloader import collate_fn_descriptor
        from lib.benchmark_utils import ransac_pose_estimation, to_o3d_pcd

        def _torch(arr):
            return torch.from_numpy(arr.astype(np.float32)).to(self._device)

        src_pcd_o3d = to_o3d_pcd(src.xyz)
        tgt_pcd_o3d = to_o3d_pcd(tgt.xyz)

        with torch.no_grad():
            src_feats, tgt_feats, src_overlap, tgt_overlap, _ = self._model(
                [src_pcd_o3d, tgt_pcd_o3d],
                [_torch(src.xyz), _torch(tgt.xyz)],
            )

        # Select keypoints by overlap scores
        src_mask = src_overlap.squeeze() > self.overlap_threshold
        tgt_mask = tgt_overlap.squeeze() > self.overlap_threshold

        src_kps = src.xyz[src_mask.cpu().numpy().astype(bool)]
        tgt_kps = tgt.xyz[tgt_mask.cpu().numpy().astype(bool)]
        src_desc = src_feats[src_mask].cpu().numpy()
        tgt_desc = tgt_feats[tgt_mask].cpu().numpy()

        # Limit number of keypoints
        if len(src_kps) > self.max_keypoints:
            idx = np.random.choice(len(src_kps), self.max_keypoints, replace=False)
            src_kps = src_kps[idx]
            src_desc = src_desc[idx]
        if len(tgt_kps) > self.max_keypoints:
            idx = np.random.choice(len(tgt_kps), self.max_keypoints, replace=False)
            tgt_kps = tgt_kps[idx]
            tgt_desc = tgt_desc[idx]

        # RANSAC registration
        T_est = ransac_pose_estimation(
            src_kps, tgt_kps, src_desc, tgt_desc,
            mutual=True, distance_threshold=self.voxel_size * 1.5,
        )
        return T_est.astype(np.float64)


def _default_predator_config():
    """Minimal config matching the 3DMatch Predator checkpoint."""
    from types import SimpleNamespace

    cfg = SimpleNamespace()
    cfg.architecture = [
        "simple", "resnetb", "resnetb_strided",
        "resnetb", "resnetb", "resnetb_strided",
        "resnetb", "resnetb", "resnetb_strided",
        "resnetb", "resnetb", "nearest_upsample",
        "unary", "nearest_upsample", "unary",
        "nearest_upsample", "last_unary",
    ]
    cfg.in_feats_dim = 1
    cfg.first_feats_dim = 128
    cfg.gnn_feats_dim = 256
    cfg.num_layers = 4
    cfg.final_feats_dim = 32
    cfg.first_subsampling_dl = 0.025
    cfg.in_points_dim = 3
    cfg.conv_radius = 2.0
    cfg.deform_radius = 5.0
    cfg.KP_extent = 2.0
    cfg.KP_influence = "linear"
    cfg.aggregation_mode = "sum"
    cfg.batch_norm_momentum = 0.02
    cfg.use_batch_norm = True
    cfg.modulated = False
    cfg.num_kernel_points = 15
    cfg.overlap_radius = 0.04
    cfg.add_cross_overlap = True
    cfg.add_dustbin = False
    cfg.num_heads = 4
    return cfg
