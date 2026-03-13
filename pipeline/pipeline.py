"""Orchestrator: load → preprocess → extract/register → match → evaluate → report."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_AXIS_MAP = {"x": 0, "y": 1, "z": 2}


class Pipeline:
    """End-to-end registration benchmarking pipeline."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self._models: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full pipeline as specified in cfg."""
        mode = self.cfg["pipeline"].get("mode", "benchmark")
        model_names = self.cfg["pipeline"].get("models", ["fpfh"])

        # Load models
        for name in model_names:
            try:
                self._models[name] = self._build_model(name)
            except ImportError as e:
                log.warning(f"Skipping {name}: {e}")

        if not self._models:
            raise RuntimeError("No models could be loaded.")

        pairs = self._collect_pairs()
        if not pairs:
            raise RuntimeError("No pairs found in config.")

        output_dir = self.cfg["pipeline"].get("output_dir", "results/")
        os.makedirs(output_dir, exist_ok=True)

        from pipeline.utils.reporting import ResultsTable

        aggregate: Dict[str, List[dict]] = {n: [] for n in self._models}
        table = ResultsTable()

        for i, (src_path, tgt_path, T_gt) in enumerate(pairs):
            pair_label = f"Pair {i+1}/{len(pairs)}: {Path(src_path).name} → {Path(tgt_path).name}"
            log.info(pair_label)
            print(f"\n{'='*60}")
            print(pair_label)
            print("=" * 60)

            src_raw, tgt_raw = self._load_pair(src_path, tgt_path)
            pair_table = ResultsTable()

            for name, model in self._models.items():
                log.info(f"  Running {name} …")
                try:
                    row = self._run_model(name, model, src_raw, tgt_raw, T_gt)
                except Exception as e:
                    log.error(f"  {name} failed: {e}", exc_info=True)
                    row = dict(
                        model=name, rre=None, rte=None, success=None,
                        inlier_ratio=None, chamfer=None, time_s=0.0, num_corr=0,
                    )

                pair_table.add(
                    model_name=row["model"],
                    T_est=row.get("T_est", np.eye(4)),
                    time_s=row.get("time_s", 0.0),
                    num_correspondences=row.get("num_corr", 0),
                    rre=row.get("rre"),
                    rte=row.get("rte"),
                    success=row.get("success"),
                    inlier_ratio=row.get("inlier_ratio"),
                    chamfer_distance=row.get("chamfer"),
                )

                aggregate[name].append(row)

            pair_table.print_table()

            # Visualise if requested
            if self.cfg["pipeline"].get("visualize", False):
                self._visualize_all(src_raw, tgt_raw, aggregate, T_gt)

        # Save CSV
        csv_path = table.save_csv(output_dir)
        if len(pairs) > 1:
            log.info("Aggregate results:")
            table.print_benchmark_summary(aggregate)

        log.info(f"Results saved to {csv_path}")

    def run_pair(
        self,
        src_path: str,
        tgt_path: str,
        model_names: List[str],
        T_gt: Optional[np.ndarray] = None,
        visualize: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run a single source-target pair through specified models.

        Returns dict of {model_name: T_est (4×4)}.
        """
        for name in model_names:
            if name not in self._models:
                try:
                    self._models[name] = self._build_model(name)
                except ImportError as e:
                    log.warning(f"Skipping {name}: {e}")

        src_raw, tgt_raw = self._load_pair(src_path, tgt_path)

        results: Dict[str, np.ndarray] = {}
        table = __import__("pipeline.utils.reporting", fromlist=["ResultsTable"]).ResultsTable()

        for name in model_names:
            model = self._models.get(name)
            if model is None:
                continue
            try:
                row = self._run_model(name, model, src_raw, tgt_raw, T_gt)
                results[name] = row["T_est"]
                table.add(
                    model_name=row["model"],
                    T_est=row["T_est"],
                    time_s=row.get("time_s", 0.0),
                    num_correspondences=row.get("num_corr", 0),
                    rre=row.get("rre"),
                    rte=row.get("rte"),
                    success=row.get("success"),
                    inlier_ratio=row.get("inlier_ratio"),
                    chamfer_distance=row.get("chamfer"),
                )
            except Exception as e:
                log.error(f"{name} failed: {e}", exc_info=True)

        table.print_table()

        if visualize:
            for name, T_est in results.items():
                from pipeline.visualization.visualizer import visualize_registration
                visualize_registration(src_raw, tgt_raw, T_est, T_gt,
                                       window_title=name)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, name: str):
        """Instantiate a model from config."""
        mcfg = self.cfg.get("models", {}).get(name, {})

        if name == "fpfh":
            from pipeline.models.fpfh import FPFHModel
            return FPFHModel(
                voxel_size=mcfg.get("voxel_size", 0.05),
                feature_radius_factor=mcfg.get("feature_radius_factor", 5.0),
                feature_max_nn=mcfg.get("feature_max_nn", 100),
                normal_radius_factor=mcfg.get("normal_radius_factor", 2.0),
                normal_max_nn=mcfg.get("normal_max_nn", 30),
                feature_radius=mcfg.get("feature_radius"),
            )
        elif name == "fgr":
            from pipeline.models.fgr import FGRModel
            return FGRModel(
                voxel_size=mcfg.get("voxel_size", 0.05),
                feature_radius=mcfg.get("feature_radius"),
                feature_radius_factor=mcfg.get("feature_radius_factor", 5.0),
                feature_max_nn=mcfg.get("feature_max_nn", 100),
                icp_refinement=mcfg.get("icp_refinement", True),
            )
        elif name == "fcgf":
            from pipeline.models.fcgf import FCGFModel
            m = FCGFModel(
                voxel_size=mcfg.get("voxel_size", 0.025),
                feature_dim=mcfg.get("feature_dim", 32),
            )
            weights = mcfg.get("weights_path")
            if weights and Path(weights).exists():
                m.load_weights(weights)
            else:
                log.warning(f"FCGF weights not found at '{weights}'. Run --download-weights.")
            return m
        elif name == "geotransformer":
            from pipeline.models.geotransformer import GeoTransformerModel
            m = GeoTransformerModel(
                voxel_size=mcfg.get("voxel_size", 0.025),
                config_path=mcfg.get("config_path"),
            )
            weights = mcfg.get("weights_path")
            if weights and Path(weights).exists():
                m.load_weights(weights)
            else:
                log.warning(
                    f"GeoTransformer weights not found at '{weights}'. "
                    "Run --download-weights."
                )
            return m
        elif name == "predator":
            from pipeline.models.predator import PredatorModel
            m = PredatorModel(
                voxel_size=mcfg.get("voxel_size", 0.025),
                max_keypoints=mcfg.get("max_keypoints", 1000),
                overlap_threshold=mcfg.get("overlap_threshold", 0.5),
            )
            weights = mcfg.get("weights_path")
            if weights and Path(weights).exists():
                m.load_weights(weights)
            else:
                log.warning(
                    f"Predator weights not found at '{weights}'. Run --download-weights."
                )
            return m
        elif name == "shot":
            from pipeline.models.shot import SHOTModel
            return SHOTModel(
                voxel_size=mcfg.get("voxel_size", 0.5),
                shot_radius=mcfg.get("shot_radius"),
                shot_radius_factor=mcfg.get("shot_radius_factor", 5.0),
                max_keypoints=mcfg.get("max_keypoints", 4000),
            )
        elif name == "gicp":
            from pipeline.models.gicp import GICPModel
            return GICPModel(
                voxel_size=mcfg.get("voxel_size", 0.5),
                feature_radius=mcfg.get("feature_radius"),
                feature_radius_factor=mcfg.get("feature_radius_factor", 5.0),
                feature_max_nn=mcfg.get("feature_max_nn", 100),
            )
        elif name == "color_icp":
            from pipeline.models.color_icp import ColorICPModel
            return ColorICPModel(
                voxel_size=mcfg.get("voxel_size", 0.5),
                feature_radius=mcfg.get("feature_radius"),
                feature_radius_factor=mcfg.get("feature_radius_factor", 5.0),
                feature_max_nn=mcfg.get("feature_max_nn", 100),
            )
        else:
            raise ValueError(f"Unknown model '{name}'")

    def _load_pair(self, src_path: str, tgt_path: str):
        from pipeline.data.loader import load_point_cloud

        src = load_point_cloud(src_path)
        tgt = load_point_cloud(tgt_path)
        log.info(f"  Loaded src: {src}, tgt: {tgt}")
        return src, tgt

    def _clip_to_extent(
        self,
        pcd: "PointCloud",
        reference: "PointCloud",
        buffer_m: float = 50.0,
        label: str = "",
    ) -> "PointCloud":
        """Return the subset of *pcd* that falls within *reference*'s XYZ bbox + buffer.

        Works in any shared coordinate system (UTM, local frame, …).
        All three axes are clipped so that points far outside the reference
        footprint (e.g. distant hillsides) are excluded.
        """
        from pipeline.models.base import PointCloud as PC

        lo = reference.xyz.min(axis=0) - buffer_m
        hi = reference.xyz.max(axis=0) + buffer_m

        mask     = np.all((pcd.xyz >= lo) & (pcd.xyz <= hi), axis=1)
        n_before = len(pcd.xyz)
        n_after  = int(mask.sum())

        log.info(
            f"Spatial clip [{label}]: {n_before:,} → {n_after:,} pts "
            f"(buffer={buffer_m:.0f} m, kept {100*n_after/max(n_before,1):.1f}%)"
        )

        if n_after == 0:
            log.warning(
                f"Spatial clip [{label}] removed ALL points — clouds may not "
                "overlap. Returning unclipped cloud."
            )
            return pcd

        return PC(
            xyz=pcd.xyz[mask],
            normals=pcd.normals[mask] if pcd.normals is not None else None,
            colors=pcd.colors[mask]   if pcd.colors  is not None else None,
        )

    def _preprocess_for_model(self, pcd, model_name: str, model):
        """Voxel-downsample (and estimate normals for FPFH/FGR/GICP/ColorICP)."""
        from pipeline.data.preprocessor import preprocess
        from pipeline.models.fpfh import FPFHModel
        from pipeline.models.fgr import FGRModel
        from pipeline.models.gicp import GICPModel
        from pipeline.models.color_icp import ColorICPModel
        from pipeline.models.shot import SHOTModel

        estimate_normals = isinstance(model, (FPFHModel, FGRModel, GICPModel, ColorICPModel, SHOTModel))
        mcfg = self.cfg.get("models", {}).get(model_name, {})

        return preprocess(
            pcd,
            voxel_size=model.voxel_size,
            estimate_normals=estimate_normals,
            normal_radius_factor=mcfg.get("normal_radius_factor", 2.0),
            normal_max_nn=mcfg.get("normal_max_nn", 30),
            skip_downsample=mcfg.get("skip_downsample", False),
            normal_radius=mcfg.get("normal_radius"),
            remove_outliers=mcfg.get("remove_outliers", False),
            outlier_nb_neighbors=mcfg.get("outlier_nb_neighbors", 20),
            outlier_std_ratio=mcfg.get("outlier_std_ratio", 2.0),
        )

    def _run_model(
        self,
        name: str,
        model,
        src_raw,
        tgt_raw,
        T_gt: Optional[np.ndarray],
    ) -> dict:
        from pipeline.models.base import PairModel
        from pipeline.matching.matcher import (
            match_features,
            ransac_registration,
            fgr_registration,
            icp_refinement,
            gicp_refinement,
            color_icp_refinement,
        )

        eval_cfg = self.cfg.get("evaluation", {})
        ransac_cfg = self.cfg.get("ransac", {})
        match_cfg = self.cfg.get("matching", {})
        reg_cfg = self.cfg.get("registration", {})
        global_method = reg_cfg.get("global_method", "ransac")
        refinement_method = reg_cfg.get("refinement", "icp")

        src = self._preprocess_for_model(src_raw, name, model)
        tgt = self._preprocess_for_model(tgt_raw, name, model)

        t0 = time.perf_counter()

        src_idx = tgt_idx = None
        T_est = np.eye(4)

        if isinstance(model, PairModel):
            T_est = model.register_pair(src, tgt)
            num_corr = 0
        else:
            src_feat = model.extract_features(src)
            tgt_feat = model.extract_features(tgt)

            from pipeline.utils.feature_diagnostics import check_discriminability
            check_discriminability(src_feat, label=f"{name}/src")
            check_discriminability(tgt_feat, label=f"{name}/tgt")

            if global_method == "fgr":
                T_est = fgr_registration(
                    src_feat.keypoints, tgt_feat.keypoints,
                    src_feat.descriptors, tgt_feat.descriptors,
                    voxel_size=model.voxel_size,
                    distance_threshold_factor=ransac_cfg.get("distance_threshold_factor", 1.5),
                )
                num_corr = len(src_feat.keypoints)
            else:
                src_idx, tgt_idx = match_features(
                    src_feat, tgt_feat,
                    method=match_cfg.get("method", "mutual_nn"),
                    ratio_threshold=match_cfg.get("ratio_threshold", 0.80),
                    use_faiss=match_cfg.get("use_faiss", False),
                )
                num_corr = len(src_idx)
                T_est = ransac_registration(
                    src_feat.keypoints, tgt_feat.keypoints,
                    src_idx, tgt_idx,
                    voxel_size=model.voxel_size,
                    max_iterations=ransac_cfg.get("max_iterations", 1_000_000),
                    confidence=ransac_cfg.get("confidence", 0.999),
                    ransac_n=ransac_cfg.get("ransac_n", 3),
                    distance_threshold_factor=ransac_cfg.get("distance_threshold_factor", 1.5),
                )

        # Refinement
        if self.cfg["pipeline"].get("icp_refinement", True):
            if refinement_method == "gicp":
                T_est = gicp_refinement(src, tgt, T_est, voxel_size=model.voxel_size)
            elif refinement_method == "color_icp":
                T_est = color_icp_refinement(src, tgt, T_est, voxel_size=model.voxel_size)
            else:
                T_est = icp_refinement(src, tgt, T_est, voxel_size=model.voxel_size)

        # Z correction (for outdoor/urban scenes where facade correspondences
        # leave Z-translation undetermined)
        z_cfg = self.cfg.get("z_correction", {})
        if z_cfg.get("enabled", False):
            from pipeline.utils.z_correction import correct_z
            T_est = correct_z(
                src.xyz, tgt.xyz, T_est,
                method=z_cfg.get("method", "ground_percentile"),
                z_axis=z_cfg.get("z_axis", 2),
                percentile=z_cfg.get("percentile", 5.0),
                bin_size=z_cfg.get("bin_size", 0.1),
                search_range=z_cfg.get("search_range", 5.0),
            )

        elapsed = time.perf_counter() - t0

        # Evaluation
        rre = rte = success = ir = cd = None
        if T_gt is not None:
            from pipeline.evaluation.metrics import evaluate_pair

            er = evaluate_pair(
                T_est, T_gt,
                rre_threshold=eval_cfg.get("rre_threshold", 15.0),
                rte_threshold=eval_cfg.get("rte_threshold", 0.30),
                src_kps=None if isinstance(model, PairModel) else src_feat.keypoints,
                tgt_kps=None if isinstance(model, PairModel) else tgt_feat.keypoints,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                src_xyz=src.xyz,
                tgt_xyz=tgt.xyz,
                voxel_size=model.voxel_size,
                compute_chamfer_dist=eval_cfg.get("compute_chamfer", False),
            )
            rre, rte, success = er.rre, er.rte, er.success
            ir, cd = er.inlier_ratio, er.chamfer_distance

        return dict(
            model=name, T_est=T_est, rre=rre, rte=rte, success=success,
            inlier_ratio=ir, chamfer=cd, time_s=elapsed, num_corr=num_corr,
        )

    def _collect_pairs(self) -> List[Tuple[str, str, Optional[np.ndarray]]]:
        """Return list of (src_path, tgt_path, T_gt_or_None) tuples."""
        from pipeline.data.loader import load_transform

        data_cfg = self.cfg.get("data", {})
        mode = data_cfg.get("mode", "custom_pairs")

        pairs = []

        if mode == "custom_pairs":
            for entry in data_cfg.get("custom_pairs", {}).get("pairs", []):
                src = entry["src"]
                tgt = entry["tgt"]
                gt_raw = entry.get("gt_transform")
                if isinstance(gt_raw, str) and gt_raw:
                    T_gt = load_transform(gt_raw)
                elif isinstance(gt_raw, list):
                    T_gt = np.array(gt_raw, dtype=np.float64)
                else:
                    T_gt = None
                pairs.append((src, tgt, T_gt))

        elif mode == "3dmatch":
            pairs = self._collect_3dmatch_pairs(data_cfg)

        return pairs

    def _collect_3dmatch_pairs(self, data_cfg: dict):
        """Collect pairs from a 3DMatch benchmark split."""
        benchmark_path = data_cfg.get("benchmark_path", "data/3dmatch")
        log.warning(
            "3DMatch benchmark loading not yet implemented. "
            f"Expecting benchmark at '{benchmark_path}'."
        )
        return []

    # ------------------------------------------------------------------
    # Localization API
    # ------------------------------------------------------------------

    def localize(
        self,
        local_path: str,
        global_path: str,
        model_names: List[str],
        visualize: bool = False,
    ) -> Dict[str, Any]:
        """Find local scan inside a global map and report N/E position + uncertainty.

        Returns dict of {model_name: LocalizationResult}.
        """
        import os
        import time

        from pipeline.localization import (
            GlobalSearcher,
            LocalizationResult,
            estimate_ne_uncertainty,
        )
        from pipeline.matching.matcher import match_features, icp_refinement
        from pipeline.models.base import PairModel

        loc_cfg = self.cfg.get("localization", {})
        match_cfg = self.cfg.get("matching", {})
        ransac_cfg = self.cfg.get("ransac", {})

        east_axis = loc_cfg.get("east_axis", "x")
        north_axis = loc_cfg.get("north_axis", "y")
        n_ransac_runs = loc_cfg.get("n_ransac_runs", 10)

        searcher = GlobalSearcher(
            tile_size=loc_cfg.get("tile_size", 20.0),
            tile_overlap=loc_cfg.get("tile_overlap", 0.3),
            max_direct_points=loc_cfg.get("max_direct_points", 100_000),
            top_k_tiles=loc_cfg.get("top_k_tiles", 3),
            min_tile_points=loc_cfg.get("min_tile_points", 200),
        )

        # Ensure models are loaded
        for name in model_names:
            if name not in self._models:
                try:
                    self._models[name] = self._build_model(name)
                except ImportError as e:
                    log.warning(f"Skipping {name}: {e}")

        local_raw, global_raw = self._load_pair(local_path, global_path)

        # Mutual spatial clipping: both clouds are trimmed to their shared
        # overlap region (+ buffer), avoiding processing of areas that can
        # never contribute a valid correspondence.
        # Both clips use the ORIGINAL extents so neither clip affects the other.
        crop_cfg = loc_cfg.get("spatial_crop", {})
        if crop_cfg.get("enabled", True):
            buf = crop_cfg.get("buffer_m", 50.0)
            local_raw_orig  = local_raw
            global_raw_orig = global_raw
            global_raw = self._clip_to_extent(
                global_raw_orig, reference=local_raw_orig,  buffer_m=buf, label="global→local"
            )
            local_raw  = self._clip_to_extent(
                local_raw_orig,  reference=global_raw_orig, buffer_m=buf, label="local→global"
            )

        # Pre-compute global map XYZ bounds for candidate validation.
        # Used to reject transforms that place the local scan outside the map.
        _g_lo = global_raw.xyz.min(axis=0)
        _g_hi = global_raw.xyz.max(axis=0)

        min_fitness    = loc_cfg.get("min_fitness", 0.05)
        bounds_axes    = [_AXIS_MAP[east_axis], _AXIS_MAP[north_axis]]  # XY by default

        results: Dict[str, LocalizationResult] = {}

        for name in model_names:
            model = self._models.get(name)
            if model is None:
                continue

            t0 = time.perf_counter()
            log.info(f"Localizing with {name} …")

            local_pp = self._preprocess_for_model(local_raw, name, model)
            global_pp = self._preprocess_for_model(global_raw, name, model)

            candidates = searcher.find_candidates(local_pp, global_pp, model.voxel_size)

            best_result: Optional[LocalizationResult] = None

            for tile_pcd, tile_origin in candidates:
                try:
                    if isinstance(model, PairModel):
                        T_final = model.register_pair(local_pp, tile_pcd)
                        # Proxy uncertainty: voxel_size (no multi-run available)
                        sigma_e = model.voxel_size
                        sigma_n = model.voxel_size
                        # Compute ICP fitness as score (fraction of src points within
                        # voxel_size * 1.5 of a target point under T_final)
                        try:
                            import open3d as o3d
                            _src = o3d.geometry.PointCloud()
                            _src.points = o3d.utility.Vector3dVector(local_pp.xyz.astype(np.float64))
                            _tgt = o3d.geometry.PointCloud()
                            _tgt.points = o3d.utility.Vector3dVector(tile_pcd.xyz.astype(np.float64))
                            _ev = o3d.pipelines.registration.evaluate_registration(
                                _src, _tgt, model.voxel_size * 1.5, T_final
                            )
                            score = float(_ev.fitness)
                        except Exception:
                            score = 0.0
                        n_corr = 0
                    else:
                        _reg_cfg = self.cfg.get("registration", {})
                        _global_method = _reg_cfg.get("global_method", "ransac")

                        if _global_method == "direct_gicp":
                            # Both clouds already in same CRS (e.g. UTM).
                            # Skip feature matching — initialise from centroid
                            # alignment and run multi-scale GICP directly.
                            # translation_only=True: keeps only the translation,
                            # discarding any spurious GICP rotation (which, when
                            # applied to large UTM coords, causes km-scale errors).
                            from pipeline.matching.matcher import multiscale_gicp
                            # For same-CRS data (both in UTM) the clouds already
                            # largely overlap — identity is the correct init.
                            # Centroid-alignment init pulls GICP into the wrong
                            # local minimum when the true offset is small (10-15m)
                            # but centroid difference is large (80-100m).
                            T_init = np.eye(4)
                            _gicp_cfg = _reg_cfg.get("direct_gicp", {})
                            T_best = multiscale_gicp(
                                local_pp, tile_pcd, T_init,
                                coarse_dist=_gicp_cfg.get("coarse_dist", 5.0),
                                fine_dist=_gicp_cfg.get("fine_dist", model.voxel_size),
                                n_scales=_gicp_cfg.get("n_scales", 4),
                                max_iter=_gicp_cfg.get("max_iter", 300),
                                translation_only=True,
                            )
                            import open3d as o3d
                            _src_o3d = o3d.geometry.PointCloud()
                            _src_o3d.points = o3d.utility.Vector3dVector(local_pp.xyz.astype(np.float64))
                            _tgt_o3d = o3d.geometry.PointCloud()
                            _tgt_o3d.points = o3d.utility.Vector3dVector(tile_pcd.xyz.astype(np.float64))
                            _ev = o3d.pipelines.registration.evaluate_registration(
                                _src_o3d, _tgt_o3d, model.voxel_size * 1.5, T_best
                            )
                            score = float(_ev.fitness)
                            sigma_e = sigma_n = model.voxel_size
                            n_corr = 0
                            log.info(
                                "  direct_gicp: fitness=%.4f  T_t=[%.3f, %.3f, %.3f]",
                                score, T_best[0, 3], T_best[1, 3], T_best[2, 3],
                            )
                            # multiscale_gicp already is the refinement — skip the
                            # extra GICP pass below to avoid a second divergence.
                            T_final = T_best
                        elif _global_method == "fgr":
                            local_feat = model.extract_features(local_pp)
                            tile_feat = model.extract_features(tile_pcd)
                            from pipeline.utils.feature_diagnostics import check_discriminability
                            check_discriminability(local_feat, label=f"{name}/local")
                            check_discriminability(tile_feat,  label=f"{name}/tile")
                            from pipeline.matching.matcher import fgr_registration
                            T_best = fgr_registration(
                                local_feat.keypoints, tile_feat.keypoints,
                                local_feat.descriptors, tile_feat.descriptors,
                                voxel_size=model.voxel_size,
                                distance_threshold_factor=ransac_cfg.get("distance_threshold_factor", 1.5),
                            )
                            import open3d as o3d
                            _src_o3d = o3d.geometry.PointCloud()
                            _src_o3d.points = o3d.utility.Vector3dVector(local_pp.xyz.astype(np.float64))
                            _tgt_o3d = o3d.geometry.PointCloud()
                            _tgt_o3d.points = o3d.utility.Vector3dVector(tile_pcd.xyz.astype(np.float64))
                            _ev = o3d.pipelines.registration.evaluate_registration(
                                _src_o3d, _tgt_o3d, model.voxel_size * 1.5, T_best
                            )
                            score = float(_ev.fitness)
                            sigma_e = sigma_n = model.voxel_size
                            n_corr = len(local_feat.keypoints)
                            log.info(
                                "  FGR: local_kps=%d  tile_kps=%d  fitness=%.4f"
                                "  T_t=[%.3f, %.3f, %.3f]",
                                len(local_feat.keypoints), len(tile_feat.keypoints),
                                score, T_best[0, 3], T_best[1, 3], T_best[2, 3],
                            )
                        else:
                            local_feat = model.extract_features(local_pp)
                            tile_feat = model.extract_features(tile_pcd)
                            from pipeline.utils.feature_diagnostics import check_discriminability
                            check_discriminability(local_feat, label=f"{name}/local")
                            check_discriminability(tile_feat,  label=f"{name}/tile")
                            si, ti = match_features(
                                local_feat, tile_feat,
                                method=match_cfg.get("method", "mutual_nn"),
                                ratio_threshold=match_cfg.get("ratio_threshold", 0.80),
                                use_faiss=match_cfg.get("use_faiss", False),
                            )
                            n_corr = len(si)
                            log.info(
                                "  Tile origin=(%.1f, %.1f, %.1f)  "
                                "local_kps=%d  tile_kps=%d  correspondences=%d",
                                tile_origin[0], tile_origin[1], tile_origin[2],
                                len(local_feat.keypoints), len(tile_feat.keypoints),
                                n_corr,
                            )
                            T_best, sigma_e, sigma_n, score = estimate_ne_uncertainty(
                                local_feat, tile_feat, si, ti,
                                voxel_size=model.voxel_size,
                                n_runs=n_ransac_runs,
                                ransac_cfg=ransac_cfg,
                                east_axis=east_axis,
                                north_axis=north_axis,
                            )

                        if _global_method != "direct_gicp":
                            # direct_gicp already ran multiscale refinement above;
                            # T_final was set there. Only refine for fgr/ransac paths.
                            if self.cfg["pipeline"].get("icp_refinement", True):
                                _ref = self.cfg.get("registration", {}).get("refinement", "icp")
                                if _ref == "gicp":
                                    from pipeline.matching.matcher import gicp_refinement as _refine
                                elif _ref == "color_icp":
                                    from pipeline.matching.matcher import color_icp_refinement as _refine
                                else:
                                    from pipeline.matching.matcher import icp_refinement as _refine
                                T_final = _refine(local_pp, tile_pcd, T_best, model.voxel_size)
                            else:
                                T_final = T_best

                    # Z correction — applied to both PairModel and BaseModel results
                    z_cfg = self.cfg.get("z_correction", {})
                    if z_cfg.get("enabled", False):
                        from pipeline.utils.z_correction import correct_z
                        T_final = correct_z(
                            local_pp.xyz, tile_pcd.xyz, T_final,
                            method=z_cfg.get("method", "ground_percentile"),
                            z_axis=z_cfg.get("z_axis", 2),
                            percentile=z_cfg.get("percentile", 5.0),
                            bin_size=z_cfg.get("bin_size", 0.1),
                            search_range=z_cfg.get("search_range", 5.0),
                        )

                    # ── Validation 1: minimum registration fitness ────────────
                    if score < min_fitness:
                        log.warning(
                            "  Candidate rejected (low fitness): score=%.4f < min_fitness=%.4f",
                            score, min_fitness,
                        )
                        continue

                    # ── Validation 2: translated centroid within global map ──
                    centroid_local  = local_pp.xyz.mean(axis=0).astype(np.float64)
                    centroid_global = T_final[:3, :3] @ centroid_local + T_final[:3, 3]
                    inside = all(
                        _g_lo[ax] <= centroid_global[ax] <= _g_hi[ax]
                        for ax in bounds_axes
                    )
                    if not inside:
                        log.warning(
                            "  Candidate rejected (out of bounds): centroid after transform "
                            "[%.1f, %.1f, %.1f] is outside global map "
                            "([%.1f, %.1f] × [%.1f, %.1f])",
                            centroid_global[0], centroid_global[1], centroid_global[2],
                            _g_lo[bounds_axes[0]], _g_hi[bounds_axes[0]],
                            _g_lo[bounds_axes[1]], _g_hi[bounds_axes[1]],
                        )
                        continue

                    # Tile keeps ENU coordinates → T translation IS already ENU
                    east_m  = float(T_final[_AXIS_MAP[east_axis],  3])
                    north_m = float(T_final[_AXIS_MAP[north_axis], 3])

                    candidate_result = LocalizationResult(
                        T_est=T_final,
                        east_m=east_m,
                        north_m=north_m,
                        sigma_east_m=sigma_e,
                        sigma_north_m=sigma_n,
                        match_score=score,
                        model_name=name,
                        time_s=time.perf_counter() - t0,
                        n_correspondences=n_corr,
                        tile_origin=tile_origin.copy(),
                    )

                    if best_result is None or score > best_result.match_score:
                        best_result = candidate_result

                except Exception as e:
                    log.error(f"  Candidate tile failed for {name}: {e}", exc_info=True)

            if best_result is not None:
                best_result.time_s = time.perf_counter() - t0
                results[name] = best_result
            else:
                log.error(f"  All candidates failed for {name}.")

        self._print_localization_table(results)

        output_dir = self.cfg["pipeline"].get("output_dir", "results/")
        os.makedirs(output_dir, exist_ok=True)
        self._save_localization_csv(results, output_dir)

        if visualize:
            from pipeline.visualization.visualizer import (
                visualize_registration,
                save_registration_screenshot,
            )
            for name, r in results.items():
                screenshot_path = os.path.join(output_dir, f"localization_{name}.png")
                log.info(f"Saving screenshot → {screenshot_path}")
                save_registration_screenshot(local_raw, global_raw, r.T_est, screenshot_path)
                log.info("Opening interactive viewer (close window to continue) …")
                visualize_registration(
                    local_raw, global_raw, r.T_est,
                    window_title=f"Localization: {name}  "
                                 f"E={r.east_m:.2f}m  N={r.north_m:.2f}m  "
                                 f"score={r.match_score:.2f}",
                )

        return results

    def _print_localization_table(self, results: Dict[str, Any]) -> None:
        """Print a formatted localization results table."""
        try:
            from tabulate import tabulate
        except ImportError:
            log.warning("tabulate not installed — printing plain results.")
            for name, r in results.items():
                print(
                    f"{name}: East={r.east_m:.3f}m, North={r.north_m:.3f}m, "
                    f"σ_E=±{r.sigma_east_m:.3f}m, σ_N=±{r.sigma_north_m:.3f}m, "
                    f"score={r.match_score:.2f}, t={r.time_s:.2f}s"
                )
            return

        headers = ["Model", "East pos (m)", "North pos (m)", "σ East (m)", "σ North (m)", "Score", "Time (s)"]
        rows = []
        for name, r in results.items():
            rows.append([
                name,
                f"{r.east_m:>12.3f}",
                f"{r.north_m:>13.3f}",
                f"±{r.sigma_east_m:.3f}",
                f"±{r.sigma_north_m:.3f}",
                f"{r.match_score:.2f}",
                f"{r.time_s:.2f}",
            ])

        print("\nLocalization Results")
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    def _save_localization_csv(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save localization results to CSV."""
        import csv
        import os

        csv_path = os.path.join(output_dir, "localization_results.csv")
        fieldnames = [
            "model", "east_m", "north_m", "sigma_east_m", "sigma_north_m",
            "match_score", "time_s", "n_correspondences",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, r in results.items():
                writer.writerow({
                    "model": name,
                    "east_m": round(r.east_m, 6),
                    "north_m": round(r.north_m, 6),
                    "sigma_east_m": round(r.sigma_east_m, 6),
                    "sigma_north_m": round(r.sigma_north_m, 6),
                    "match_score": round(r.match_score, 4),
                    "time_s": round(r.time_s, 3),
                    "n_correspondences": r.n_correspondences,
                })
        log.info(f"Localization results saved to {csv_path}")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _visualize_all(self, src_raw, tgt_raw, aggregate, T_gt):
        from pipeline.visualization.visualizer import visualize_registration

        for name, rows in aggregate.items():
            if rows:
                T_est = rows[-1].get("T_est")
                if T_est is not None:
                    visualize_registration(
                        src_raw, tgt_raw, T_est, T_gt,
                        window_title=f"Registration: {name}",
                    )
