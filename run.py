#!/usr/bin/env python3
"""CLI entry point for the 3D Scene Matching Pipeline.

Usage examples:

  # Single pair, inference only
  python run.py --src data/a.ply --tgt data/b.ply --models fpfh,fcgf

  # With GT evaluation
  python run.py --src data/a.ply --tgt data/b.ply --gt data/T_gt.npy \\
                --models fpfh,fcgf,geotransformer

  # Full benchmark driven by config.yaml
  python run.py

  # Download pretrained weights
  python run.py --download-weights [--models fcgf,geotransformer]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="3D Scene Matching Benchmarking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    p.add_argument(
        "--src", default=None,
        help="Source point cloud (PLY / PCD / NPZ). Overrides config.",
    )
    p.add_argument(
        "--tgt", default=None,
        help="Target point cloud (PLY / PCD / NPZ). Overrides config.",
    )
    p.add_argument(
        "--gt", default=None,
        help="Ground-truth transform (.npy or .txt 4×4 matrix). Optional.",
    )
    p.add_argument(
        "--models", default=None,
        help="Comma-separated list of models to run, e.g. 'fpfh,fcgf'. "
             "Overrides config.",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Directory for output CSV / screenshots. Overrides config.",
    )
    p.add_argument(
        "--visualize", action="store_true",
        help="Open Open3D visualiser after registration.",
    )
    p.add_argument(
        "--no-icp", action="store_true",
        help="Disable ICP refinement.",
    )
    p.add_argument(
        "--local", default=None,
        help="Local query scan (PLY / PCD / NPZ). Enables localization mode when "
             "combined with --global.",
    )
    p.add_argument(
        "--global", dest="global_map", default=None,
        help="Global reference map (PLY / PCD / NPZ). Enables localization mode when "
             "combined with --local.",
    )
    p.add_argument(
        "--download-weights", action="store_true",
        help="Download pretrained weights and exit.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        # Return a minimal default config
        return {
            "pipeline": {"mode": "benchmark", "models": ["fpfh"], "icp_refinement": True,
                         "output_dir": "results/", "visualize": False},
            "data": {"mode": "custom_pairs", "custom_pairs": {"pairs": []}},
            "models": {
                "fpfh": {"voxel_size": 0.05},
                "fcgf": {"voxel_size": 0.025, "weights_path": "weights/fcgf_3dmatch.pth"},
                "geotransformer": {"voxel_size": 0.025,
                                   "weights_path": "weights/geotransformer-3dmatch.pth.tar"},
                "predator": {"voxel_size": 0.025, "weights_path": "weights/predator_3dmatch.pth"},
            },
            "matching": {"method": "mutual_nn", "ratio_threshold": 0.80},
            "evaluation": {"rre_threshold": 15.0, "rte_threshold": 0.30,
                           "compute_chamfer": False},
            "ransac": {"max_iterations": 1_000_000, "confidence": 0.999,
                       "ransac_n": 3, "distance_threshold_factor": 1.5},
        }
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge CLI arguments into the config dict (CLI takes priority)."""
    if args.models:
        cfg["pipeline"]["models"] = [m.strip() for m in args.models.split(",")]

    if args.output_dir:
        cfg["pipeline"]["output_dir"] = args.output_dir

    if args.no_icp:
        cfg["pipeline"]["icp_refinement"] = False

    if args.visualize:
        cfg["pipeline"]["visualize"] = True

    if args.src and args.tgt:
        cfg["pipeline"]["mode"] = "inference" if args.gt is None else "benchmark"
        cfg["data"]["mode"] = "custom_pairs"

        pair = {"src": args.src, "tgt": args.tgt, "gt_transform": args.gt}
        cfg["data"]["custom_pairs"] = {"pairs": [pair]}

    return cfg


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # --- Download weights mode -------------------------------------------
    if args.download_weights:
        from pipeline.utils.downloader import download_weights

        model_list = None
        if args.models:
            model_list = [m.strip() for m in args.models.split(",")]

        cfg = _load_config(args.config)
        weights_dir = cfg.get("pipeline", {}).get("weights_dir", "weights")
        download_weights(models=model_list, weights_dir=weights_dir)
        return 0

    # --- Normal run ---------------------------------------------------------
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    from pipeline.pipeline import Pipeline

    pipeline = Pipeline(cfg)

    # --- Localization mode -----------------------------------------------
    if args.local and args.global_map:
        model_names = cfg["pipeline"]["models"]
        pipeline.localize(
            local_path=args.local,
            global_path=args.global_map,
            model_names=model_names,
            visualize=cfg["pipeline"].get("visualize", False),
        )
        return 0

    # --- Normal run ---------------------------------------------------------
    pairs = cfg.get("data", {}).get("custom_pairs", {}).get("pairs", [])
    if not pairs and not (args.src and args.tgt):
        log.error(
            "No pairs specified. Either use --src/--tgt flags, "
            "--local/--global for localization, or configure "
            "data.custom_pairs.pairs in config.yaml."
        )
        return 1

    if args.src and args.tgt:
        # Direct CLI pair invocation
        model_names = cfg["pipeline"]["models"]
        T_gt = None
        if args.gt:
            from pipeline.data.loader import load_transform
            T_gt = load_transform(args.gt)

        pipeline.run_pair(
            args.src, args.tgt,
            model_names=model_names,
            T_gt=T_gt,
            visualize=cfg["pipeline"].get("visualize", False),
        )
    else:
        pipeline.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
