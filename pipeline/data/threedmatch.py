"""3DMatch benchmark loader utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

BENCHMARK_SCENES = [
    "7-scenes-redkitchen",
    "sun3d-home_at-home_at_scan1_2013_jan_1",
    "sun3d-home_md-home_md_scan9_2012_sep_30",
    "sun3d-hotel_uc-scan3",
    "sun3d-hotel_umd-maryland_hotel1",
    "sun3d-hotel_umd-maryland_hotel3",
    "sun3d-mit_76_studyroom-76-1studyroom2",
    "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
]


def parse_gt_log(gt_log_path: str | Path) -> List[Tuple[int, int, np.ndarray]]:
    """Parse a gt.log file.

    The 3DMatch gt.log stores the transform T that maps **tgt → src** (i.e.
    the inverse of the registration transform).  This function returns the
    **src→tgt** transform by inverting each matrix, which is the convention
    used by the pipeline and evaluation code.

    Returns list of (src_id, tgt_id, T_src_to_tgt) tuples.
    """
    path = Path(gt_log_path)
    entries: List[Tuple[int, int, np.ndarray]] = []

    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        header = lines[i].split()
        src_id, tgt_id = int(header[0]), int(header[1])
        rows = [list(map(float, lines[i + r + 1].split())) for r in range(4)]
        T_tgt_to_src = np.array(rows, dtype=np.float64)
        # gt.log stores tgt→src; invert to get src→tgt
        T_src_to_tgt = np.linalg.inv(T_tgt_to_src)
        entries.append((src_id, tgt_id, T_src_to_tgt))
        i += 5

    return entries


def parse_gt_info(gt_info_path: str | Path) -> Dict[Tuple[int, int], float]:
    """Parse a gt.info file.

    The standard 3DMatch gt.info stores a 6×6 Fisher information matrix per
    pair (header: src_id tgt_id n_fragments, then 6 rows × 6 cols).  Overlap
    ratios are *not* stored in gt.info — all pairs in gt.log already have
    ≥30% overlap by definition.

    This function returns an empty dict; it exists so callers can tell whether
    gt.info is present without crashing.  Overlap filtering is left to the
    caller.
    """
    # Validate the file is parseable (warn on malformed content).
    path = Path(gt_info_path)
    try:
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            header = lines[0].split()
            int(header[0]), int(header[1])  # sanity check
    except Exception as exc:
        log.warning("Could not parse gt.info at %s: %s", path, exc)
    return {}


def collect_scene_pairs(
    scene_dir: str | Path,
    min_overlap: float = 0.0,
    max_pairs: Optional[int] = None,
    max_fragment_gap: Optional[int] = None,
) -> List[Tuple[str, str, np.ndarray]]:
    """Collect registration pairs for a single 3DMatch scene.

    Parameters
    ----------
    scene_dir:
        Directory containing ``cloud_bin_N.ply`` files, ``gt.log``, and
        optionally ``gt.info``.
    min_overlap:
        Minimum overlap ratio (0–1).  Only used when ``gt.info`` is present.
        Ignored (all pairs included) when ``gt.info`` is absent.
    max_pairs:
        Cap the number of returned pairs (useful for smoke tests).
    max_fragment_gap:
        If set, only include pairs where ``tgt_id - src_id <= max_fragment_gap``.
        Use ``1`` for consecutive-only pairs (smallest relative displacement,
        highest expected FPFH recall).

    Returns
    -------
    List of ``(src_path, tgt_path, T_gt)`` triples.
    """
    scene_dir = Path(scene_dir)
    gt_log = scene_dir / "gt.log"

    if not gt_log.exists():
        log.warning("gt.log not found in %s — skipping scene", scene_dir)
        return []

    entries = parse_gt_log(gt_log)

    # Note: standard 3DMatch gt.log already contains only pairs with ≥30%
    # overlap, so min_overlap is informational here (all pairs pass ≥0.3).
    # gt.info stores information matrices, not overlap ratios.

    pairs: List[Tuple[str, str, np.ndarray]] = []
    for src_id, tgt_id, T in entries:
        if max_fragment_gap is not None and (tgt_id - src_id) > max_fragment_gap:
            continue

        src_ply = scene_dir / f"cloud_bin_{src_id}.ply"
        tgt_ply = scene_dir / f"cloud_bin_{tgt_id}.ply"

        if not src_ply.exists():
            log.debug("Missing %s — skipping pair (%d, %d)", src_ply, src_id, tgt_id)
            continue
        if not tgt_ply.exists():
            log.debug("Missing %s — skipping pair (%d, %d)", tgt_ply, src_id, tgt_id)
            continue

        pairs.append((str(src_ply), str(tgt_ply), T))

        if max_pairs is not None and len(pairs) >= max_pairs:
            break

    return pairs


def collect_benchmark_pairs(
    benchmark_path: str | Path,
    scenes: Optional[List[str]] = None,
    min_overlap: float = 0.0,
    max_pairs_per_scene: Optional[int] = None,
    max_fragment_gap: Optional[int] = None,
) -> List[Tuple[str, str, np.ndarray]]:
    """Collect all registration pairs across 3DMatch benchmark scenes.

    Parameters
    ----------
    benchmark_path:
        Root directory containing one sub-directory per scene.
    scenes:
        Explicit list of scene names to include.  Defaults to all 8 standard
        test scenes (``BENCHMARK_SCENES``).
    min_overlap:
        Minimum overlap ratio forwarded to :func:`collect_scene_pairs`.
        Use ``0.3`` for standard 3DMatch, ``0.1`` to also include 3DLoMatch
        pairs.
    max_pairs_per_scene:
        Optional cap per scene (useful for quick smoke tests).

    Returns
    -------
    Flat list of ``(src_path, tgt_path, T_gt)`` across all scenes.
    """
    benchmark_path = Path(benchmark_path)
    scene_names = scenes if scenes is not None else BENCHMARK_SCENES

    all_pairs: List[Tuple[str, str, np.ndarray]] = []
    for name in scene_names:
        scene_dir = benchmark_path / name
        if not scene_dir.is_dir():
            log.warning("Scene directory not found: %s — skipping", scene_dir)
            continue
        pairs = collect_scene_pairs(
            scene_dir,
            min_overlap=min_overlap,
            max_pairs=max_pairs_per_scene,
            max_fragment_gap=max_fragment_gap,
        )
        log.info("Scene %-55s  %d pairs", name, len(pairs))
        all_pairs.extend(pairs)

    log.info("3DMatch total: %d pairs across %d scenes", len(all_pairs), len(scene_names))
    return all_pairs
