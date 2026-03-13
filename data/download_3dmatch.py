#!/usr/bin/env python3
"""Download 3DMatch test scenes from http://3dmatch.cs.princeton.edu/.

Usage
-----
    python data/download_3dmatch.py
    python data/download_3dmatch.py --dest data/3dmatch
    python data/download_3dmatch.py --scenes redkitchen sun3d-hotel_uc-scan3
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "http://3dvision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments"

# Mapping short alias → full scene name (for convenience flags like --scenes redkitchen)
ALIAS = {
    "redkitchen": "7-scenes-redkitchen",
    "home_at": "sun3d-home_at-home_at_scan1_2013_jan_1",
    "home_md": "sun3d-home_md-home_md_scan9_2012_sep_30",
    "hotel_uc": "sun3d-hotel_uc-scan3",
    "hotel_umd1": "sun3d-hotel_umd-maryland_hotel1",
    "hotel_umd3": "sun3d-hotel_umd-maryland_hotel3",
    "studyroom": "sun3d-mit_76_studyroom-76-1studyroom2",
    "lab_hj": "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
}

ALL_SCENES = list(ALIAS.values())


def _progress_hook(dest_name: str):
    """Return a urllib reporthook that prints download progress."""
    def hook(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            bar = "#" * int(pct / 2)
            print(f"\r  {dest_name}: [{bar:<50}] {pct:5.1f}%", end="", flush=True)
        else:
            mb = downloaded / 1_048_576
            print(f"\r  {dest_name}: {mb:.1f} MB downloaded", end="", flush=True)
    return hook


def download_scene(scene: str, dest: Path) -> None:
    """Download and extract a single 3DMatch scene."""
    scene_dir = dest / scene
    if scene_dir.exists() and any(scene_dir.glob("cloud_bin_*.ply")):
        print(f"  {scene}: already downloaded — skipping")
        return

    dest.mkdir(parents=True, exist_ok=True)
    archive_name = f"{scene}.zip"
    url = f"{BASE_URL}/{archive_name}"
    local_archive = dest / archive_name

    print(f"\nDownloading {scene} ...")
    try:
        urllib.request.urlretrieve(url, local_archive, _progress_hook(archive_name))
        print()  # newline after progress bar
    except Exception as exc:
        print(f"\n  ERROR downloading {scene}: {exc}", file=sys.stderr)
        if local_archive.exists():
            local_archive.unlink()
        return

    print(f"  Extracting {archive_name} ...")
    import zipfile
    with zipfile.ZipFile(local_archive) as zf:
        zf.extractall(dest)
    local_archive.unlink()
    print(f"  {scene}: done  ({sum(1 for _ in scene_dir.glob('cloud_bin_*.ply'))} fragments)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download 3DMatch test scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dest",
        default="data/3dmatch",
        help="Destination directory (default: data/3dmatch)",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        metavar="SCENE",
        help=(
            "Scene names or short aliases to download. "
            "Defaults to all 8 scenes. "
            f"Short aliases: {', '.join(ALIAS)}"
        ),
    )
    args = parser.parse_args()

    dest = Path(args.dest)

    # Resolve scene list
    if args.scenes:
        scenes = [ALIAS.get(s, s) for s in args.scenes]
    else:
        scenes = ALL_SCENES

    print(f"Downloading {len(scenes)} scene(s) to '{dest}' ...")
    for scene in scenes:
        download_scene(scene, dest)

    print(
        "\n"
        "NOTE: Ground-truth transforms (gt.log / gt.info) are distributed\n"
        "      separately. Download 'geometric-registration-benchmark.zip'\n"
        f"      from {BASE_URL}/ and extract into '{dest}'.\n"
        "      Each scene sub-directory should then contain gt.log and gt.info."
    )


if __name__ == "__main__":
    main()
