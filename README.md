# 3D Scene Matching Pipeline

A modular, configurable pipeline for **3D point cloud registration** and **outdoor localization**. Designed for noisy, sparse stereo-generated and LiDAR point clouds in large-scale scenes, including data captured in absolute coordinate systems (UTM/ENU).

---

## Overview

The pipeline supports two primary modes:

| Mode | Description |
|------|-------------|
| **Benchmark** | Register a source cloud to a target cloud and evaluate against a ground-truth transform |
| **Localization** | Estimate the position of a local query scan within a large global reference map |

---

## Project Structure

```
3d_scene_matching/
├── run.py                          # CLI entry point
├── config.yaml                     # Full configuration schema
├── requirements.txt
└── pipeline/
    ├── pipeline.py                 # Orchestrator (Pipeline class)
    ├── data/
    │   ├── loader.py               # PLY / PCD / NPZ → PointCloud
    │   └── preprocessor.py         # Voxel downsample + normal estimation
    ├── models/
    │   ├── base.py                 # PointCloud, FeatureOutput, BaseModel, PairModel
    │   ├── fpfh.py                 # Open3D FPFH (classical, no GPU required)
    │   ├── shot.py                 # SHOT descriptor + ISS keypoints
    │   ├── fgr.py                  # Fast Global Registration model
    │   ├── gicp.py                 # Generalized ICP model
    │   ├── color_icp.py            # Color ICP model
    │   ├── fcgf.py                 # FCGF deep descriptor (requires MinkowskiEngine)
    │   ├── geotransformer.py       # GeoTransformer pair model
    │   └── predator.py             # OverlapPredator pair model
    ├── matching/
    │   └── matcher.py              # Feature matching, FGR, RANSAC, multiscale GICP
    ├── evaluation/
    │   └── metrics.py              # RRE / RTE / Inlier Ratio / Chamfer Distance
    ├── localization/
    │   ├── global_searcher.py      # Tile global map + Z-histogram pre-screening
    │   ├── result.py               # LocalizationResult dataclass
    │   └── uncertainty.py          # Multi-run RANSAC → σ_East, σ_North
    ├── visualization/
    │   └── visualizer.py           # Open3D before/after viewer
    └── utils/
        ├── reporting.py            # Tabulate table + CSV export
        └── downloader.py           # Pretrained weight download helpers
```

---

## Installation

### Core dependencies (classical methods only)

```bash
pip install open3d numpy scipy pyyaml tqdm pandas tabulate
```

### Deep learning backends (optional)

| Model | Dependency | Notes |
|-------|------------|-------|
| FCGF | MinkowskiEngine + PyTorch | Build from source |
| GeoTransformer | `geotransformer` package | Install from official repo |
| OverlapPredator | `OverlapPredator` repo | Clone `prs-eth/OverlapPredator` |

---

## Quick Start

### Benchmark — register a pair with ground truth evaluation

```bash
python run.py --src data/scene_a.npz --tgt data/scene_b.npz --gt data/T_gt.npy --models fpfh
```

### Inference — register without ground truth

```bash
python run.py --src data/a.ply --tgt data/b.ply --models fpfh,fgr
```

### Localization — find a local scan within a global map

```bash
python run.py --local data/local_scan.ply --global data/global_map.ply --models fpfh
```

### Full benchmark from config

```bash
python run.py                          # uses config.yaml
python run.py --config my_config.yaml  # custom config
```

### Visualize results

```bash
python run.py --src data/a.ply --tgt data/b.ply --models fpfh --visualize
```

### Download pretrained weights

```bash
python run.py --download-weights --models fcgf,geotransformer
```

---

## Registration Methods

### Classical Feature-Based

#### FPFH — Fast Point Feature Histograms
- Computes 33-dimensional local geometry descriptors around each keypoint
- Captures curvature and surface normal distributions within a neighbourhood
- **Best for**: Large outdoor scenes, no GPU required, interpretable
- **Pipeline**: Voxel downsample → normal estimation → FPFH extraction → feature matching → global registration → ICP refinement

#### SHOT — Signature of Histograms of Orientations
- Local geometry descriptor based on normal histogram binning in spherical shells
- Combined with ISS (Intrinsic Shape Signatures) keypoint detection
- **Best for**: Indoor scenes with distinctive geometric features

#### FGR — Fast Global Registration
- Uses a **Geman-McClure robust loss** instead of hard inlier/outlier classification
- Significantly more robust to noise and outlier correspondences than RANSAC
- **Best for**: Noisy stereo-generated sparse point clouds; faster than RANSAC at equivalent accuracy

#### GICP — Generalized ICP
- Extends point-to-point ICP with **per-point covariance matrices**
- More robust to uneven point density and depth noise than standard ICP
- Used both as a global initializer (with FPFH for initial alignment) and as a refinement step

#### Color ICP
- Augments geometric ICP with photometric (RGB) consistency terms
- **Best for**: RGB-D data with reliable color information

### Deep Learning Models

#### FCGF — Fully Convolutional Geometric Features
- Sparse 3D convolutions (MinkowskiEngine) produce dense 32-dim descriptors
- Learned from 3DMatch indoor dataset; strong generalization to outdoor
- Requires: MinkowskiEngine, PyTorch, pretrained weights (`weights/fcgf_3dmatch.pth`)

#### GeoTransformer
- Transformer-based architecture with geometric self-attention
- State-of-the-art on 3DMatch and KITTI benchmarks
- Requires: `geotransformer` package, pretrained weights

#### OverlapPredator
- Overlap-aware registration: predicts matchability scores per keypoint
- Focuses feature matching on the overlapping region
- Requires: OverlapPredator repo, pretrained weights

---

## Global Registration Strategies

Controlled via `config.yaml` under `registration.global_method`:

| Method | When to Use |
|--------|-------------|
| `direct_gicp` | Same coordinate system (e.g. both UTM/ENU) — clouds already roughly aligned. Skips feature extraction; uses multiscale GICP from identity initialization |
| `fgr` | Different coordinate systems or large unknown offset; robust to stereo noise |
| `ransac` | Classical fallback; slower and less robust than FGR for noisy data |

### Multiscale GICP (`direct_gicp`)
Designed specifically for large-scale outdoor data in absolute coordinate systems:
1. Centers both clouds to the source centroid to eliminate UTM floating-point precision issues
2. Runs GICP at geometrically-spaced correspondence distances (e.g. 5 m → 0.5 m over 4 scales)
3. Recovers the final transform in the original coordinate frame
4. `translation_only` mode strips spurious rotation — critical when both clouds share a CRS (a small erroneous rotation × large UTM coordinates = kilometre-scale translation error)

---

## Refinement Methods

Controlled via `registration.refinement`:

| Method | Description |
|--------|-------------|
| `gicp` | Generalized ICP — recommended for noisy/sparse outdoor clouds |
| `icp` | Standard point-to-point ICP |
| `color_icp` | Color-assisted ICP (RGB-D only) |

---

## Feature Matching

Controlled via `matching.method`:

| Method | Description |
|--------|-------------|
| `mutual_nn` | Mutual nearest-neighbour — high precision, fewer correspondences |
| `ratio_test` | Lowe ratio test — more correspondences, better for low-overlap or noisy data |
| `combined` | Intersection of mutual NN and ratio test — highest precision |

Optional FAISS acceleration: `matching.use_faiss: true` (requires `faiss-cpu` or `faiss-gpu`).

---

## Localization Mode

Estimates the East/North position of a local query scan within a large global reference map.

### Pipeline
1. **Spatial crop**: Both clouds clipped to their shared extent (+ configurable buffer) to avoid processing areas that cannot overlap
2. **Tiling** (optional): Global map split into overlapping tiles when point count exceeds `max_direct_points`
3. **Z-histogram pre-screening**: Candidate tiles ranked by Z-profile similarity before full registration
4. **Registration**: One of `direct_gicp` / `fgr` / `ransac` per tile
5. **Candidate validation**: Transformed centroid checked against original global map bounds (+ `bounds_buffer_m`)
6. **Uncertainty estimation**: Multi-run RANSAC → σ_East, σ_North (for RANSAC/FGR paths)
7. **Output**: East/North position in metres, uncertainty, match score, CSV

### Output table example
```
Localization Results
╭─────────┬────────────────┬─────────────────┬──────────────┬───────────────┬─────────┬────────────╮
│ Model   │   East pos (m) │   North pos (m) │ σ East (m)   │ σ North (m)   │   Score │   Time (s) │
├─────────┼────────────────┼─────────────────┼──────────────┼───────────────┼─────────┼────────────┤
│ fpfh    │          9.376 │          -1.682 │ ±0.500       │ ±0.500        │    0.26 │      10.11 │
╰─────────┴────────────────┴─────────────────┴──────────────┴───────────────┴─────────┴────────────╯
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RRE** | Relative Rotation Error — geodesic angle between estimated and GT rotation (degrees) |
| **RTE** | Relative Translation Error — L2 distance between estimated and GT translation (metres) |
| **Success** | RRE < threshold AND RTE < threshold (configurable per dataset) |
| **Inlier Ratio** | Fraction of correspondences that are geometrically consistent under GT transform |
| **Chamfer Distance** | Mean symmetric nearest-neighbour distance after alignment (optional; expensive) |

Default thresholds: RRE < 15°, RTE < 0.30 m (3DMatch standard). Adjust in `config.yaml` under `evaluation`.

---

## Supported File Formats

| Format | Support |
|--------|---------|
| `.ply` | Read/write via Open3D |
| `.pcd` | Read via Open3D |
| `.npz` | Read/write via NumPy — expects `xyz` array; optionally `normals`, `colors` |
| `.npy` | Ground-truth transforms (4×4 matrix) |
| `.txt` / `.csv` | Ground-truth transforms (space/comma delimited 4×4) |

---

## Configuration Reference

Key sections in `config.yaml`:

```yaml
pipeline:
  mode: "benchmark"          # "benchmark" | "inference"
  models: ["fpfh", "fgr"]
  icp_refinement: true
  output_dir: "results/"

models:
  fpfh:
    voxel_size: 0.5          # metres — tune to scene scale
    feature_radius: 5.0      # descriptor neighbourhood radius
    normal_radius: 1.0       # normal estimation radius
    remove_outliers: true    # statistical outlier removal

registration:
  global_method: "direct_gicp"   # "direct_gicp" | "fgr" | "ransac"
  refinement: "gicp"             # "gicp" | "icp" | "color_icp"
  direct_gicp:
    coarse_dist: 5.0   # starting correspondence distance (m)
    fine_dist: 0.5     # final correspondence distance (m)
    n_scales: 4        # number of distance steps
    max_iter: 300      # GICP iterations per scale

matching:
  method: "ratio_test"
  ratio_threshold: 0.90

evaluation:
  rre_threshold: 15.0   # degrees
  rte_threshold: 0.30   # metres

localization:
  tile_size: 20.0
  tile_overlap: 0.3
  n_ransac_runs: 10
  bounds_buffer_m: 50.0  # tolerance when validating edge-of-map results
  spatial_crop:
    enabled: true
    buffer_m: 50.0
```

---

## Design Notes

- **float64 throughout**: All XYZ coordinates are stored as `float64`. Using `float32` on UTM-scale coordinates (~10⁶ m) causes catastrophic precision loss in `mean()` and covariance computations.
- **Centering for GICP**: Large absolute coordinates (UTM: X~700 000, Y~5 600 000) are internally re-centered before GICP to avoid floating-point instability in covariance matrix computation.
- **BaseModel vs PairModel**: Descriptor models (`fpfh`, `fcgf`, `shot`) implement `extract_features(pcd) → FeatureOutput`; end-to-end pair models (`geotransformer`, `predator`) implement `register_pair(src, tgt) → 4×4`. The pipeline dispatches automatically.
- **Reproducibility**: RANSAC uses `o3d.utility.random.seed(42)` + `np.random.seed(42)`.
