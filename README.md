# HairFlow

An open-vocabulary hair segmentation pipeline built on **YOLO-World** (detection) and **SAM** (segmentation). Ships with a small set of synthetic sample images and a bonus 3D strand visualizer.

## Repository layout

```
HairFlow/
├── scripts/
│   ├── segment.py              # hair segmentation (YOLO-World + SAM)
│   └── visualize_strands.py    # optional: 3D strand viewer for .npz files
├── synthetic_data/             # 10 sample RGB renders + metadata (from DiffLocks)
│   └── synthetic001/ … synthetic010/
├── segmented_images/           # segmentation outputs land here (gitignored)
└── checkpoints/                # model weights (gitignored, auto-downloaded)
```

## Installation

```bash
conda create -n hairflow python=3.10
conda activate hairflow
pip install opencv-python numpy ultralytics segment-anything matplotlib torch torchvision
```

First run of `segment.py` auto-downloads two checkpoints into `checkpoints/` (~2.7 GB total):
- `sam_vit_h_4b8939.pth` — SAM ViT-H
- `yolov8x-world.pt` — YOLO-World

## Hair segmentation

```bash
# Single image
python scripts/segment.py --path synthetic_data/synthetic005/rgb.png --save

# All sample images
python scripts/segment.py --all --save

# Dry run (no output saved)
python scripts/segment.py
```

**Flags**
- `--path <path>` — input image (default: `synthetic_data/synthetic001/rgb.png`)
- `--all` — process every `synthetic_data/synthetic*` folder
- `--save` — write outputs to `segmented_images/`

**Outputs** (per image, written to `segmented_images/`):
- `segmented_<name>_mask.png` — binary hair mask
- `segmented_<name>_overlay.png` — original image with green hair overlay

### How it works

1. **YOLO-World** runs open-vocabulary detection with the class `"hair"` and picks the highest-confidence box.
2. The box is padded by 6% for coverage.
3. **SAM** uses the padded box as a prompt and returns the best of its multi-mask outputs.
4. If YOLO finds no hair, the pipeline falls back to a centered box prompt.

## Strand visualization (optional)

For users working with DiffLocks-format strand data:

```bash
python scripts/visualize_strands.py --path synthetic_data/synthetic001/guide_strands.npz
python scripts/visualize_strands.py --path <file.npz> --fast   # downsample for big files
```

Opens an interactive matplotlib 3D window. The sample `synthetic_data/` folders ship with RGB + metadata only — the full `.npz` strand files are gitignored due to size. Grab them from the [DiffLocks dataset](https://radualexandru.github.io/difflocks/) if you want strands.

## Sample data

Each `synthetic_data/synthetic0XX/` contains:
- `rgb.png` — rendered hair portrait (segmentation input)
- `density.png`, `partition.png` — scalp density / parting maps
- `metadata.json` — Blender render parameters (curl, clump, BSDF, etc.)
- *(gitignored)* `guide_strands.npz`, `interpolated_strands.npz`, `full_strands.npz`, `cam.npz` — regenerate or download from DiffLocks

## Acknowledgments

- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) via Ultralytics — open-vocab detection
- [Segment Anything](https://github.com/facebookresearch/segment-anything) — Meta AI
- [DiffLocks](https://radualexandru.github.io/difflocks/) — synthetic hair renders + strand format
