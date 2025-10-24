# HairFlow

A hair segmentation and visualization pipeline using YOLO-World and SAM (Segment Anything Model) for automated hair detection and segmentation.

## Installation
```bash
# Create a conda environment (recommended)
conda create -n hairflow python=3.10
conda activate hairflow

# Install dependencies
pip install opencv-python numpy ultralytics segment-anything matplotlib
pip install torch torchvision  # if not already installed
```

**Model Checkpoints**: The segmentation script will automatically download required model weights (~2.5GB total) to the `checkpoints/` directory on first run.

## Sample Data

### Synthetic Data Structure
```
synthetic_data/
├── synthetic001/
│   ├── rgb.png              # Input RGB image
│   ├── guide_strands.npz    # Main/guide hair strands
│   ├── interpolated_strands.npz  # Interpolated strands
│   └── full_strands.npz     # Complete strand data
├── synthetic002/
└── ...
```

**Note**: This repository includes sample RGB images in the `synthetic_data/` folders that you can use immediately for testing the segmentation pipeline.

### Additional Validation Data

For more extensive validation data, the DiffLocks validation set provides high-quality hair capture data.

## Usage

### Hair Segmentation

Segment hair from images using YOLO-World for detection and SAM for precise segmentation.
```bash
# Process default image without saving
python scripts/segment.py

# Process a specific image and save results
python scripts/segment.py --path synthetic_data/synthetic005/rgb.png --save

# Process ALL synthetic images and save
python scripts/segment.py --all --save
```

**Flags**:
* `--path <path>` - Path to input image (default: `synthetic_data/synthetic001/rgb.png`)
* `--save` - Save segmentation outputs to `segmented_images/` folder
* `--all` - Process all images in `synthetic_data/` folder

**Output**: Saved as `segmented_synthetic_001_mask.png` and `segmented_synthetic_001_overlay.png`

### Strand Visualization

Visualize 3D hair strand data from `.npz` files in an interactive matplotlib window.
```bash
# Visualize guide strands
python scripts/visualize_synthetic_data.py --path synthetic_data/synthetic001/guide_strands.npz

# Visualize interpolated strands
python scripts/visualize_synthetic_data.py --path synthetic_data/synthetic001/interpolated_strands.npz

# Visualize full strands
python scripts/visualize_synthetic_data.py --path synthetic_data/synthetic001/full_strands.npz
```

**Interactive Controls**:
* Rotate: Left-click and drag
* Zoom: Scroll wheel
* Pan: Middle-click and drag

**Strand Types**:
* `guide_strands.npz`: Main strands that define overall hair structure
* `interpolated_strands.npz`: Strands generated between guide strands
* `full_strands.npz`: Complete set (guide + interpolated)

## Pipeline Overview

The segmentation pipeline works in three steps:

1. **YOLO-World Detection**: Finds "hair" in the image using open-vocabulary detection
2. **Bounding Box Refinement**: Pads detected box by 6% for better coverage
3. **SAM Segmentation**: Uses the box as a prompt to generate a precise hair mask

If YOLO doesn't detect hair, it automatically falls back to a center-box prompt.

## Acknowledgments

* **YOLO-World**: Ultralytics YOLO-World for open-vocabulary object detection
* **SAM**: Meta's Segment Anything Model for segmentation
* **DiffLocks**: For validation dataset and hair rendering research
