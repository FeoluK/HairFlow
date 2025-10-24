# scripts/segment.py
# YOLO weights auto-download via Ultralytics; SAM saved to ./checkpoints.

import os, urllib.request
import cv2, numpy as np
import argparse
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# -------- EDIT THIS --------
IMG_PATH = "synthetic_data/synthetic001/rgb.png"
SEG_DIR = "segmented_images"
CHECKPOINT_DIR = "checkpoints"
SAM_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sam_vit_h_4b8939.pth")
YOLO_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "yolov8x-world.pt")
SAM_MODEL_TYPE = "vit_h"
# ---------------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def download_if_missing(url, dest):
    if not os.path.exists(dest):
        ensure_dir(os.path.dirname(dest))
        print(f"Downloading {os.path.basename(dest)} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved {dest}")

def setup_checkpoints():
    # Download SAM checkpoint
    download_if_missing(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        SAM_CHECKPOINT
    )
    
    # Download YOLO checkpoint
    download_if_missing(
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-world.pt",
        YOLO_CHECKPOINT
    )

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"cv2 failed to read {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb

# --- Lazy-load models so we only initialize once ---
_yolo_model = None
def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        print("Loading YOLO-World model...")
        _yolo_model = YOLO(YOLO_CHECKPOINT)
    return _yolo_model

_sam_predictor = None
def get_sam_predictor(img_rgb=None):
    global _sam_predictor
    if _sam_predictor is None:
        print("Loading SAM model...")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        _sam_predictor = SamPredictor(sam)
    if img_rgb is not None:
        _sam_predictor.set_image(img_rgb)
    return _sam_predictor

def run_yolo_world(img_rgb):
    model = get_yolo_model()
    model.set_classes(["hair"])  # open-vocab target
    res = model.predict(img_rgb, conf=0.05, iou=0.5, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    return xyxy[int(np.argmax(conf))]

def pad_box(box, w, h, pad=0.06):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 -= pad*bw; y1 -= pad*bh; x2 += pad*bw; y2 += pad*bh
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def run_sam(img_rgb, box_xyxy):
    predictor = get_sam_predictor(img_rgb)
    masks, scores, _ = predictor.predict(box=box_xyxy[None, :], multimask_output=True)
    return masks[int(np.argmax(scores))].astype(np.uint8)

def save_outputs(img_bgr, mask01, out_base):
    ensure_dir(SEG_DIR)
    mask_path = os.path.join(SEG_DIR, f"segmented_{out_base}_mask.png")
    overlay_path = os.path.join(SEG_DIR, f"segmented_{out_base}_overlay.png")

    cv2.imwrite(mask_path, (mask01 * 255).astype(np.uint8))

    # Create overlay without using the 'mask' parameter
    overlay = img_bgr.copy()
    tint = np.zeros_like(img_bgr)
    tint[:,:,1] = 255  # green
    
    # Apply the tint only where mask is 1
    alpha = 0.45
    mask_bool = mask01.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(
        img_bgr[mask_bool], 1.0, 
        tint[mask_bool], alpha, 
        0
    )
    
    cv2.imwrite(overlay_path, overlay)
    print(f"Saved:\n  {mask_path}\n  {overlay_path}")

def main():
    parser = argparse.ArgumentParser(description="Segment hair from images using YOLO + SAM")
    parser.add_argument("--save", action="store_true", help="Save the segmented images to disk")
    parser.add_argument("--path", type=str, default=IMG_PATH, help="Path to input image")
    parser.add_argument("--all", action="store_true", help="Process all images in synthetic_data folder")
    args = parser.parse_args()
    
    setup_checkpoints()

    # Determine which images to process
    if args.all:
        # Find all synthetic folders
        synthetic_folders = []
        if os.path.exists("synthetic_data"):
            synthetic_folders = sorted([
                os.path.join("synthetic_data", d) 
                for d in os.listdir("synthetic_data") 
                if os.path.isdir(os.path.join("synthetic_data", d)) and d.startswith("synthetic")
            ])
        
        if not synthetic_folders:
            print("No synthetic folders found in synthetic_data/")
            return
        
        print(f"Found {len(synthetic_folders)} synthetic folders to process")
        
        for folder in synthetic_folders:
            img_path = os.path.join(folder, "rgb.png")
            if not os.path.exists(img_path):
                print(f"Skipping {folder}: rgb.png not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing: {img_path}")
            print(f"{'='*60}")
            
            try:
                process_single_image(img_path, args.save)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Completed processing {len(synthetic_folders)} images")
        print(f"{'='*60}")
    else:
        # Process single image
        process_single_image(args.path, args.save)

def process_single_image(img_path, should_save):
    """Process a single image through the segmentation pipeline"""
    img_bgr, img_rgb = load_image(img_path)
    h, w = img_rgb.shape[:2]

    box = run_yolo_world(img_rgb)
    if box is None:
        print("YOLO didn't find 'hair' — falling back to a center box for SAM.")
        cx, cy = w//2, h//2
        box = np.array([cx - w*0.15, cy - h*0.15, cx + w*0.15, cy + h*0.15], dtype=np.float32)

    box = pad_box(box, w, h, pad=0.06)
    mask01 = run_sam(img_rgb, box)

    # Extract synthetic number from path (e.g., synthetic001 -> 001)
    base_name = os.path.basename(os.path.dirname(img_path))
    if base_name.startswith("synthetic"):
        number = base_name.replace("synthetic", "")
        out_base = f"synthetic_{number}"
    else:
        # Fallback to original filename if not in expected format
        out_base = os.path.splitext(os.path.basename(img_path))[0]
    
    if should_save:
        save_outputs(img_bgr, mask01, out_base)
    else:
        print("Segmentation complete. Use --save flag to save outputs.")
        print(f"Would save as: segmented_{out_base}_mask.png and segmented_{out_base}_overlay.png")

if __name__ == "__main__":
    main()