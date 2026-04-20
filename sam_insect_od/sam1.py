"""
insect_detection.py
-------------------
Detect insects in images using SAM (Segment Anything Model) and return
bounding boxes ready to crop and pass to a downstream classifier.

Usage:
    python insect_detection.py --image path/to/image.jpg
    python insect_detection.py --image path/to/image.jpg --output detections.json
    python insect_detection.py --image path/to/image.jpg --visualize

Download SAM checkpoint first:
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# ── Configuration ─────────────────────────────────────────────────────────────

# Model type: "vit_b" (fast), "vit_l" (balanced), "vit_h" (best quality)
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"

# Large images are resized before SAM to avoid GPU OOM.
# Bounding boxes are then scaled back to original pixel coordinates.
# 1024 works well on a ~6 GB GPU; lower to 768 if you still OOM.
SAM_MAX_SIDE = 1024

# Mask generation settings — tweak these to tune recall vs. noise
SAM_POINTS_PER_SIDE = 32       # Grid density; raise (e.g. 48) for small/dense insects
SAM_PRED_IOU_THRESH = 0.86     # Mask quality; lower to catch more candidates
SAM_STABILITY_SCORE_THRESH = 0.92
SAM_MIN_MASK_REGION_AREA = 200  # Minimum mask size in pixels; lower for tiny insects

# Size filter: keep only masks whose bounding box falls within these relative
# areas (fraction of total image area). Drops full-image and dust-spec masks.
MIN_RELATIVE_AREA = 0.0005   # 0.05 % of image
MAX_RELATIVE_AREA = 0.50     # 50 % of image

# Non-Maximum Suppression IoU threshold
NMS_IOU_THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model loading ──────────────────────────────────────────────────────────────

def load_sam(checkpoint: str = SAM_CHECKPOINT, model_type: str = SAM_MODEL_TYPE):
    if not Path(checkpoint).exists():
        print(f"[ERROR] SAM checkpoint not found: {checkpoint}")
        print("Download it with:")
        print(f"  wget https://dl.fbaipublicfiles.com/segment_anything/{Path(checkpoint).name}")
        sys.exit(1)

    print(f"Loading SAM ({model_type}) on {DEVICE}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=DEVICE)

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=SAM_POINTS_PER_SIDE,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
        min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
    )
    print("SAM ready.\n")
    return generator


# ── Core detection ─────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load image from disk and return as RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def mask_to_bbox(mask: np.ndarray):
    """Convert a boolean mask to (x, y, w, h)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)


def bbox_iou(b1, b2) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list, iou_threshold: float = NMS_IOU_THRESHOLD) -> list:
    """Greedy NMS; boxes sorted by SAM predicted IoU (higher = better mask)."""
    boxes = sorted(boxes, key=lambda b: b["sam_score"], reverse=True)
    kept = []
    for box in boxes:
        if all(bbox_iou(box["bbox"], k["bbox"]) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def resize_for_sam(image: np.ndarray, max_side: int = SAM_MAX_SIDE):
    """
    Downscale image so its longest side is at most max_side.
    Returns (resized_image, scale_factor). scale_factor < 1 means it was shrunk.
    """
    H, W = image.shape[:2]
    longest = max(H, W)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    new_W, new_H = int(W * scale), int(H * scale)
    resized = np.array(Image.fromarray(image).resize((new_W, new_H), Image.LANCZOS))
    print(f"  Resized {W}x{H} → {new_W}x{new_H} (scale={scale:.3f}) to fit GPU memory")
    return resized, scale


def detect_insects(image: np.ndarray, generator: SamAutomaticMaskGenerator) -> list:
    """
    Run SAM on an image and return filtered bounding boxes.

    Returns a list of dicts:
        [{"bbox": (x, y, w, h), "sam_score": float}, ...]

    bbox is in ORIGINAL pixel coordinates: x/y = top-left corner, w/h = width/height.
    """
    H, W = image.shape[:2]
    total_area = H * W

    print(f"Image size: {W}x{H}  |  Running SAM...")

    # Resize for GPU, run SAM, then scale boxes back to original resolution
    sam_image, scale = resize_for_sam(image)
    masks = generator.generate(sam_image)
    print(f"  {len(masks)} candidate masks generated")

    candidates = []
    for m in masks:
        sx, sy, sw, sh = mask_to_bbox(m["segmentation"])

        # Scale bbox back to original image coordinates
        if scale < 1.0:
            x = int(sx / scale)
            y = int(sy / scale)
            w = int(sw / scale)
            h = int(sh / scale)
        else:
            x, y, w, h = sx, sy, sw, sh

        relative_area = (w * h) / total_area

        # Filter out masks that are too small or too large
        if relative_area < MIN_RELATIVE_AREA or relative_area > MAX_RELATIVE_AREA:
            continue

        candidates.append({
            "bbox": (x, y, w, h),
            "sam_score": float(m["predicted_iou"]),
        })

    print(f"  {len(candidates)} masks after size filtering")

    detections = nms(candidates)
    print(f"  {len(detections)} detections after NMS")
    return detections


# ── Crops ─────────────────────────────────────────────────────────────────────

def crop_detections(image: np.ndarray, detections: list, padding: int = 10) -> list:
    """
    Crop each detection from the image (with optional padding).

    Returns a list of RGB numpy arrays.
    """
    H, W = image.shape[:2]
    crops = []
    for det in detections:
        x, y, w, h = det["bbox"]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(W, x + w + padding)
        y2 = min(H, y + h + padding)
        crops.append(image[y1:y2, x1:x2])
    return crops


def save_crops(crops: list, output_dir: str, stem: str = "insect"):
    """Save crop images to a directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, crop in enumerate(crops):
        path = out / f"{stem}_{i+1:03d}.jpg"
        Image.fromarray(crop).save(path)
        paths.append(str(path))
    print(f"Saved {len(crops)} crop(s) to '{output_dir}/'")
    return paths


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(image: np.ndarray, detections: list, output_path: str = None):
    """Draw bounding boxes on the image and show/save it."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    colors = plt.cm.tab10.colors
    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        color = colors[i % len(colors)]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 6, f"#{i+1}", color="white", fontsize=9, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"))

    title = f"SAM: {len(detections)} insect(s) detected" if detections else "SAM: No insects detected"
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to '{output_path}'")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Detect insects in an image using SAM.")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--checkpoint", default=SAM_CHECKPOINT, help="SAM checkpoint path")
    p.add_argument("--model-type", default=SAM_MODEL_TYPE, choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--output", default=None, help="Save detections as JSON (optional)")
    p.add_argument("--save-crops", default=None, metavar="DIR",
                   help="Save cropped insect images to this directory")
    p.add_argument("--crop-padding", type=int, default=10,
                   help="Pixel padding around crops (default: 10)")
    p.add_argument("--visualize", action="store_true", help="Show/save visualization")
    p.add_argument("--vis-output", default=None,
                   help="Path to save visualization image (shows interactively if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    generator = load_sam(args.checkpoint, args.model_type)
    image = load_image(args.image)
    detections = detect_insects(image, generator)

    # Print results
    print(f"\nDetections ({len(detections)}):")
    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        print(f"  [{i+1}] x={x}, y={y}, w={w}, h={h}  (sam_score={det['sam_score']:.4f})")

    # Save JSON
    if args.output:
        export = [{"bbox": list(d["bbox"]), "sam_score": round(d["sam_score"], 4)}
                  for d in detections]
        with open(args.output, "w") as f:
            json.dump(export, f, indent=2)
        print(f"\nDetections saved to '{args.output}'")

    # Save crops
    if args.save_crops:
        crops = crop_detections(image, detections, padding=args.crop_padding)
        stem = Path(args.image).stem
        save_crops(crops, args.save_crops, stem=stem)

    # Visualize
    if args.visualize:
        visualize(image, detections, output_path=args.vis_output)


if __name__ == "__main__":
    main()