"""
insect_detection.py
-------------------
Detect insects using SAM 3 (Segment Anything Model 3) with text prompts.
SAM 3 natively finds ALL instances of a concept from a short text phrase —
no separate classifier, no C++ extensions, no custom CUDA ops.

Install:
    pip install torch torchvision
    pip install git+https://github.com/huggingface/transformers.git  # SAM3 needs latest transformers
    pip install Pillow tqdm matplotlib

    Note: You need to accept the Meta license on HuggingFace before the model
    will download. Visit https://huggingface.co/facebook/sam3 and click "Agree".
    Then log in once with: huggingface-cli login

Usage:
    # Single image with visualization
    python insect_detection.py --image photo.jpg --visualize

    # Evaluate against your labeled val set
    python insect_detection.py --split val

    # Save crops for your downstream classifier
    python insect_detection.py --image photo.jpg --save-crops ./crops
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_ROOT = "./df6"
MODEL_NAME   = "facebook/sam3"

# Text prompt — SAM 3 finds ALL instances of this concept in the image.
# You can make it more specific, e.g. "insect on a leaf" if you get too many
# false positives from non-insect objects.
TEXT_PROMPT  = "insect"

# Confidence threshold: raise to reduce false positives, lower to catch more insects
SCORE_THRESHOLD = 0.50
MASK_THRESHOLD  = 0.50

# IoU threshold for evaluation matching
EVAL_IOU_THRESHOLD = 0.50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    from transformers import Sam3Model, Sam3Processor
    print(f"Loading SAM 3 ({MODEL_NAME}) on {DEVICE}...")
    print("(First run will download ~3 GB of weights from HuggingFace)\n")
    processor = Sam3Processor.from_pretrained(MODEL_NAME)
    model = Sam3Model.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print("SAM 3 ready.\n")
    return processor, model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(processor, model, image_path: str):
    """
    Run SAM 3 on one image with the text prompt.
    Returns:
        boxes  — (N, 4) float32 array of absolute (x1, y1, x2, y2) pixel coords
        scores — (N,)   float32 confidence scores
        (H, W) — original image size
    """
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    inputs = processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=SCORE_THRESHOLD,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    if len(results["boxes"]) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), (H, W)

    boxes  = results["boxes"].cpu().numpy().astype(np.float32)
    scores = results["scores"].cpu().numpy().astype(np.float32)
    masks  = results["masks"]  # kept for optional use but not returned

    return boxes, scores, (H, W)


# ── Ground truth ──────────────────────────────────────────────────────────────

def load_gt_boxes(label_path: str, H: int, W: int) -> np.ndarray:
    """Parse YOLO-format label file → absolute (x1, y1, x2, y2)."""
    if not Path(label_path).exists():
        return np.empty((0, 4), dtype=np.float32)
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            boxes.append([
                (cx - bw / 2) * W, (cy - bh / 2) * H,
                (cx + bw / 2) * W, (cy + bh / 2) * H,
            ])
    return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)


# ── Evaluation ────────────────────────────────────────────────────────────────

def box_iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    ix1 = np.maximum(pred[:, 0:1], gt[:, 0])
    iy1 = np.maximum(pred[:, 1:2], gt[:, 1])
    ix2 = np.minimum(pred[:, 2:3], gt[:, 2])
    iy2 = np.minimum(pred[:, 3:4], gt[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return np.where(union > 0, inter / union, 0).astype(np.float32)


def compute_ap(all_scores, all_tp, all_fp, n_gt):
    if n_gt == 0 or len(all_scores) == 0:
        return 0.0, 0.0, 0.0
    order = np.argsort(-np.array(all_scores))
    tp = np.cumsum(np.array(all_tp)[order])
    fp = np.cumsum(np.array(all_fp)[order])
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (n_gt + 1e-9)
    ap = sum(
        precision[recall >= t].max() if np.any(recall >= t) else 0.0
        for t in np.linspace(0, 1, 11)
    ) / 11.0
    return float(precision[-1]), float(recall[-1]), float(ap)


def run_evaluation(processor, model, split: str = "val"):
    img_dir   = Path(DATASET_ROOT) / "images" / split
    label_dir = Path(DATASET_ROOT) / "labels" / split

    image_paths = sorted(img_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {img_dir}")

    print(f"Evaluating {len(image_paths)} images ({split} split)...\n")

    all_scores, all_tp, all_fp = [], [], []
    n_gt_total = 0

    for img_path in tqdm(image_paths, desc="Detecting"):
        label_path = label_dir / (img_path.stem + ".txt")
        pred_boxes, scores, (H, W) = predict(processor, model, str(img_path))
        gt_boxes = load_gt_boxes(str(label_path), H, W)

        n_gt_total += len(gt_boxes)
        matched_gt = set()
        ious = box_iou_matrix(pred_boxes, gt_boxes)

        for i, score in enumerate(scores):
            all_scores.append(float(score))
            if len(gt_boxes) > 0:
                best_gt = int(np.argmax(ious[i]))
                if ious[i, best_gt] >= EVAL_IOU_THRESHOLD and best_gt not in matched_gt:
                    all_tp.append(1); all_fp.append(0)
                    matched_gt.add(best_gt)
                else:
                    all_tp.append(0); all_fp.append(1)
            else:
                all_tp.append(0); all_fp.append(1)

    precision, recall, ap = compute_ap(all_scores, all_tp, all_fp, n_gt_total)

    metrics = {
        "model":                MODEL_NAME,
        "text_prompt":          TEXT_PROMPT,
        "split":                split,
        "n_images":             len(image_paths),
        "n_gt_boxes":           n_gt_total,
        "score_threshold":      SCORE_THRESHOLD,
        "eval_iou_threshold":   EVAL_IOU_THRESHOLD,
        "precision":            round(precision, 4),
        "recall":               round(recall, 4),
        f"AP@{EVAL_IOU_THRESHOLD}": round(ap, 4),
    }

    print("\n── Evaluation Results ───────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<26}: {v}")
    print("─────────────────────────────────────────────────────\n")
    return metrics


# ── Single image ──────────────────────────────────────────────────────────────

def run_single(processor, model, image_path: str,
               visualize: bool = False, save_crops: str = None, crop_padding: int = 10):
    pred_boxes, scores, (H, W) = predict(processor, model, image_path)

    print(f"\nDetections ({len(pred_boxes)}) in '{image_path}':")
    for i, (box, score) in enumerate(zip(pred_boxes, scores)):
        x1, y1, x2, y2 = box
        print(f"  [{i+1}] x1={x1:.0f} y1={y1:.0f} x2={x2:.0f} y2={y2:.0f}  score={score:.4f}")

    if save_crops and len(pred_boxes) > 0:
        image_np = np.array(Image.open(image_path).convert("RGB"))
        out_dir = Path(save_crops)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = map(int, box)
            x1c = max(0, x1 - crop_padding)
            y1c = max(0, y1 - crop_padding)
            x2c = min(W, x2 + crop_padding)
            y2c = min(H, y2 + crop_padding)
            Image.fromarray(image_np[y1c:y2c, x1c:x2c]).save(
                out_dir / f"{stem}_{i+1:03d}.jpg"
            )
        print(f"Saved {len(pred_boxes)} crop(s) to '{save_crops}/'")

    if visualize:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        image_np = np.array(Image.open(image_path).convert("RGB"))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image_np)
        colors = plt.cm.tab10.colors
        for i, (box, score) in enumerate(zip(pred_boxes, scores)):
            x1, y1, x2, y2 = box
            color = colors[i % len(colors)]
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            ))
            ax.text(x1, y1 - 6, f"#{i+1} {score:.2f}", color="white",
                    fontsize=9, fontweight="bold",
                    bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"))
        title = f"SAM 3: {len(pred_boxes)} insect(s) detected" if len(pred_boxes) else "SAM 3: No insects detected"
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    return pred_boxes, scores


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Insect detection with SAM 3 text prompts")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", choices=["train", "val"])
    group.add_argument("--image")

    p.add_argument("--dataset",         default=DATASET_ROOT)
    p.add_argument("--prompt",          default=TEXT_PROMPT,
                   help=f"Text concept to detect (default: '{TEXT_PROMPT}')")
    p.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD,
                   help="Confidence threshold (default: 0.50)")
    p.add_argument("--save-crops",      default=None, metavar="DIR")
    p.add_argument("--crop-padding",    type=int, default=10)
    p.add_argument("--output",          default=None, help="Save eval metrics to JSON")
    p.add_argument("--visualize",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    global DATASET_ROOT, TEXT_PROMPT, SCORE_THRESHOLD
    DATASET_ROOT    = args.dataset
    TEXT_PROMPT     = args.prompt
    SCORE_THRESHOLD = args.score_threshold

    processor, model = load_model()

    if args.split:
        metrics = run_evaluation(processor, model, split=args.split)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to '{args.output}'")
    else:
        run_single(processor, model, args.image,
                   visualize=args.visualize,
                   save_crops=args.save_crops,
                   crop_padding=args.crop_padding)


if __name__ == "__main__":
    main()