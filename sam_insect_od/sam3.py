# Install:
#     pip install torch torchvision
#     pip install git+https://github.com/huggingface/transformers.git  # SAM3 needs latest transformers
#     pip install Pillow tqdm matplotlib
#
#     You need to accept the Meta license on HuggingFace before the model will download
#     Go to https://huggingface.co/facebook/sam3
#     Then log in once with "hf auth login"
#
# Usage:
#     # Single image with visualization
#     python insect_detection.py --image photo.jpg --visualize
#
#     # Process a folder of images and save visualizations
#     python insect_detection.py --image-folder ./photos --vis-folder ./results
#
#     # Folder with crops saved too
#     python insect_detection.py --image-folder ./photos --vis-folder ./results --save-crops ./crops
#
#     # Evaluate against your labeled val set
#     python insect_detection.py --split val
#
#     # Save crops for your downstream classifier
#     python insect_detection.py --image photo.jpg --save-crops ./crops

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


DATASET_ROOT = "./df6"
MODEL_NAME   = "facebook/sam3"

TEXT_PROMPT  = "insect"

SCORE_THRESHOLD = 0.50
MASK_THRESHOLD  = 0.50

EVAL_IOU_THRESHOLD = 0.50

CROP_PADDING = 20      # extra pixels added on each side of a detected bounding box when saving crops
VIS_BOX_PADDING = 20   # extra pixels added on each side of a bounding box drawn in visualizations

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    from transformers import Sam3Model, Sam3Processor
    print(f"Loading SAM 3 ({MODEL_NAME}) on {DEVICE}...")
    print("(First run will download about 3 GB of weights from HuggingFace)\n")
    processor = Sam3Processor.from_pretrained(MODEL_NAME)
    model = Sam3Model.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print("SAM 3 ready.\n")
    return processor, model


def predict(processor, model, image_path: str):
    # Returns:
    #  boxes  - (N, 4) float32 array of absolute (x1, y1, x2, y2) pixel coords
    #  scores - (N,)   float32 confidence scores
    #  (H, W) - original image size

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


def load_gt_boxes(label_path: str, H: int, W: int) -> np.ndarray:
    # Parse YOLO-format label file -> absolute (x1, y1, x2, y2).
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


# Evaluation
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

    print("\n== Evaluation Results ===============================")
    for k, v in metrics.items():
        print(f"  {k:<26}: {v}")
    print("=====================================================\n")
    return metrics


def make_visualization(image_path: str, pred_boxes: np.ndarray, scores: np.ndarray,
                        save_path: str = None, show: bool = False):
    """Render bounding boxes on the image. Saves to save_path and/or displays it."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")  # non-interactive backend when only saving
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    image_np = np.array(Image.open(image_path).convert("RGB"))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_np)
    colors = plt.cm.tab10.colors
    for i, (box, score) in enumerate(zip(pred_boxes, scores)):
        x1, y1, x2, y2 = box
        x1 -= VIS_BOX_PADDING
        y1 -= VIS_BOX_PADDING
        x2 += VIS_BOX_PADDING
        y2 += VIS_BOX_PADDING
        color = colors[i % len(colors)]
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        ))
        ax.text(x1, y1 - 6, f"#{i+1} {score:.2f}", color="white",
                fontsize=9, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"))
    title = (f"SAM 3: {len(pred_boxes)} insect(s) detected"
             if len(pred_boxes) else "SAM 3: No insects detected")
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def run_folder(processor, model, image_folder: str, vis_folder: str = None,
               save_crops: str = None, crop_padding: int = 10):
    """Run detection on every image in image_folder.

    Args:
        image_folder: Path to the input folder containing images.
        vis_folder:   If given, annotated images are saved here as JPEGs.
        save_crops:   If given, individual insect crops are saved here.
        crop_padding: Pixel padding around each crop box.
    """
    in_dir = Path(image_folder)
    image_paths = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in '{image_folder}'. "
            f"Supported extensions: {IMAGE_EXTENSIONS}"
        )

    # Prepare output directories
    if vis_folder:
        vis_dir = Path(vis_folder)
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"Visualizations will be saved to: {vis_dir.resolve()}")

    if save_crops:
        crops_dir = Path(save_crops)
        crops_dir.mkdir(parents=True, exist_ok=True)
        print(f"Crops will be saved to:          {crops_dir.resolve()}")

    print(f"\nProcessing {len(image_paths)} image(s) from '{in_dir.resolve()}'...\n")

    total_detections = 0
    summary_rows = []

    for img_path in tqdm(image_paths, desc="Detecting"):
        pred_boxes, scores, (H, W) = predict(processor, model, str(img_path))
        n = len(pred_boxes)
        total_detections += n

        tqdm.write(f"  {img_path.name}: {n} detection(s)")

        # Save visualization
        if vis_folder:
            out_path = vis_dir / (img_path.stem + "_vis.jpg")
            make_visualization(str(img_path), pred_boxes, scores,
                               save_path=str(out_path), show=False)

        # Save crops
        if save_crops and n > 0:
            image_np = np.array(Image.open(img_path).convert("RGB"))
            for i, box in enumerate(pred_boxes):
                x1, y1, x2, y2 = map(int, box)
                x1c = max(0, x1 - crop_padding)
                y1c = max(0, y1 - crop_padding)
                x2c = min(W, x2 + crop_padding)
                y2c = min(H, y2 + crop_padding)
                crop_name = f"{img_path.stem}_{i+1:03d}.jpg"
                Image.fromarray(image_np[y1c:y2c, x1c:x2c]).save(
                    crops_dir / crop_name
                )

        summary_rows.append({
            "image": img_path.name,
            "detections": n,
            "scores": [round(float(s), 4) for s in scores],
        })

    print(f"\nDone. Total detections across all images: {total_detections}")
    if vis_folder:
        print(f"Annotated images saved to: '{vis_folder}/'")
    if save_crops:
        print(f"Crops saved to:            '{save_crops}/'")

    return summary_rows


def run_single_image(processor, model, image_path: str,
                     visualize: bool = False, vis_folder: str = None,
                     save_crops: str = None, crop_padding: int = 10):
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

    # Save to vis_folder if provided (non-interactive)
    if vis_folder:
        vis_dir = Path(vis_folder)
        vis_dir.mkdir(parents=True, exist_ok=True)
        out_path = vis_dir / (Path(image_path).stem + "_vis.jpg")
        make_visualization(image_path, pred_boxes, scores,
                           save_path=str(out_path), show=False)
        print(f"Visualization saved to '{out_path}'")

    if visualize:
        make_visualization(image_path, pred_boxes, scores, show=True)

    return pred_boxes, scores


def parse_args():
    p = argparse.ArgumentParser(description="Insect detection with SAM 3 text prompts")

    # --- Input mode (mutually exclusive) ---
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--split",        choices=["train", "val"],
                             help="Evaluate a dataset split (train or val)")
    input_group.add_argument("--image",        metavar="FILE",
                             help="Run on a single image file")
    input_group.add_argument("--image-folder", metavar="DIR",
                             help="Run on all images inside a folder")

    # --- Output options ---
    p.add_argument("--vis-folder",      metavar="DIR", default=None,
                   help="Save annotated visualizations to this folder "
                        "(works with --image and --image-folder)")
    p.add_argument("--visualize",       action="store_true",
                   help="Display visualization interactively (single image only)")
    p.add_argument("--save-crops",      default=None, metavar="DIR",
                   help="Save individual insect crops here")
    p.add_argument("--crop-padding",    type=int, default=CROP_PADDING,
                   help="Pixel padding around crop boxes (default: 10)")
    p.add_argument("--output",          default=None, metavar="FILE",
                   help="Save eval metrics / folder summary to a JSON file")

    # --- Model / detection options ---
    p.add_argument("--dataset",         default=DATASET_ROOT,
                   help=f"Dataset root for --split mode (default: '{DATASET_ROOT}')")
    p.add_argument("--prompt",          default=TEXT_PROMPT,
                   help=f"Text concept to detect (default: '{TEXT_PROMPT}')")
    p.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD,
                   help=f"Confidence threshold (default: {SCORE_THRESHOLD})")

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

    elif args.image_folder:
        summary = run_folder(
            processor, model,
            image_folder=args.image_folder,
            vis_folder=args.vis_folder,
            save_crops=args.save_crops,
            crop_padding=args.crop_padding,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to '{args.output}'")

    else:  # --image
        run_single_image(
            processor, model, args.image,
            visualize=args.visualize,
            vis_folder=args.vis_folder,
            save_crops=args.save_crops,
            crop_padding=args.crop_padding,
        )


if __name__ == "__main__":
    main()