import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

MODEL_NAME   = "facebook/sam3"

TEXT_PROMPT  = "insect"

DETECTION_INSET_PIXELS = 10

SCORE_THRESHOLD = 0.50
MASK_THRESHOLD  = 0.50

EVAL_IOU_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.4

CROP_PADDING_PIXELS = 20
VISUAL_BOX_PADDING_PIXELS = 20

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

def drop_overlapping_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = NMS_IOU_THRESHOLD) -> np.ndarray:
    """Return indices of boxes to keep after Non-Maximum Suppression."""
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # highest score first

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU of this box with all remaining boxes
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou < iou_threshold]  # drop overlapping boxes

    return np.array(keep, dtype=int)

def predict(processor, model, image_path: str):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

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
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), (h, w)

    boxes  = results["boxes"].cpu().numpy().astype(np.float32)
    scores = results["scores"].cpu().numpy().astype(np.float32)

    inset = DETECTION_INSET_PIXELS
    keep = (
        (boxes[:, 0] >= inset) &
        (boxes[:, 1] >= inset) &
        (boxes[:, 2] <= w - inset) &
        (boxes[:, 3] <= h - inset)
    )
    boxes  = boxes[keep]
    scores = scores[keep]

    keep = drop_overlapping_boxes(boxes, scores, iou_threshold=NMS_IOU_THRESHOLD)
    boxes  = boxes[keep]
    scores = scores[keep]

    return boxes, scores, (h, w)

def make_visualization(image_path: str, pred_boxes: np.ndarray, scores: np.ndarray,
                        save_path: str = None, show: bool = False):
    """Render bounding boxes on the image. Saves to save_path and/or displays it."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    image_np = np.array(Image.open(image_path).convert("RGB"))
    H, W = image_np.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_np)

    inset = DETECTION_INSET_PIXELS
    ax.add_patch(patches.Rectangle(
        (inset, inset), W - 2 * inset, H - 2 * inset,
        linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--"
    ))

    colors = plt.cm.tab10.colors
    for i, (box, score) in enumerate(zip(pred_boxes, scores)):
        x1, y1, x2, y2 = box
        x1 -= VISUAL_BOX_PADDING_PIXELS
        y1 -= VISUAL_BOX_PADDING_PIXELS
        x2 += VISUAL_BOX_PADDING_PIXELS
        y2 += VISUAL_BOX_PADDING_PIXELS
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


def _image_exif_datetime(image_path: Path) -> str:
    """Return EXIF DateTimeOriginal (or DateTimeDigitized / DateTime) as a string, or ''."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()  # returns None for non-JPEG or missing EXIF
        if exif:
            # 36867 = DateTimeOriginal, 36868 = DateTimeDigitized, 306 = DateTime
            for tag_id in (36867, 36868, 306):
                value = exif.get(tag_id)
                if value:
                    return str(value)
    except Exception:
        pass
    return ""


def run_folder(processor, model, image_folder: str, visualize_folder: str = None,
               save_crops: str = None, crop_padding: int = 10, csv_path: str = None):
    """Run detection on every image in image_folder, searching recursively.

    Args:
        image_folder: Path to the input folder containing images (searched recursively).
        visualize_folder: If given, annotated images are saved here as JPEGs.
        save_crops:   If given, individual insect crops are saved here.
        crop_padding: Pixel padding around each crop box.
        csv_path:     If given, a CSV with per-detection metadata is written here.
    """
    in_dir = Path(image_folder)
    image_paths = sorted(
        p for ext in IMAGE_EXTENSIONS
        for p in in_dir.rglob(f"*{ext}")
        if p.is_file()
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in '{image_folder}' (searched recursively). "
            f"Supported extensions: {IMAGE_EXTENSIONS}"
        )

    if visualize_folder:
        visualize_directory = Path(visualize_folder)
        visualize_directory.mkdir(parents=True, exist_ok=True)
        print(f"Visualizations will be saved to: {visualize_directory.resolve()}")

    if save_crops:
        crops_dir = Path(save_crops)
        crops_dir.mkdir(parents=True, exist_ok=True)
        print(f"Crops will be saved to:          {crops_dir.resolve()}")

    print(f"\nProcessing {len(image_paths)} image(s) from '{in_dir.resolve()}' (recursive)...\n")

    total_detections = 0
    csv_rows = []

    for img_path in tqdm(image_paths, desc="Detecting"):
        pred_boxes, scores, (h, w) = predict(processor, model, str(img_path))
        n = len(pred_boxes)
        total_detections += n

        rel_path = img_path.relative_to(in_dir)
        safe_prefix = "_".join(rel_path.parent.parts + (img_path.stem,))
        tqdm.write(f"  {rel_path}: {n} detection(s)")

        # Save visualization flat output folder
        vis_save_path = ""
        if visualize_folder:
            out_path = visualize_directory / (safe_prefix + "_visual.jpg")
            make_visualization(str(img_path), pred_boxes, scores,
                               save_path=str(out_path), show=False)
            vis_save_path = str(out_path)

        # Save crops flat output folder
        image_np = None
        if save_crops and n > 0:
            image_np = np.array(Image.open(img_path).convert("RGB"))

        for i, (box, score) in enumerate(zip(pred_boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            x1c, y1c, x2c, y2c = 0, 0, 0, 0

            crop_save_path = ""
            if save_crops and image_np is not None:
                x1c = max(0, x1 - crop_padding)
                y1c = max(0, y1 - crop_padding)
                x2c = min(w, x2 + crop_padding)
                y2c = min(h, y2 + crop_padding)
                crop_name = f"{safe_prefix}_{i+1:03d}.jpg"
                crop_out = crops_dir / crop_name
                Image.fromarray(image_np[y1c:y2c, x1c:x2c]).save(crop_out)
                crop_save_path = str(crop_out)

            csv_rows.append({
                "image_path": str(rel_path),
                "detections_in_image": n,
                "detection_index":   i + 1,
                "x1":                x1c,
                "y1":                y1c,
                "x2":                x2c,
                "y2":                y2c,
                "confidence":        round(float(score), 6),
                "crop_path":         crop_save_path,
                "visualization_path": vis_save_path,
            })

        # Emit one no-detection row so every processed image appears in the CSV
        if n == 0:
            csv_rows.append({
                "image_path": str(rel_path),
                "detections_in_image": 0,
                "detection_index":   0,
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "confidence":        "",
                "crop_path":         "",
                "visualization_path": vis_save_path,
            })

    print(f"\nDone. Total detections across all images: {total_detections}")
    if visualize_folder:
        print(f"Annotated images saved to: '{visualize_folder}/'")
    if save_crops:
        print(f"Crops saved to:            '{save_crops}/'")

    if csv_path and csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV saved to:              '{csv_path}'")

    return csv_rows


def parse_args():
    p = argparse.ArgumentParser(description="Insect detection with SAM 3 text prompts")

    p.add_argument("--image-folder", metavar="DIR", required=True,
                   help="Run on all images inside a folder")

    # --- Output options ---
    p.add_argument("--vis-folder",      metavar="DIR", default=None,
                   help="Save annotated visualizations to this folder")
    p.add_argument("--save-crops",      default=None, metavar="DIR",
                   help="Save individual insect crops here")
    p.add_argument("--crop-padding", type=int, default=CROP_PADDING_PIXELS,
                   help="Pixel padding around crop boxes (default: 10)")
    p.add_argument("--csv",             default=None, metavar="FILE",
                   help="Save per-detection metadata to a CSV file")

    # --- Model / detection options ---
    p.add_argument("--prompt",          default=TEXT_PROMPT,
                   help=f"Text concept to detect (default: '{TEXT_PROMPT}')")
    p.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD,
                   help=f"Confidence threshold (default: {SCORE_THRESHOLD})")

    return p.parse_args()


def main():
    args = parse_args()

    global TEXT_PROMPT, SCORE_THRESHOLD
    TEXT_PROMPT     = args.prompt
    SCORE_THRESHOLD = args.score_threshold

    processor, model = load_model()

    run_folder(
        processor, model,
        image_folder=args.image_folder,
        visualize_folder=args.vis_folder,
        save_crops=args.save_crops,
        crop_padding=args.crop_padding,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()