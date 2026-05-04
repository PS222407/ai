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

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

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


def run_folder(processor, model, image_folder: str, visualize_folder: str = None,
               save_crops: str = None, crop_padding: int = 10):
    """Run detection on every image in image_folder, searching recursively.

    Args:
        image_folder: Path to the input folder containing images (searched recursively).
        visualize_folder: If given, annotated images are saved here as JPEGs.
        save_crops:   If given, individual insect crops are saved here.
        crop_padding: Pixel padding around each crop box.
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
    summary_rows = []

    for img_path in tqdm(image_paths, desc="Detecting"):
        pred_boxes, scores, (H, W) = predict(processor, model, str(img_path))
        n = len(pred_boxes)
        total_detections += n

        rel_path = img_path.relative_to(in_dir)
        tqdm.write(f"  {rel_path}: {n} detection(s)")

        # Save visualization — flat output folder
        if visualize_folder:
            out_path = visualize_directory / (img_path.stem + "_visual.jpg")
            make_visualization(str(img_path), pred_boxes, scores,
                               save_path=str(out_path), show=False)

        # Save crops — flat output folder, prefix stem to avoid collisions
        if save_crops and n > 0:
            image_np = np.array(Image.open(img_path).convert("RGB"))
            safe_prefix = "_".join(rel_path.parent.parts + (img_path.stem,))
            for i, box in enumerate(pred_boxes):
                x1, y1, x2, y2 = map(int, box)
                x1c = max(0, x1 - crop_padding)
                y1c = max(0, y1 - crop_padding)
                x2c = min(W, x2 + crop_padding)
                y2c = min(H, y2 + crop_padding)
                crop_name = f"{safe_prefix}_{i+1:03d}.jpg"
                Image.fromarray(image_np[y1c:y2c, x1c:x2c]).save(
                    crops_dir / crop_name
                )

        summary_rows.append({
            "image": str(rel_path),
            "detections": n,
            "scores": [round(float(s), 4) for s in scores],
        })

    print(f"\nDone. Total detections across all images: {total_detections}")
    if visualize_folder:
        print(f"Annotated images saved to: '{visualize_folder}/'")
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
        out_path = vis_dir / (Path(image_path).stem + "_visual.jpg")
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

    if args.image_folder:
        summary = run_folder(
            processor, model,
            image_folder=args.image_folder,
            visualize_folder=args.vis_folder,
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