#!/usr/bin/env python3
"""
Simple zero-shot baseline for the Prompted Segmentation assignment.

This script:
- loads one of the provided COCO exports
- builds ground-truth masks from polygons (or boxes when segmentation is missing)
- runs zero-shot CLIPSeg with a text prompt
- computes IoU and Dice
- optionally saves predicted masks

Example:
  python3 scripts/zero_shot_clipseg_baseline.py \
    --dataset cracks \
    --limit 25 \
    --save-dir outputs/cracks_clipseg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


DATASETS: Dict[str, Dict[str, str]] = {
    "cracks": {
        "root": "cracks.coco/train",
        "annotations": "cracks.coco/train/_annotations.coco.json",
        "prompt": "segment crack",
    },
    "drywall": {
        "root": "Drywall-Join-Detect.coco/train",
        "annotations": "Drywall-Join-Detect.coco/train/_annotations.coco.json",
        "prompt": "segment taping area",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        required=True,
        help="Which dataset to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of images to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for predicted masks.",
    )
    parser.add_argument(
        "--model-id",
        default="CIDAS/clipseg-rd64-refined",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save comparison visualizations as PNGs to this directory.",
    )
    return parser.parse_args()


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_annotation_index(annotations: Iterable[dict]) -> Dict[int, List[dict]]:
    index: Dict[int, List[dict]] = {}
    for ann in annotations:
        index.setdefault(ann["image_id"], []).append(ann)
    return index


def draw_polygon_mask(size: Tuple[int, int], polygons: List[List[float]]) -> np.ndarray:
    width, height = size
    mask = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) < 6:
            continue
        xy = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        drawer.polygon(xy, fill=255)
    return np.array(mask, dtype=np.uint8)


def draw_bbox_mask(size: Tuple[int, int], bbox: List[float]) -> np.ndarray:
    width, height = size
    x, y, w, h = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


def build_gt_mask(image_record: dict, anns: List[dict]) -> np.ndarray:
    size = (image_record["width"], image_record["height"])
    merged = np.zeros((size[1], size[0]), dtype=np.uint8)
    for ann in anns:
        if ann.get("segmentation"):
            ann_mask = draw_polygon_mask(size, ann["segmentation"])
        else:
            ann_mask = draw_bbox_mask(size, ann["bbox"])
        merged = np.maximum(merged, ann_mask)
    return merged


def compute_iou_and_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    iou = float(intersection / union) if union else 1.0
    dice = float((2 * intersection) / (pred_sum + gt_sum)) if (pred_sum + gt_sum) else 1.0
    return iou, dice


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def create_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: int = 110,
) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    mask_bool = mask > 0
    alpha_f = alpha / 255.0
    for c in range(3):
        base[..., c] = np.where(
            mask_bool,
            base[..., c] * (1.0 - alpha_f) + color[c] * alpha_f,
            base[..., c],
        )
    return Image.fromarray(base.astype(np.uint8))


def save_visualization(
    image: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    iou: float,
    dice: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    original = image.convert("RGB")
    gt_panel = create_overlay(original, gt_mask, color=(0, 200, 0))
    pred_panel = create_overlay(original, pred_mask, color=(220, 30, 30))

    width, height = original.size
    header_height = 48
    canvas = Image.new("RGB", (width * 3, height + header_height), color=(255, 255, 255))
    canvas.paste(original, (0, header_height))
    canvas.paste(gt_panel, (width, header_height))
    canvas.paste(pred_panel, (width * 2, header_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Original", fill=(0, 0, 0))
    draw.text((width + 10, 8), "Ground Truth", fill=(0, 0, 0))
    draw.text((width * 2 + 10, 8), "Prediction", fill=(0, 0, 0))
    draw.text(
        (10, 28),
        f"IoU={iou:.4f}  Dice={dice:.4f}",
        fill=(0, 0, 0),
    )

    canvas.save(path)


def normalize_logits_shape(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CLIPSeg logits into (N, 1, H, W) so they can be resized safely.

    Depending on the transformers version/model variant, logits may be returned as:
    - (N, H, W)
    - (N, HW)
    - (N, 1, H, W)
    """
    if logits.ndim == 4:
        return logits

    if logits.ndim == 3:
        return logits.unsqueeze(1)

    if logits.ndim == 2:
        # CLIPSeg's decoder applies `.squeeze()`, so a single-image batch often
        # comes back as (H, W) instead of (1, H, W).
        if logits.shape[0] == logits.shape[1]:
            return logits.unsqueeze(0).unsqueeze(0)

        # Fallback for flat maps shaped like (N, H*W).
        side = int(round(logits.shape[1] ** 0.5))
        if side * side == logits.shape[1]:
            return logits.view(logits.shape[0], 1, side, side)

        raise ValueError(
            f"Unsupported 2D logits shape {tuple(logits.shape)}; cannot infer segmentation map."
        )

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def main() -> None:
    args = parse_args()
    dataset_cfg = DATASETS[args.dataset]
    dataset_root = Path(dataset_cfg["root"])
    annotations_path = Path(dataset_cfg["annotations"])
    prompt = dataset_cfg["prompt"]

    coco = load_coco(annotations_path)
    image_records = coco["images"][: args.limit]
    ann_index = build_annotation_index(coco["annotations"])

    processor = CLIPSegProcessor.from_pretrained(args.model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model_id)
    model.to(args.device)
    model.eval()

    per_image_scores: List[Tuple[str, float, float]] = []

    with torch.inference_mode():
        for image_record in image_records:
            image_path = dataset_root / image_record["file_name"]
            image = Image.open(image_path).convert("RGB")
            gt_mask = build_gt_mask(image_record, ann_index.get(image_record["id"], []))

            inputs = processor(
                text=[prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(args.device) for key, value in inputs.items()}

            outputs = model(**inputs)
            logits = normalize_logits_shape(outputs.logits)
            logits = torch.nn.functional.interpolate(
                logits,
                size=(image_record["height"], image_record["width"]),
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            pred_mask = (probs >= args.threshold).astype(np.uint8) * 255

            iou, dice = compute_iou_and_dice(pred_mask, gt_mask)
            per_image_scores.append((image_record["file_name"], iou, dice))

            if args.save_dir:
                prompt_slug = prompt.replace(" ", "_")
                comparison_name = f"{image_record['id']}__{prompt_slug}__comparison.png"
                save_visualization(
                    image=image,
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    iou=iou,
                    dice=dice,
                    path=args.save_dir / comparison_name,
                )

    mean_iou = sum(score[1] for score in per_image_scores) / len(per_image_scores)
    mean_dice = sum(score[2] for score in per_image_scores) / len(per_image_scores)

    print(f"dataset={args.dataset}")
    print(f"prompt={prompt}")
    print(f"model_id={args.model_id}")
    print(f"images_evaluated={len(per_image_scores)}")
    print(f"mean_iou={mean_iou:.4f}")
    print(f"mean_dice={mean_dice:.4f}")
    print()
    print("sample_results")
    for file_name, iou, dice in per_image_scores[:10]:
        print(f"{file_name}\tiou={iou:.4f}\tdice={dice:.4f}")

    if args.dataset == "drywall":
        print()
        print(
            "note=ground truth for drywall is derived from bounding boxes because the "
            "provided COCO export has empty segmentation arrays."
        )


if __name__ == "__main__":
    main()
