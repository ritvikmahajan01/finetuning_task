#!/usr/bin/env python3
"""
Grounding DINO + SAM zero-shot baseline.

This script:
- runs Grounding DINO with a text prompt to predict boxes
- feeds those boxes into SAM to produce masks
- merges all predicted masks per image
- computes IoU and Dice against the provided dataset masks
- optionally saves a single comparison visualization per image

Example:
  python3 scripts/grounded_sam_baseline.py \
    --dataset cracks \
    --limit 10 \
    --save-dir outputs/cracks_grounded_sam
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import (
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
    SamModel,
    SamProcessor,
)


DATASETS: Dict[str, Dict[str, str]] = {
    "cracks": {
        "root": "cracks.coco/train",
        "annotations": "cracks.coco/train/_annotations.coco.json",
        "prompt": "crack",
    },
    "drywall": {
        "root": "Drywall-Join-Detect.coco/train",
        "annotations": "Drywall-Join-Detect.coco/train/_annotations.coco.json",
        "prompt": "taping area",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model-id", default="facebook/sam-vit-base")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.0,
        help="Threshold for SAM mask logits after post-processing.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save one comparison PNG per image.",
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
    draw.text((10, 28), f"IoU={iou:.4f}  Dice={dice:.4f}", fill=(0, 0, 0))
    canvas.save(path)


def prompt_for_grounding(text: str) -> str:
    text = text.strip().lower()
    return text if text.endswith(".") else f"{text}."


def normalize_sam_masks(masks: object) -> List[np.ndarray]:
    """
    Normalize processor.post_process_masks output into a list of HxW arrays.
    """
    if isinstance(masks, list):
        if not masks:
            return []
        first = masks[0]
        if torch.is_tensor(first):
            return normalize_sam_masks(first)
        if isinstance(first, np.ndarray):
            return normalize_sam_masks(np.asarray(first))

    if torch.is_tensor(masks):
        arr = masks.detach().cpu().numpy()
    else:
        arr = np.asarray(masks)

    if arr.ndim == 4:
        # e.g. (num_boxes, 1, H, W)
        return [arr[i, 0] for i in range(arr.shape[0])]
    if arr.ndim == 3:
        # e.g. (num_boxes, H, W)
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 2:
        return [arr]

    raise ValueError(f"Unsupported SAM mask shape: {arr.shape}")


def merge_masks(mask_list: List[np.ndarray], threshold: float, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    merged = np.zeros((height, width), dtype=np.uint8)
    for mask in mask_list:
        merged = np.maximum(merged, (mask > threshold).astype(np.uint8) * 255)
    return merged


def main() -> None:
    args = parse_args()
    dataset_cfg = DATASETS[args.dataset]
    dataset_root = Path(dataset_cfg["root"])
    annotations_path = Path(dataset_cfg["annotations"])
    prompt = dataset_cfg["prompt"]
    grounding_text = prompt_for_grounding(prompt)

    coco = load_coco(annotations_path)
    image_records = coco["images"][: args.limit]
    ann_index = build_annotation_index(coco["annotations"])

    grounding_processor = GroundingDinoProcessor.from_pretrained(args.grounding_model_id)
    grounding_model = GroundingDinoForObjectDetection.from_pretrained(args.grounding_model_id)
    sam_processor = SamProcessor.from_pretrained(args.sam_model_id)
    sam_model = SamModel.from_pretrained(args.sam_model_id)

    grounding_model.to(args.device)
    sam_model.to(args.device)
    grounding_model.eval()
    sam_model.eval()

    per_image_scores: List[Tuple[str, float, float, int]] = []

    with torch.inference_mode():
        for image_record in image_records:
            image_path = dataset_root / image_record["file_name"]
            image = Image.open(image_path).convert("RGB")
            gt_mask = build_gt_mask(image_record, ann_index.get(image_record["id"], []))
            image_size = (image_record["width"], image_record["height"])

            gd_inputs = grounding_processor(images=image, text=grounding_text, return_tensors="pt")
            gd_inputs = {key: value.to(args.device) for key, value in gd_inputs.items()}
            gd_outputs = grounding_model(**gd_inputs)

            detections = grounding_processor.post_process_grounded_object_detection(
                gd_outputs,
                input_ids=gd_inputs.get("input_ids"),
                threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[(image_record["height"], image_record["width"])],
            )[0]

            boxes = detections["boxes"].detach().cpu()
            if boxes.numel() == 0:
                pred_mask = np.zeros((image_record["height"], image_record["width"]), dtype=np.uint8)
                num_boxes = 0
            else:
                sam_inputs = sam_processor(
                    images=image,
                    input_boxes=[boxes.tolist()],
                    return_tensors="pt",
                )
                sam_inputs = {key: value.to(args.device) for key, value in sam_inputs.items()}

                sam_outputs = sam_model(**sam_inputs, multimask_output=False)
                processed_masks = sam_processor.post_process_masks(
                    sam_outputs.pred_masks.detach().cpu(),
                    sam_inputs["original_sizes"].detach().cpu(),
                    sam_inputs["reshaped_input_sizes"].detach().cpu(),
                )
                mask_list = normalize_sam_masks(processed_masks)
                pred_mask = merge_masks(mask_list, threshold=args.mask_threshold, size=image_size)
                num_boxes = int(boxes.shape[0])

            iou, dice = compute_iou_and_dice(pred_mask, gt_mask)
            per_image_scores.append((image_record["file_name"], iou, dice, num_boxes))

            if args.save_dir:
                prompt_slug = prompt.replace(" ", "_")
                file_name = f"{image_record['id']}__{prompt_slug}__comparison.png"
                save_visualization(
                    image=image,
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    iou=iou,
                    dice=dice,
                    path=args.save_dir / file_name,
                )

    mean_iou = sum(score[1] for score in per_image_scores) / len(per_image_scores)
    mean_dice = sum(score[2] for score in per_image_scores) / len(per_image_scores)
    mean_boxes = sum(score[3] for score in per_image_scores) / len(per_image_scores)

    print(f"dataset={args.dataset}")
    print(f"prompt={prompt}")
    print(f"grounding_text={grounding_text}")
    print(f"grounding_model_id={args.grounding_model_id}")
    print(f"sam_model_id={args.sam_model_id}")
    print(f"images_evaluated={len(per_image_scores)}")
    print(f"mean_iou={mean_iou:.4f}")
    print(f"mean_dice={mean_dice:.4f}")
    print(f"mean_boxes={mean_boxes:.2f}")
    print()
    print("sample_results")
    for file_name, iou, dice, num_boxes in per_image_scores[:10]:
        print(f"{file_name}\tboxes={num_boxes}\tiou={iou:.4f}\tdice={dice:.4f}")

    if args.dataset == "drywall":
        print()
        print(
            "note=ground truth for drywall is derived from bounding boxes because the "
            "provided COCO export has empty segmentation arrays."
        )


if __name__ == "__main__":
    main()
