#!/usr/bin/env python3
"""
Compare a fine-tuned CLIPSeg checkpoint against the original base CLIPSeg model.

This script is for post-training inspection. It:
- loads the saved `best_model/` checkpoint from a training run
- loads the original pretrained CLIPSeg model
- rebuilds the same dataset split settings from `run_config.json`
- runs both models on the same samples
- compares each prediction against ground truth
- saves a single side-by-side comparison image per sample if requested

Each saved comparison image contains:
- original image
- ground-truth overlay
- base model prediction overlay
- fine-tuned model prediction overlay

Example:
  python3 scripts/compare_clipseg_models.py \
    --run-dir runs/clipseg_smoke_test \
    --split test \
    --limit 20 \
    --save-dir outputs/clipseg_compare_test
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "cracks": {
        "root": "cracks.coco/train",
        "annotations": "cracks.coco/train/_annotations.coco.json",
        "canonical_prompt": "segment crack",
    },
    "drywall": {
        "root": "Drywall-Join-Detect.coco/train",
        "annotations": "Drywall-Join-Detect.coco/train/_annotations.coco.json",
        "canonical_prompt": "segment taping area",
    },
}


@dataclass
class SampleRecord:
    image_path: Path
    width: int
    height: int
    mask: np.ndarray
    dataset_name: str
    canonical_prompt: str
    image_id: int
    file_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training output directory that contains run_config.json and best_model/.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which reconstructed split to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap the number of samples per dataset from the selected split.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Mask threshold; defaults to eval_threshold from run_config.json when present.",
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
        help="If set, save one comparison PNG per evaluated image.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def build_merged_mask(image_record: dict, anns: List[dict]) -> np.ndarray:
    size = (image_record["width"], image_record["height"])
    merged = np.zeros((size[1], size[0]), dtype=np.uint8)
    for ann in anns:
        if ann.get("segmentation"):
            ann_mask = draw_polygon_mask(size, ann["segmentation"])
        else:
            ann_mask = draw_bbox_mask(size, ann["bbox"])
        merged = np.maximum(merged, ann_mask)
    return merged


def build_records(limit_per_dataset: int | None) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for dataset_name, cfg in DATASET_CONFIGS.items():
        root = Path(str(cfg["root"]))
        annotations_path = Path(str(cfg["annotations"]))
        coco = load_coco(annotations_path)
        ann_index = build_annotation_index(coco["annotations"])
        image_records = coco["images"]
        if limit_per_dataset is not None:
            image_records = image_records[:limit_per_dataset]

        for image_record in image_records:
            records.append(
                SampleRecord(
                    image_path=root / image_record["file_name"],
                    width=int(image_record["width"]),
                    height=int(image_record["height"]),
                    mask=build_merged_mask(image_record, ann_index.get(image_record["id"], [])),
                    dataset_name=dataset_name,
                    canonical_prompt=str(cfg["canonical_prompt"]),
                    image_id=int(image_record["id"]),
                    file_name=str(image_record["file_name"]),
                )
            )
    return records


def split_records(
    records: Sequence[SampleRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.dataset_name, []).append(record)

    train_records: List[SampleRecord] = []
    val_records: List[SampleRecord] = []
    test_records: List[SampleRecord] = []

    for dataset_name, group in grouped.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        if total >= 3:
            train_end = max(1, min(train_end, total - 2))
            val_end = max(train_end + 1, min(val_end, total - 1))

        train_records.extend(shuffled[:train_end])
        val_records.extend(shuffled[train_end:val_end])
        test_records.extend(shuffled[val_end:])

        print(
            f"split[{dataset_name}] train={train_end} val={val_end - train_end} test={total - val_end}"
        )

    return train_records, val_records, test_records


def normalize_logits_shape(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 4:
        return logits
    if logits.ndim == 3:
        return logits.unsqueeze(1)
    if logits.ndim == 2:
        if logits.shape[0] == logits.shape[1]:
            return logits.unsqueeze(0).unsqueeze(0)
        side = int(round(logits.shape[1] ** 0.5))
        if side * side == logits.shape[1]:
            return logits.view(logits.shape[0], 1, side, side)
        raise ValueError(f"Unsupported 2D logits shape: {tuple(logits.shape)}")
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def predict_mask(
    model: CLIPSegForImageSegmentation,
    processor: CLIPSegProcessor,
    image: Image.Image,
    prompt: str,
    width: int,
    height: int,
    threshold: float,
    device: str,
) -> np.ndarray:
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = normalize_logits_shape(outputs.logits)
    logits = torch.nn.functional.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return (probs >= threshold).astype(np.uint8) * 255


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
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()
    mask_bool = mask > 0

    width, height = base.size
    for y in range(height):
        for x in range(width):
            if mask_bool[y, x]:
                overlay_pixels[x, y] = (color[0], color[1], color[2], alpha)

    return Image.alpha_composite(base, overlay).convert("RGB")


def save_comparison(
    image: Image.Image,
    gt_mask: np.ndarray,
    base_mask: np.ndarray,
    ft_mask: np.ndarray,
    base_iou: float,
    base_dice: float,
    ft_iou: float,
    ft_dice: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    original = image.convert("RGB")
    gt_panel = create_overlay(original, gt_mask, color=(0, 200, 0))
    base_panel = create_overlay(original, base_mask, color=(255, 140, 0))
    ft_panel = create_overlay(original, ft_mask, color=(220, 30, 30))

    width, height = original.size
    header_height = 64
    canvas = Image.new("RGB", (width * 4, height + header_height), color=(255, 255, 255))
    canvas.paste(original, (0, header_height))
    canvas.paste(gt_panel, (width, header_height))
    canvas.paste(base_panel, (width * 2, header_height))
    canvas.paste(ft_panel, (width * 3, header_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Original", fill=(0, 0, 0))
    draw.text((width + 10, 8), "Ground Truth", fill=(0, 0, 0))
    draw.text((width * 2 + 10, 8), "Base CLIPSeg", fill=(0, 0, 0))
    draw.text((width * 3 + 10, 8), "Fine-Tuned CLIPSeg", fill=(0, 0, 0))
    draw.text((width * 2 + 10, 28), f"IoU={base_iou:.4f} Dice={base_dice:.4f}", fill=(0, 0, 0))
    draw.text((width * 3 + 10, 28), f"IoU={ft_iou:.4f} Dice={ft_dice:.4f}", fill=(0, 0, 0))

    canvas.save(path)


def summarize(scores: List[float]) -> float:
    return float(sum(scores) / len(scores)) if scores else 0.0


def apply_per_dataset_limit(records: Sequence[SampleRecord], limit: int | None) -> List[SampleRecord]:
    if limit is None:
        return list(records)

    grouped: Dict[str, List[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.dataset_name, []).append(record)

    limited: List[SampleRecord] = []
    for dataset_name in sorted(grouped):
        limited.extend(grouped[dataset_name][:limit])
    return limited


def main() -> None:
    args = parse_args()
    run_config_path = args.run_dir / "run_config.json"
    best_model_dir = args.run_dir / "best_model"

    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    if not best_model_dir.exists():
        raise FileNotFoundError(f"Missing fine-tuned checkpoint: {best_model_dir}")

    run_config = load_json(run_config_path)
    model_id = str(run_config.get("model_id", "CIDAS/clipseg-rd64-refined"))
    train_ratio = float(run_config.get("train_ratio", 0.7))
    val_ratio = float(run_config.get("val_ratio", 0.15))
    seed = int(run_config.get("seed", 7))
    limit_per_dataset = run_config.get("limit_per_dataset")
    eval_threshold = float(
        args.threshold if args.threshold is not None else run_config.get("eval_threshold", 0.5)
    )

    records = build_records(limit_per_dataset=limit_per_dataset)
    train_records, val_records, test_records = split_records(
        records=records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    split_map = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    selected_records = apply_per_dataset_limit(split_map[args.split], args.limit)

    processor = CLIPSegProcessor.from_pretrained(model_id)
    base_model = CLIPSegForImageSegmentation.from_pretrained(model_id)
    fine_tuned_model = CLIPSegForImageSegmentation.from_pretrained(best_model_dir)
    base_model.to(args.device)
    fine_tuned_model.to(args.device)
    base_model.eval()
    fine_tuned_model.eval()

    base_ious: List[float] = []
    base_dices: List[float] = []
    ft_ious: List[float] = []
    ft_dices: List[float] = []
    delta_ious: List[float] = []
    delta_dices: List[float] = []

    print(f"split={args.split}")
    print(f"images_evaluated={len(selected_records)}")
    print(f"threshold={eval_threshold:.4f}")
    print()
    print("sample_results")

    for record in selected_records:
        image = Image.open(record.image_path).convert("RGB")
        gt_mask = record.mask

        base_mask = predict_mask(
            model=base_model,
            processor=processor,
            image=image,
            prompt=record.canonical_prompt,
            width=record.width,
            height=record.height,
            threshold=eval_threshold,
            device=args.device,
        )
        ft_mask = predict_mask(
            model=fine_tuned_model,
            processor=processor,
            image=image,
            prompt=record.canonical_prompt,
            width=record.width,
            height=record.height,
            threshold=eval_threshold,
            device=args.device,
        )

        base_iou, base_dice = compute_iou_and_dice(base_mask, gt_mask)
        ft_iou, ft_dice = compute_iou_and_dice(ft_mask, gt_mask)

        base_ious.append(base_iou)
        base_dices.append(base_dice)
        ft_ious.append(ft_iou)
        ft_dices.append(ft_dice)
        delta_ious.append(ft_iou - base_iou)
        delta_dices.append(ft_dice - base_dice)

        print(
            f"{record.dataset_name}\t{record.file_name}\t"
            f"base_iou={base_iou:.4f}\tft_iou={ft_iou:.4f}\t"
            f"base_dice={base_dice:.4f}\tft_dice={ft_dice:.4f}"
        )

        if args.save_dir:
            prompt_slug = record.canonical_prompt.replace(" ", "_")
            file_name = f"{record.dataset_name}__{record.image_id}__{prompt_slug}__comparison.png"
            save_comparison(
                image=image,
                gt_mask=gt_mask,
                base_mask=base_mask,
                ft_mask=ft_mask,
                base_iou=base_iou,
                base_dice=base_dice,
                ft_iou=ft_iou,
                ft_dice=ft_dice,
                path=args.save_dir / file_name,
            )

    print()
    print("aggregate_metrics")
    print(f"base_mean_iou={summarize(base_ious):.4f}")
    print(f"base_mean_dice={summarize(base_dices):.4f}")
    print(f"fine_tuned_mean_iou={summarize(ft_ious):.4f}")
    print(f"fine_tuned_mean_dice={summarize(ft_dices):.4f}")
    print(f"mean_iou_gain={summarize(delta_ious):.4f}")
    print(f"mean_dice_gain={summarize(delta_dices):.4f}")


if __name__ == "__main__":
    main()
