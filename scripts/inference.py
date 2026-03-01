#!/usr/bin/env python3
"""
Run inference with a fine-tuned CLIPSeg model and save prediction masks.

Output masks follow the assignment spec:
- PNG, single-channel, same spatial size as source image, values {0, 255}
- Filename: <image_id>__<prompt_slug>.png  (e.g. 123__segment_crack.png)

Also records per-image inference time and computes metrics against ground truth.

Example:
  python3 scripts/inference.py \
    --run-dir runs/clipseg_v2 \
    --output-dir outputs/predictions \
    --split test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
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
    annotations: List[dict]
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
        help="Training output directory containing best_model/ and run_config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save prediction mask PNGs.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which split to run inference on.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of images per dataset from the selected split.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
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


def build_merged_mask(size: Tuple[int, int], anns: List[dict]) -> np.ndarray:
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
        coco = load_json(annotations_path)
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
                    annotations=ann_index.get(image_record["id"], []),
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

    return train_records, val_records, test_records


def apply_per_dataset_limit(
    records: Sequence[SampleRecord], limit: int | None
) -> List[SampleRecord]:
    if limit is None:
        return list(records)
    grouped: Dict[str, List[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.dataset_name, []).append(record)
    limited: List[SampleRecord] = []
    for dataset_name in sorted(grouped):
        limited.extend(grouped[dataset_name][:limit])
    return limited


def normalize_logits_shape(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 4:
        return logits
    if logits.ndim == 3:
        return logits.unsqueeze(1)
    if logits.ndim == 2:
        if logits.shape[0] == logits.shape[1]:
            return logits.unsqueeze(0).unsqueeze(0)
        side = int(round(math.sqrt(logits.shape[1])))
        if side * side == logits.shape[1]:
            return logits.view(logits.shape[0], 1, side, side)
        raise ValueError(f"Unsupported 2D logits shape: {tuple(logits.shape)}")
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def compute_iou_and_dice(
    pred_mask: np.ndarray, gt_mask: np.ndarray
) -> Tuple[float, float]:
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    iou = float(intersection / union) if union else 1.0
    dice = float((2 * intersection) / (pred_sum + gt_sum)) if (pred_sum + gt_sum) else 1.0
    return iou, dice


def main() -> None:
    args = parse_args()
    run_config_path = args.run_dir / "run_config.json"
    best_model_dir = args.run_dir / "best_model"
    best_thresholds_path = args.run_dir / "best_thresholds.json"

    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    if not best_model_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_model_dir}")

    run_config = load_json(run_config_path)
    model_id = str(run_config.get("model_id", "CIDAS/clipseg-rd64-refined"))
    train_ratio = float(run_config.get("train_ratio", 0.7))
    val_ratio = float(run_config.get("val_ratio", 0.15))
    seed = int(run_config.get("seed", 7))
    limit_per_dataset = run_config.get("limit_per_dataset")

    # Load per-task optimal thresholds (fall back to 0.5).
    per_task_thresholds: Dict[str, float] = {}
    if best_thresholds_path.exists():
        raw = load_json(best_thresholds_path)
        for task_name, info in raw.items():
            per_task_thresholds[task_name] = float(info["threshold"])
    print(f"thresholds: {per_task_thresholds}")

    # Reconstruct splits.
    records = build_records(limit_per_dataset=limit_per_dataset)
    train_records, val_records, test_records = split_records(
        records=records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    split_map = {"train": train_records, "val": val_records, "test": test_records}
    selected_records = apply_per_dataset_limit(split_map[args.split], args.limit)

    # Load model.
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(best_model_dir)
    model.to(args.device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_task_ious: Dict[str, List[float]] = {}
    per_task_dices: Dict[str, List[float]] = {}
    inference_times: List[float] = []

    print(f"split={args.split}")
    print(f"images={len(selected_records)}")
    print()

    with torch.inference_mode():
        for record in selected_records:
            image = Image.open(record.image_path).convert("RGB")
            prompt = record.canonical_prompt
            threshold = per_task_thresholds.get(record.dataset_name, 0.5)

            # Timed inference.
            t0 = time.perf_counter()

            inputs = processor(
                text=[prompt], images=[image], padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            outputs = model(**inputs)

            logits = normalize_logits_shape(outputs.logits)
            logits = torch.nn.functional.interpolate(
                logits,
                size=(record.height, record.width),
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_mask = (probs >= threshold).astype(np.uint8) * 255

            t1 = time.perf_counter()
            inference_times.append(t1 - t0)

            # Save mask: <image_id>__<prompt_slug>.png
            prompt_slug = prompt.replace(" ", "_")
            mask_filename = f"{record.image_id}__{prompt_slug}.png"
            mask_path = args.output_dir / mask_filename
            Image.fromarray(pred_mask, mode="L").save(mask_path)

            # Compute metrics against GT.
            gt_mask = build_merged_mask(
                (record.width, record.height), record.annotations
            )
            iou, dice = compute_iou_and_dice(pred_mask, gt_mask)
            per_task_ious.setdefault(record.dataset_name, []).append(iou)
            per_task_dices.setdefault(record.dataset_name, []).append(dice)

            print(
                f"  {mask_filename}  iou={iou:.4f}  dice={dice:.4f}  "
                f"time={inference_times[-1]*1000:.1f}ms"
            )

    # Summary.
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    print()
    print("summary")
    print(f"  total_images={len(inference_times)}")
    print(f"  avg_inference_time={avg_time*1000:.1f}ms")

    for dataset_name in sorted(per_task_ious):
        ious = per_task_ious[dataset_name]
        dices = per_task_dices[dataset_name]
        print(
            f"  {dataset_name}: "
            f"iou={np.mean(ious):.4f} (std={np.std(ious):.4f}, min={np.min(ious):.4f})  "
            f"dice={np.mean(dices):.4f} (std={np.std(dices):.4f}, min={np.min(dices):.4f})"
        )

    macro_iou = float(np.mean([np.mean(v) for v in per_task_ious.values()]))
    macro_dice = float(np.mean([np.mean(v) for v in per_task_dices.values()]))
    print(f"  macro_iou={macro_iou:.4f}")
    print(f"  macro_dice={macro_dice:.4f}")

    # Save results JSON.
    results = {
        "split": args.split,
        "total_images": len(inference_times),
        "avg_inference_time_ms": round(avg_time * 1000, 1),
        "macro_iou": round(macro_iou, 4),
        "macro_dice": round(macro_dice, 4),
        "thresholds": per_task_thresholds,
    }
    for dataset_name in sorted(per_task_ious):
        ious = per_task_ious[dataset_name]
        dices = per_task_dices[dataset_name]
        results[f"{dataset_name}_iou"] = round(float(np.mean(ious)), 4)
        results[f"{dataset_name}_iou_std"] = round(float(np.std(ious)), 4)
        results[f"{dataset_name}_dice"] = round(float(np.mean(dices)), 4)
        results[f"{dataset_name}_dice_std"] = round(float(np.std(dices)), 4)

    results_path = args.output_dir / "inference_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nsaved masks to {args.output_dir}/")
    print(f"saved results to {results_path}")


if __name__ == "__main__":
    main()
