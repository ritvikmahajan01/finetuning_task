#!/usr/bin/env python3
"""
Fine-tune CLIPSeg on the combined crack + drywall datasets.

This script is intentionally simple and explicit:
- it reads the two COCO exports already present in this repository
- it merges annotations into one binary mask per image
- it trains one text-conditioned model across both tasks
- it uses prompt variants during training and canonical prompts for validation/test
- it treats drywall labels as weak supervision because that dataset only contains boxes

The default training strategy is a low-risk first pass:
- freeze the CLIP text encoder
- freeze most of the vision encoder
- train the CLIPSeg decoder
- optionally unfreeze the last N vision layers

Example:
  python3 scripts/finetune_clipseg.py \
    --output-dir runs/clipseg_combined \
    --epochs 4 \
    --batch-size 4 \
    --limit-per-dataset 200
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "cracks": {
        "root": "cracks.coco/train",
        "annotations": "cracks.coco/train/_annotations.coco.json",
        "canonical_prompt": "segment crack",
        "prompt_variants": [
            "segment crack",
            "segment wall crack",
            "segment structural crack",
            "segment surface crack",
        ],
        "task_weight": 1.0,
        "positive_weight": 3.0,
        "uses_boxes_as_masks": False,
    },
    "drywall": {
        "root": "Drywall-Join-Detect.coco/train",
        "annotations": "Drywall-Join-Detect.coco/train/_annotations.coco.json",
        "canonical_prompt": "segment taping area",
        "prompt_variants": [
            "segment taping area",
            "segment joint/tape",
            "segment drywall seam",
            "segment drywall joint",
        ],
        "task_weight": 0.6,
        "positive_weight": 1.5,
        "uses_boxes_as_masks": True,
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
    prompt_variants: List[str]
    task_weight: float
    positive_weight: float
    uses_boxes_as_masks: bool
    image_id: int


class PromptedSegmentationDataset(Dataset):
    """
    Dataset wrapper that keeps mask generation outside the training loop and
    chooses prompt variants only for the training split.
    """

    def __init__(self, records: Sequence[SampleRecord], split_name: str, use_prompt_variants: bool) -> None:
        self.records = list(records)
        self.split_name = split_name
        self.use_prompt_variants = use_prompt_variants

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        if self.use_prompt_variants:
            prompt = random.choice(record.prompt_variants)
        else:
            prompt = record.canonical_prompt

        return {
            "image": image,
            "mask": torch.from_numpy((record.mask > 0).astype(np.float32)),
            "prompt": prompt,
            "dataset_name": record.dataset_name,
            "task_weight": torch.tensor(record.task_weight, dtype=torch.float32),
            "positive_weight": torch.tensor(record.positive_weight, dtype=torch.float32),
            "uses_boxes_as_masks": record.uses_boxes_as_masks,
            "image_id": record.image_id,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="CIDAS/clipseg-rd64-refined")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="If set, cap the number of images loaded from each dataset for faster experiments.",
    )
    parser.add_argument(
        "--unfreeze-vision-layers",
        type=int,
        default=0,
        help="Unfreeze the last N CLIP vision encoder layers in addition to the decoder.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Set >0 only if your local Python/PIL setup is stable with worker processes.",
    )
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize sigmoid outputs during validation/test.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    all_records: List[SampleRecord] = []
    for dataset_name, cfg in DATASET_CONFIGS.items():
        root = Path(str(cfg["root"]))
        annotations_path = Path(str(cfg["annotations"]))
        coco = load_coco(annotations_path)
        ann_index = build_annotation_index(coco["annotations"])
        image_records = coco["images"]
        if limit_per_dataset is not None:
            image_records = image_records[:limit_per_dataset]

        for image_record in image_records:
            mask = build_merged_mask(image_record, ann_index.get(image_record["id"], []))
            all_records.append(
                SampleRecord(
                    image_path=root / image_record["file_name"],
                    width=image_record["width"],
                    height=image_record["height"],
                    mask=mask,
                    dataset_name=dataset_name,
                    canonical_prompt=str(cfg["canonical_prompt"]),
                    prompt_variants=list(cfg["prompt_variants"]),
                    task_weight=float(cfg["task_weight"]),
                    positive_weight=float(cfg["positive_weight"]),
                    uses_boxes_as_masks=bool(cfg["uses_boxes_as_masks"]),
                    image_id=int(image_record["id"]),
                )
            )
    return all_records


def split_records(
    records: Sequence[SampleRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    """
    Split each dataset separately so the smaller drywall set is not drowned out.
    """
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1) or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios. Require 0 < train_ratio, val_ratio and train+val < 1.")

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

        # Keep every split non-empty when possible.
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


def make_collate_fn(processor: CLIPSegProcessor):
    def collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        images = [item["image"] for item in batch]
        prompts = [str(item["prompt"]) for item in batch]
        masks = torch.stack([item["mask"] for item in batch], dim=0)
        task_weights = torch.stack([item["task_weight"] for item in batch], dim=0)
        positive_weights = torch.stack([item["positive_weight"] for item in batch], dim=0)
        dataset_names = [str(item["dataset_name"]) for item in batch]
        uses_boxes_as_masks = [bool(item["uses_boxes_as_masks"]) for item in batch]
        image_ids = [int(item["image_id"]) for item in batch]

        model_inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
        model_inputs["masks"] = masks
        model_inputs["task_weights"] = task_weights
        model_inputs["positive_weights"] = positive_weights
        model_inputs["dataset_names"] = dataset_names
        model_inputs["uses_boxes_as_masks"] = uses_boxes_as_masks
        model_inputs["prompts"] = prompts
        model_inputs["image_ids"] = image_ids
        return model_inputs

    return collate


def normalize_logits_shape(logits: torch.Tensor) -> torch.Tensor:
    """
    CLIPSeg returns different shapes across versions and batch sizes.
    This normalizes everything to (batch, 1, h, w).
    """
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


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weights: torch.Tensor,
    task_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Apply:
    - per-task positive pixel weighting (cracks need stronger positive emphasis)
    - per-task sample weighting (drywall labels are weaker, so they count less)
    """
    raw = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pixel_weights = 1.0 + (positive_weights[:, None, None, None] - 1.0) * targets
    per_sample = (raw * pixel_weights).mean(dim=(1, 2, 3))
    return (per_sample * task_weights).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, task_weights: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    per_sample = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0))
    return (per_sample * task_weights).mean()


def compute_batch_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> Tuple[List[float], List[float]]:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    gts = targets >= 0.5

    ious: List[float] = []
    dices: List[float] = []
    for idx in range(preds.shape[0]):
        pred = preds[idx]
        gt = gts[idx]
        intersection = torch.logical_and(pred, gt).sum().item()
        union = torch.logical_or(pred, gt).sum().item()
        pred_sum = pred.sum().item()
        gt_sum = gt.sum().item()
        iou = float(intersection / union) if union else 1.0
        dice = float((2.0 * intersection) / (pred_sum + gt_sum)) if (pred_sum + gt_sum) else 1.0
        ious.append(iou)
        dices.append(dice)
    return ious, dices


def freeze_model_for_first_pass(model: CLIPSegForImageSegmentation, unfreeze_vision_layers: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Train the decoder by default.
    for parameter in model.decoder.parameters():
        parameter.requires_grad = True

    if unfreeze_vision_layers > 0:
        vision_layers = model.clip.vision_model.encoder.layers
        for layer in vision_layers[-unfreeze_vision_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    # Keep projection layers trainable when any vision layers are unfrozen.
    if unfreeze_vision_layers > 0:
        for parameter in model.clip.visual_projection.parameters():
            parameter.requires_grad = True


def move_tensor_batch_to_device(batch: Dict[str, object], device: str) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def evaluate(
    model: CLIPSegForImageSegmentation,
    loader: DataLoader,
    device: str,
    threshold: float,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    per_task_iou: Dict[str, List[float]] = {}
    per_task_dice: Dict[str, List[float]] = {}

    with torch.inference_mode():
        for batch in loader:
            batch = move_tensor_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
            )

            logits = normalize_logits_shape(outputs.logits)
            targets = batch["masks"].unsqueeze(1)
            logits = nn.functional.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            bce = weighted_bce_loss(logits, targets, batch["positive_weights"], batch["task_weights"])
            dice = dice_loss(logits, targets, batch["task_weights"])
            loss = bce + dice
            losses.append(float(loss.item()))

            batch_ious, batch_dices = compute_batch_metrics(logits, targets, threshold=threshold)
            for idx, dataset_name in enumerate(batch["dataset_names"]):
                per_task_iou.setdefault(dataset_name, []).append(batch_ious[idx])
                per_task_dice.setdefault(dataset_name, []).append(batch_dices[idx])

    metrics: Dict[str, float] = {
        "loss": float(sum(losses) / len(losses)) if losses else 0.0,
    }

    all_ious: List[float] = []
    all_dices: List[float] = []
    for dataset_name in sorted(per_task_iou):
        mean_iou = float(sum(per_task_iou[dataset_name]) / len(per_task_iou[dataset_name]))
        mean_dice = float(sum(per_task_dice[dataset_name]) / len(per_task_dice[dataset_name]))
        metrics[f"{dataset_name}_iou"] = mean_iou
        metrics[f"{dataset_name}_dice"] = mean_dice
        all_ious.extend(per_task_iou[dataset_name])
        all_dices.extend(per_task_dice[dataset_name])

    metrics["macro_iou"] = float(sum(all_ious) / len(all_ious)) if all_ious else 0.0
    metrics["macro_dice"] = float(sum(all_dices) / len(all_dices)) if all_dices else 0.0
    return metrics


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return trainable, total


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = CLIPSegProcessor.from_pretrained(args.model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model_id)
    freeze_model_for_first_pass(model, unfreeze_vision_layers=args.unfreeze_vision_layers)
    model.to(args.device)

    trainable_params, total_params = count_trainable_parameters(model)
    print(f"trainable_parameters={trainable_params}")
    print(f"total_parameters={total_params}")

    records = build_records(limit_per_dataset=args.limit_per_dataset)
    train_records, val_records, test_records = split_records(
        records=records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_dataset = PromptedSegmentationDataset(
        train_records,
        split_name="train",
        use_prompt_variants=True,
    )
    val_dataset = PromptedSegmentationDataset(
        val_records,
        split_name="val",
        use_prompt_variants=False,
    )
    test_dataset = PromptedSegmentationDataset(
        test_records,
        split_name="test",
        use_prompt_variants=False,
    )

    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: List[Dict[str, float]] = []
    best_val_iou = float("-inf")

    run_config = {
        "model_id": args.model_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": args.device,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "unfreeze_vision_layers": args.unfreeze_vision_layers,
        "eval_threshold": args.eval_threshold,
        "limit_per_dataset": args.limit_per_dataset,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "dataset_notes": {
            "cracks": "polygon masks when available, bbox fallback for rare empty segmentations",
            "drywall": "all labels are bbox-derived pseudo-masks because the provided COCO export is box-only",
        },
        "prompt_policy": {
            "train": {name: cfg["prompt_variants"] for name, cfg in DATASET_CONFIGS.items()},
            "eval": {name: cfg["canonical_prompt"] for name, cfg in DATASET_CONFIGS.items()},
        },
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
    }
    save_json(run_config, args.output_dir / "run_config.json")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for batch in train_loader:
            batch = move_tensor_batch_to_device(batch, args.device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
            )

            logits = normalize_logits_shape(outputs.logits)
            targets = batch["masks"].unsqueeze(1)
            logits = nn.functional.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            bce = weighted_bce_loss(logits, targets, batch["positive_weights"], batch["task_weights"])
            dice = dice_loss(logits, targets, batch["task_weights"])
            loss = bce + dice

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        train_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0
        val_metrics = evaluate(model, val_loader, device=args.device, threshold=args.eval_threshold)
        epoch_summary: Dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss}
        epoch_summary.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(epoch_summary)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_macro_iou={val_metrics['macro_iou']:.4f} "
            f"val_macro_dice={val_metrics['macro_dice']:.4f}"
        )

        if val_metrics["macro_iou"] > best_val_iou:
            best_val_iou = val_metrics["macro_iou"]
            model.save_pretrained(args.output_dir / "best_model")
            processor.save_pretrained(args.output_dir / "best_model")
            save_json(epoch_summary, args.output_dir / "best_model" / "best_val_metrics.json")

        save_json(
            {
                "history": history,
            },
            args.output_dir / "history.json",
        )

    # Reuse the in-memory model if no best checkpoint was saved for some reason.
    best_model_dir = args.output_dir / "best_model"
    if best_model_dir.exists():
        model = CLIPSegForImageSegmentation.from_pretrained(best_model_dir)
        model.to(args.device)

    test_metrics = evaluate(model, test_loader, device=args.device, threshold=args.eval_threshold)
    save_json(test_metrics, args.output_dir / "test_metrics.json")

    print("test_metrics")
    for key in sorted(test_metrics):
        print(f"{key}={test_metrics[key]:.4f}")


if __name__ == "__main__":
    main()
