#!/usr/bin/env python3
"""
Plot training curves from a CLIPSeg training run.

Generates:
  - training loss curve
  - validation macro IoU / Dice curves
  - per-task validation IoU curves

Example:
  python3 scripts/plot_training.py --run-dir runs/clipseg_v2 --output-dir figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.run_dir / "history.json").open() as f:
        history = json.load(f)["history"]

    epochs = [int(h["epoch"]) for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    macro_iou = [h["val_macro_iou"] for h in history]
    macro_dice = [h["val_macro_dice"] for h in history]
    cracks_iou = [h["val_cracks_iou"] for h in history]
    drywall_iou = [h["val_drywall_iou"] for h in history]
    cracks_dice = [h["val_cracks_dice"] for h in history]
    drywall_dice = [h["val_drywall_dice"] for h in history]

    # --- Plot 1: Loss curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, "o-", label="Train Loss", color="#2196F3")
    ax.plot(epochs, val_loss, "s-", label="Val Loss", color="#FF5722")
    ax.set_xlabel("Epoch")
    ax.set_xticks(epochs)
    ax.set_ylabel("Loss (BCE + Dice)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Macro IoU & Dice ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, macro_iou, "o-", label="Macro IoU", color="#4CAF50")
    ax.plot(epochs, macro_dice, "s-", label="Macro Dice", color="#9C27B0")
    ax.set_xlabel("Epoch")
    # x axis should be integer epochs
    ax.set_xticks(epochs)
    ax.set_ylabel("Score")
    ax.set_title("Validation Macro IoU & Dice")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(args.output_dir / "macro_metrics.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Per-task IoU ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, cracks_iou, "o-", label="Cracks IoU", color="#F44336")
    ax.plot(epochs, drywall_iou, "s-", label="Drywall IoU", color="#2196F3")
    ax.plot(epochs, cracks_dice, "^--", label="Cracks Dice", color="#EF9A9A")
    ax.plot(epochs, drywall_dice, "v--", label="Drywall Dice", color="#90CAF9")
    ax.set_xlabel("Epoch")
    ax.set_xticks(epochs)
    ax.set_ylabel("Score")
    ax.set_title("Per-Task Validation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(args.output_dir / "per_task_metrics.png", dpi=150)
    plt.close(fig)

    print(f"Saved plots to {args.output_dir}/")


if __name__ == "__main__":
    main()
