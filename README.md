# Prompted Segmentation for Drywall QA

## Assignment Goal

Per [Prompted_Segmentation_for_Drywall_QA.pdf](/home/ritvik/finetuning_task/Prompted_Segmentation_for_Drywall_QA.pdf), the task is to train or fine-tune a text-conditioned segmentation model that takes an image plus a natural-language prompt and produces a binary mask for:

- `segment crack` using Dataset 2 (`cracks.coco`)
- `segment taping area` using Dataset 1 (`Drywall-Join-Detect.coco`)

The assignment also allows prompt variants during training, including:

- `segment wall crack`
- `segment joint/tape`
- `segment drywall seam`

This repository currently contains dataset exports and the assignment PDF, not a training pipeline. The checked-in assets are two Roboflow COCO datasets:

- `cracks.coco`
- `Drywall-Join-Detect.coco`

Both datasets are exported as COCO JSON under a single `train/` directory, with images resized to `640x640`.

## Repository Layout

```text
.
├── README.md
├── Prompted_Segmentation_for_Drywall_QA.pdf
├── cracks.coco/
│   ├── README.roboflow.txt
│   └── train/
│       ├── _annotations.coco.json
│       └── *.jpg
└── Drywall-Join-Detect.coco/
    ├── README.roboflow.txt
    └── train/
        ├── _annotations.coco.json
        └── *.jpg
```

There are no `val/` or `test/` splits in the current export.

## Dataset Summary

### `cracks.coco`

- Exported from Roboflow on February 28, 2026.
- `5000` images.
- `7930` annotations.
- All images are `640x640`.
- COCO category IDs include:
  - `1`: `NewCracks - v2 2024-05-18 10:54pm` (used by all annotations)
  - `0`: `crack` (present in `categories`, unused in annotations)
- Annotation format is mostly segmentation-ready:
  - `7890` annotations have non-empty polygon `segmentation`
  - `40` annotations have empty `segmentation`

### `Drywall-Join-Detect.coco`

- Exported from Roboflow on February 28, 2026.
- `1022` images.
- `1424` annotations.
- All images are `640x640`.
- COCO category IDs include:
  - `1`: `drywall-join` (used by all annotations)
  - `0`: `Drywall-Join` (present in `categories`, unused in annotations)
- Annotation format is box-only COCO:
  - all `1424` annotations have empty `segmentation`
  - bounding boxes are present in `bbox`

## Assignment Output Requirements

The PDF requires prediction masks to be:

- PNG
- single-channel
- the same spatial size as the source image
- pixel values `{0,255}`

Expected filenames should include both image id and prompt, for example:

- `123__segment_crack.png`

The report is expected to include:

- approach and model tried
- short goal summary
- data split counts
- metrics
- 3 to 4 visual examples in `original | ground truth | prediction` format
- brief failure notes
- runtime and footprint details: training time, average inference time per image, model size

The grading rubric in the PDF is:

- Correctness (`50`): `mIoU` and `Dice` on both prompts
- Consistency (`30`): stability across varied scenes
- Presentation (`20`): clear README, documented seeds, clear report with tables and visuals

## COCO JSON Format In This Repo

Each dataset uses a standard COCO-style file at `train/_annotations.coco.json` with these top-level keys:

- `info`
- `licenses`
- `images`
- `annotations`
- `categories`

### Image Entries

Each item in `images` looks like:

```json
{
  "id": 0,
  "license": 1,
  "file_name": "example.jpg",
  "height": 640,
  "width": 640,
  "date_captured": "2026-02-28T13:42:15+00:00",
  "extra": {
    "name": "original_filename.jpg"
  }
}
```

### Annotation Entries

Each item in `annotations` contains:

- `id`
- `image_id`
- `category_id`
- `bbox`
- `area`
- `iscrowd`
- `segmentation`

Example segmentation annotation from `cracks.coco`:

```json
{
  "id": 1,
  "image_id": 0,
  "category_id": 1,
  "bbox": [300, 9, 34, 631],
  "area": 0,
  "iscrowd": 0,
  "segmentation": [[327.641, 640, 328.61, 633.696]]
}
```

Example box-only annotation from `Drywall-Join-Detect.coco`:

```json
{
  "id": 1,
  "image_id": 0,
  "category_id": 1,
  "bbox": [315, 118, 29, 379],
  "area": 0,
  "iscrowd": 0,
  "segmentation": []
}
```

## Important Dataset Mismatch

The assignment expects segmentation supervision for both prompts, but the datasets currently checked into this repository do not match that requirement equally:

- `cracks.coco` is usable for segmentation, with a small number of annotations missing polygon masks.
- `Drywall-Join-Detect.coco` does not currently provide segmentation polygons or masks; it only provides bounding boxes in COCO format.

That means true binary-mask supervision is available for the crack task, but not for the drywall-join task unless you do one of the following:

- re-export `Drywall-Join-Detect` with segmentation annotations
- generate pseudo-masks from bounding boxes
- manually add segmentation masks

Any model training or evaluation described in this repository should document that limitation clearly, since it affects both feasibility and fairness relative to the grading rubric.
