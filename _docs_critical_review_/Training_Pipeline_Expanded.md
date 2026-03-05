# Training Pipeline — Expanded Notes (Talking Points)

**End-to-end training of the fire segmentation model: model, loss, metrics, checkpointing, and CLI.**

---

## 1. Overview

- **Goal**: Train a **semantic segmentation** model on CEMS wildfire patches (7-channel input, pixel-wise labels) and optionally use it for **binary fire detection** (patch-level “has fire” derived from segmentation).
- **Entry point**: `train.py` in `fire-pipeline/`. It builds data loaders (from the [data transformation pipeline](Pipeline_Data_Transformation_Expanded.md)), model, loss, optimizer, scheduler, and metrics; runs train/val loops; saves checkpoints and supports resume and W&B.
- **Outputs**: `output_dir/config.json`, `output_dir/checkpoints/best_model.pt`, `final_model.pt`, and periodic checkpoints.

**Talking point**: “One training script covers both DEL (binary) and GRA (5-class); we pick the best model by validation fire IoU and can resume or log to W&B.”

---

## 2. Model

### 2.1 Architecture

- **Library**: `segmentation_models_pytorch` (SMP). Encoder is pretrained on ImageNet; **first conv is adapted to 7 input channels** (Sentinel-2 bands).
- **Encoder options**: `resnet18`, `resnet34`, `resnet50`, `efficientnet-b0`, `efficientnet-b1`, `efficientnet-b2`, `mobilenet_v2`. Default: **resnet34**.
- **Decoder / head options**: **unet**, **unetplusplus**, **deeplabv3plus**. Default: **unet**.
- **Output**: Logits of shape **(B, num_classes, H, W)**. For DEL: num_classes=2 (background, fire). For GRA: num_classes=5 (no damage → destroyed).

### 2.2 FireSegmentationModel wrapper

- **forward(x)** → logits (B, C, H, W).
- **predict_segmentation(x)** → class index per pixel (B, H, W).
- **predict_probabilities(x)** → softmax (B, C, H, W).
- **predict_fire_detection(x)** → boolean (B,) “any pixel is fire?”
- **predict_fire_confidence(x)** → max fire probability over pixels (B,) in [0, 1].
- **predict_fire_fraction(x)** → fraction of pixels predicted as fire (B,).

So a **single model** does pixel-wise segmentation and patch-level detection/confidence.

**Talking point**: “We don’t train a separate classifier; detection is derived from segmentation by ‘any fire pixel in the patch’.”

---

## 3. Loss

### 3.1 CombinedLoss

- **Formula**: `total = 0.5 * CE + 0.5 * Dice`.
- **CE**: Pixel-wise cross-entropy; optional **class weights** (inverse frequency); optional **Focal** variant (γ=2) instead of plain CE.
- **Dice**: 1 − mean(Dice per class); smooth term 1e-6. Helps with **imbalanced** fire pixels (region-based).
- **Returns**: (total_loss, dict with `ce_loss`, `dice_loss`) for logging.

### 3.2 Class weights and Focal

- **Class weights**: If `use_class_weights=True`, weights are computed from the **training** split (`compute_class_weights(patches_dir/train, num_classes)`) and passed to CE (or Focal). Fire class(es) get higher weight.
- **Focal**: `--focal-loss` switches CE to Focal (same weights, γ=2 by default). Down-weights easy pixels and focuses on hard ones.

**Talking point**: “We combine CE and Dice and can add class weights or Focal to handle rare fire pixels.”

---

## 4. Optimizer and scheduler

- **Optimizer**: **AdamW** (lr=1e-4, weight_decay=1e-4).
- **Scheduler**: **ReduceLROnPlateau** on **validation fire_iou** (mode=max): factor=0.5, patience=5. So LR is reduced when val fire IoU stops improving.

---

## 5. Metrics

### 5.1 CombinedMetrics

- **SegmentationMetrics**: Confusion matrix over pixels; then per-class and aggregate:
  - **IoU**, **Dice**, **Precision**, **Recall** per class.
  - **mean_iou**, **mean_dice**.
  - **fire_iou**, **fire_dice**, **fire_precision**, **fire_recall** (all classes with index > 0 pooled).
- **DetectionMetrics**: Patch-level. “Has fire” = any pixel predicted as fire; same for target. Then **accuracy**, **precision**, **recall**, **F1** (and tp/fp/tn/fn).

### 5.2 What is used for model selection and logging

- **Best checkpoint**: Saved when **validation fire_iou** improves. This is the **primary** metric.
- **Scheduler**: Steps on **val fire_iou** (maximize).
- **Printed each epoch**: Train/val loss; val fire_iou, fire_recall, detection_f1.
- **W&B** (if enabled): train/val loss, train/val fire_iou, train/val detection_f1, val fire_recall, learning rate.

**Talking point**: “We optimize for fire IoU; detection F1 and recall are monitored but the saved best model is by fire IoU.”

---

## 6. Checkpointing and early stopping

- **Best model**: Saved to `checkpoints/best_model.pt` whenever val fire_iou improves.
- **Periodic**: Every `save_every` epochs (default 5) → `checkpoint_epoch_{epoch}.pt`.
- **Final**: At end of training → `final_model.pt`.
- **Resume**: `--resume path/to/checkpoint.pt` loads model and optimizer state, continues from next epoch; best_metric is restored from checkpoint for early stopping.
- **Early stopping**: If validation fire_iou does not improve for **patience** epochs (default 10), training stops; best and final checkpoints are still saved.

---

## 7. Data and class imbalance (training side)

- **Data module**: `WildfireDataModule(patches_root, batch_size, ...)` builds train/val (and test) loaders. Uses **standard augmentation** by default; optional **fire_augment** (stronger for fire patches) and **weighted sampling** (oversample fire patches). See [Pipeline_Data_Transformation_Expanded.md](Pipeline_Data_Transformation_Expanded.md).
- **CLI**: `--weighted-sampling`, `--fire-weight` (default 5.0), `--no-fire-augment` to disable fire-specific augmentation.
- So **class imbalance** is addressed by: (1) loss: class weights and/or Focal; (2) data: weighted sampling and fire-aware augmentation.

---

## 8. CLI and config

**Essential arguments**:

- `--patches-dir`: Path to directory with `train/`, `val/`, `test/` (default `./patches`).
- `--output-dir`: Where to write config, checkpoints (default `./output`).
- `--num-classes`: 2 (DEL) or 5 (GRA).

**Model**: `--encoder` (e.g. resnet34), `--architecture` (unet, unetplusplus, deeplabv3plus).

**Training**: `--batch-size`, `--epochs`, `--lr`, `--weight-decay`, `--num-workers`.

**Loss**: `--no-class-weights`, `--focal-loss`, `--focal-gamma`.

**Sampling/augment**: `--weighted-sampling`, `--fire-weight`, `--no-fire-augment`.

**Checkpointing**: `--resume`, `--save-every`, `--patience`.

**Logging**: `--wandb`, `--project`, `--run-name`.

**Device**: `--device` (auto, cuda, mps, cpu).

**Config**: Full run config is written to `output_dir/config.json` (patches_dir, num_classes, encoder, architecture, batch_size, epochs, lr, flags, etc.).

---

## 9. Key files

| File | Role |
|------|------|
| `train.py` | CLI, train/val loops, checkpointing, W&B, early stopping. |
| `model.py` | FireSegmentationModel, create_segmentation_model, CombinedLoss, DiceLoss, FocalLoss. |
| `metrics.py` | SegmentationMetrics, DetectionMetrics, CombinedMetrics. |
| `dataset.py` | WildfireDataModule, augmentation, weighted sampling (used by train.py). |
| `constants.py` | get_device, get_class_names, NUM_INPUT_CHANNELS, etc. |

---

## 10. End-to-end flow

1. **Prepare data**: Run [patch generation](Pipeline_Data_Transformation_Summary.md) → `patches/train`, `patches/val`, `patches/test`.
2. **Train**: `uv run python train.py --patches-dir ./patches --num-classes 2` (add `--wandb` if desired).
3. **Output**: `output/checkpoints/best_model.pt` (and config, final, periodic).
4. **Inference**: Load `best_model.pt` in `inference.py` or app for segmentation and fire detection.

**Talking point**: “Training is one script; we choose DEL vs GRA with --num-classes and pick the best model by fire IoU; detection comes from segmentation for free.”

---

*Short slide version: [Training_Pipeline_Summary.md](Training_Pipeline_Summary.md).*
