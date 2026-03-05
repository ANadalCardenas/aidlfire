# Training Pipeline — Summary (Slide)

**One-slide overview: training the fire segmentation model (CEMS Wildfire).**

---

## Training at a Glance

| Stage | What it does |
|-------|----------------|
| **Entry** | `train.py` — loads patches, builds model, trains with validation and checkpointing. |
| **Model** | Segmentation backbone (U-Net / U-Net++ / DeepLabV3+) with pretrained encoder; 7 input channels, `num_classes` output (2 = DEL, 5 = GRA). |
| **Loss** | **Combined**: 0.5 × CrossEntropy + 0.5 × Dice; optional Focal instead of CE; optional class weights. |
| **Optimizer** | AdamW (lr=1e-4, weight_decay=1e-4); ReduceLROnPlateau on validation fire IoU. |
| **Metrics** | Segmentation (IoU, Dice, precision, recall per class; **fire_iou** primary) + patch-level detection (accuracy, precision, recall, F1). |
| **Checkpointing** | Best model by **fire_iou**; periodic every N epochs; final; resume supported. |
| **Stopping** | Early stopping after 10 epochs without improvement (configurable). |

---

## Model

- **Wrapper**: `FireSegmentationModel` (in `model.py`) — forward returns logits `(B, num_classes, H, W)`; helpers: `predict_segmentation`, `predict_fire_detection`, `predict_fire_confidence`.
- **Backbones**: ResNet18/34/50, EfficientNet-B0/B1/B2, MobileNetV2 (pretrained ImageNet; first layer adapted to **7 channels**).
- **Architectures**: U-Net (default), U-Net++, DeepLabV3+ (via `segmentation_models_pytorch`).
- **Input/Output**: (B, 7, 256, 256) → (B, num_classes, 256, 256).

---

## Loss and class imbalance

- **CombinedLoss**: **CE** (pixel-wise) + **Dice** (region-based); default 50/50. Good for imbalanced fire pixels.
- **Class weights**: Optional inverse-frequency weights from training set (`compute_class_weights`); applied in CE (or Focal).
- **Focal loss** (optional): Replaces CE with Focal (γ=2) to down-weight easy examples; helps when fire is rare.
- **Data-side**: Weighted sampling and fire-aware augmentation are configured in the data module (see [Pipeline_Data_Transformation_Summary.md](Pipeline_Data_Transformation_Summary.md)).

---

## Metrics and selection

- **Primary for best model**: **fire_iou** (IoU over all fire classes, i.e. class > 0). Scheduler also uses fire_iou (maximize).
- **Logged**: train/val loss; val fire IoU, fire recall, detection F1; per-class IoU/Dice if needed.
- **Detection**: Patch-level “has fire” if any pixel predicted as fire; detection accuracy, precision, recall, F1.

---

## Quick reference

| Item | Default |
|------|---------|
| Batch size | 16 |
| Epochs | 50 |
| Learning rate | 1e-4 |
| Encoder | resnet34 |
| Architecture | unet |
| Best model | By val fire_iou |
| Early stopping | 10 epochs |

**Run**: `uv run python train.py --patches-dir ./patches --num-classes 2` (optional: `--wandb --project fire-detection`, `--resume checkpoints/best_model.pt`).

---

*For details and talking points, see [Training_Pipeline_Expanded.md](Training_Pipeline_Expanded.md).*
