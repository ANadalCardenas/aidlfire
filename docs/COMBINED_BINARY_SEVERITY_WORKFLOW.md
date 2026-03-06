# Combined Binary + Severity Training Workflow

This document describes the **two-phase training workflow** for wildfire detection and severity assessment:

1. **Phase 1**: Train a binary fire model on combined CEMS + Sen2Fire data (with vegetation/NDVI)
2. **Phase 2**: Fine-tune the severity head on CEMS GRA data only (with vegetation/NDVI)

This approach maximizes geographic diversity for fire detection (Europe + Australia) before specializing on severity (CEMS-only, since Sen2Fire has no severity labels).

---

## Table of Contents

- [Overview](#overview)
- [Why This Workflow?](#why-this-workflow)
- [Vegetation (NDVI)](#vegetation-ndvi)
- [Phase 1: Combined Binary Training](#phase-1-combined-binary-training)
- [Phase 2: Severity Fine-Tuning](#phase-2-severity-fine-tuning)
- [Two-Head Model Rationale](#two-head-model-rationale)
- [Data Requirements](#data-requirements)
- [Implementation Status](#implementation-status)

---

## Overview

| Phase | Data | Model | Output |
|-------|------|-------|--------|
| **1** | CEMS DEL + Sen2Fire (combined) | Single-head binary (2 classes) | Binary fire detection model |
| **2** | CEMS GRA only | Dual-head (binary frozen + severity trained) | Binary + severity model |

**Input channels**: 8 (7 spectral bands + NDVI) for both phases.

---

## Why This Workflow?

| Aspect | Benefit |
|--------|---------|
| **Phase 1 – Combined binary** | More fire examples from two continents (Europe + Australia); better generalization |
| **Phase 2 – Severity on CEMS** | Severity (GRA) exists only in CEMS; fine-tune a dedicated head without diluting binary performance |
| **Two-phase separation** | Binary head learns from all data; severity head learns only where labels exist |
| **Vegetation (NDVI)** | Helps separate burn scars from water, shadow, and soil; improves both detection and severity |

---

## Vegetation (NDVI)

**NDVI** (Normalized Difference Vegetation Index) is used as the 8th input channel:

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$

- **High NDVI** → healthy vegetation  
- **Low NDVI** → burned areas, water, bare soil, shadow  

Both CEMS and Sen2Fire support 8-channel input (7 bands + NDVI). Use NDVI by default; disable with `--no-ndvi` if needed.

---

## Phase 1: Combined Binary Training

**Goal**: Train a binary fire detection model on CEMS + Sen2Fire.

### Data

- **CEMS**: DEL patches (binary mask: 0=no fire, 1=fire)
- **Sen2Fire**: Binary patches (scene1 + scene2 for train; scene3 for val; scene4 for test)
- **Combined**: `ConcatDataset` of both loaders

### Model

- Single-head binary (2 classes)
- 8 input channels (7 bands + NDVI)
- Same architecture as the binary head of `FireDualHeadModel` (encoder + decoder + segmentation head)

### Training

```bash
# 1. Generate CEMS DEL patches (if not already done)
uv run python run_pipeline.py \
    --dataset-dir ../wildfires-cems \
    --output-dir ./patches \
    --mask-type DEL

# 2. Ensure Sen2Fire data is available (scene1, scene2, scene3, scene4)

# 3. Train combined binary model (script: train_combined_binary.py)
uv run python train_combined_binary.py \
    --patches-dir ./patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/combined_binary \
    --epochs 50
```

**Output**: Checkpoint with encoder, decoder, and binary head. No severity head yet.

---

## Phase 2: Severity Fine-Tuning

**Goal**: Add a severity head and train it on CEMS GRA data.

### Data

- **CEMS GRA patches only** (5 severity classes: no damage, negligible, moderate, high, destroyed)
- Generate with: `run_pipeline.py --mask-type GRA --output-dir ./patches_gra`

### Model

- `FireDualHeadModel`: binary head (from Phase 1) + severity head (new)
- **Frozen**: encoder, decoder, binary head
- **Trained**: severity head only

### Training

```bash
# 1. Generate CEMS GRA patches (if not already done)
uv run python run_pipeline.py \
    --dataset-dir ../wildfires-cems \
    --output-dir ./patches_gra \
    --mask-type GRA

# 2. Fine-tune severity head (script: train_severity_finetune.py)
uv run python train_severity_finetune.py \
    --checkpoint ./output/combined_binary/checkpoints/best_model.pt \
    --patches-dir ./patches_gra \
    --output-dir ./output/severity_finetune \
    --epochs 30
```

**Output**: Dual-head checkpoint (binary + severity). Use with inference and the app for both fire detection and severity maps.

---

## Two-Head Model Rationale

Phase 2 uses a **2-head model** (`FireDualHeadModel`) because:

1. **Inference**: One model provides both binary fire detection and severity maps.
2. **Hierarchy**: Severity is only meaningful where fire exists; binary acts as a gate.
3. **Stability**: Phase 1 binary head stays frozen; only the severity head is trained.
4. **Reuse**: Existing `FireDualHeadModel` and inference pipeline support this setup.

The severity head is a single `Conv2d` layer on top of the shared decoder output. It is randomly initialized in Phase 2 and trained only on CEMS GRA data.

---

## Data Requirements

| Dataset | Phase | Mask type | Patches |
|---------|-------|-----------|---------|
| CEMS | 1 | DEL (binary) | `./patches` (train/val/test) |
| Sen2Fire | 1 | Binary | `../data-sen2fire` (scene1–4) |
| CEMS | 2 | GRA (severity) | `./patches_gra` (train/val/test) |

**Note**: Not all CEMS images have GRA masks. Check `satelliteData.csv` column `GRA` for availability.

---

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| `FireDualHeadModel` | Done | `fire-pipeline/model.py` |
| CEMS dataset (DEL, GRA) | Done | `fire-pipeline/dataset.py`, `run_pipeline.py` |
| Sen2Fire dataset | Done | `fire-pipeline/sen2fire_dataset.py` |
| NDVI (8 channels) | Done | `patch_generator.py`, `constants.py` |
| **Phase 1: `train_combined_binary.py`** | Done | `fire-pipeline/train_combined_binary.py` |
| **Phase 2: `train_severity_finetune.py`** | Done | `fire-pipeline/train_severity_finetune.py` |
| Inference (dual-head) | Done | `fire-pipeline/inference.py` |
| App (binary + severity layers) | Done | `fire-pipeline/app.py` |

---

## Quick Reference

```
Phase 1: CEMS DEL + Sen2Fire → binary model (8 ch, NDVI)
Phase 2: Load Phase 1 → add severity head → train severity on CEMS GRA (8 ch, NDVI)
Result:  Dual-head model for fire detection + severity assessment
```

## See Also

- **[V3 Pipeline: Architectures & Training Commands](V3_PIPELINE_ARCHITECTURES.md)** — Encoder/decoder options, recommended model combos, and copy-paste commands for each variant.
