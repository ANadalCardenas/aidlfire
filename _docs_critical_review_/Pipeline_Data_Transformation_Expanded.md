# Data Transformation Pipeline — Expanded Notes (Talking Points)

**Patch generation and training-time data pipeline for the CEMS Wildfire dataset.**

---

## 1. Why a transformation pipeline?

- **Raw tiles** are large (variable size, e.g. hundreds to ~2000 px per side) and 12-band; models need **fixed-size**, **multi-channel** inputs and manageable GPU memory.
- **Patch generation** turns each tile into many **256×256** patches, selects **7 bands**, and filters by **cloud cover** so training sees mostly usable pixels.
- **Training** then loads these patches, optionally **augments** them on the fly, and feeds (image, mask) pairs to the model. Augmentation and **weighted sampling** help with **class imbalance** (few fire pixels overall).

**Talking point**: “We don’t train on full scenes — we slice into 256×256 patches, drop cloudy ones, and optionally oversample and augment fire patches.”

---

## 2. Patch generation (offline)

### 2.1 Inputs

- **Satellite image**: `*_S2L2A.tif` (12-band Sentinel-2 L2A, float32, ~[0, 1]).
- **Mask**: `*_DEL.tif` (binary) or `*_GRA.tif` (5-class severity).
- **Cloud mask**: `*_CM.tif` (0=clear, 1=cloud, 2=light cloud, 3=shadow).

All from the same tile directory; pipeline prefers `.tif` and falls back to `.png` if needed (e.g. HuggingFace assets).

### 2.2 Sliding window

- **Patch size**: **256×256** (constant in `constants.py`: `DEFAULT_PATCH_SIZE = 256`).
- **Stride**:
  - **Training**: **224** → 32 px overlap between adjacent patches → more samples per tile.
  - **Inference**: **256** → no overlap → clean, non-overlapping tiling for prediction.
- Window moves row-by-row and column-by-column; only patches with **cloud cover ≤ threshold** (default **50%**) are kept.

### 2.3 Band selection and values

- From the 12 S2L2A bands, **7** are kept: indices **1, 2, 3, 7, 8, 10, 11** → B02, B03, B04, B08, B8A, B11, B12.
- Rationale: RGB (2,3,4), NIR (8, 8A) and SWIR (11, 12) capture vegetation and burn; fire shows as **low NIR, high SWIR**.
- Values are **clipped to [0, 1]** and stored as **float32**; no extra normalization at save time.

### 2.4 Output layout

- **Files per patch**: `{patch_id}_image.npy`, `{patch_id}_mask.npy`.
- **patch_id** format: `{source_name}_r{row}_c{col}` (e.g. `EMSR230_AOI01_01_r128_c256`).
- **metadata.csv** (per split): `patch_id`, `source_image`, `row`, `col`, `x_min`, `y_min`, `x_max`, `y_max`, `cloud_cover`, `burn_fraction`.
- Use: filtering (e.g. by `burn_fraction` or `cloud_cover`), geolocation, and analysis (e.g. `analyze_patches.py`).

### 2.5 Running the pipeline

- **Entry point**: `run_pipeline.py` (from `fire-pipeline/`).
- **Typical**: `uv run python run_pipeline.py --skip-extraction --output-dir ./patches`
- **Options**: `--mask-type DEL|GRA`, `--max-cloud-cover`, `--splits`, `--force` (regenerate all).

**Talking point**: “Patch generation is offline and deterministic; we only regenerate when we change stride, cloud threshold, or band set.”

---

## 3. Training pipeline: loading and augmentation

### 3.1 Loading

- **WildfirePatchDataset**: scans a split directory for `*_image.npy` / `*_mask.npy` pairs, loads as NumPy, optionally augments, then converts to PyTorch **(C, H, W)** and **(H, W)**.
- **WildfireDataModule**: wires train/val/test under a single `patches_root`, builds DataLoaders, and supports **weighted sampling** and **fire-aware augmentation**.

### 3.2 Standard augmentation (Albumentations)

Applied **on-the-fly** to NumPy (image + mask) before converting to tensor:

| Transform | Typical params | Purpose |
|-----------|----------------|---------|
| HorizontalFlip | p=0.5 | Invariance to viewing direction |
| VerticalFlip | p=0.5 | Same |
| RandomRotate90 | p=0.5 | 0°/90°/180°/270° |
| RandomBrightnessContrast | ±10%, p=0.3 | Atmospheric / illumination variation |
| GaussNoise | var_limit=(0, 0.01), p=0.3 | Robustness to noise |

Same transform is applied to **image and mask** so they stay aligned.

### 3.3 Strong augmentation (fire patches)

Used optionally **only for patches that contain fire** (e.g. >1% fire pixels):

- All of the above with **higher probabilities** (e.g. p=0.5–0.7).
- **ElasticTransform** (e.g. α=100, σ=10, p=0.3) to vary fire shapes.
- **Stronger brightness/contrast** (e.g. ±15%).

This is **fire-aware augmentation**: fire patches get more variation; background-only patches get standard augmentation. Helps with **class imbalance** (Strategy 5 in the spec).

### 3.4 Fire-aware logic

- **fire_threshold**: e.g. **0.01** (1% of pixels are fire) → patch is “fire patch.”
- If `fire_fraction > fire_threshold` and `fire_augment` is set → use **strong** pipeline; else use **standard** (or none).

### 3.5 Weighted sampling

- **Problem**: Many patches have no or very little fire; model can underfit fire class.
- **Option**: `WildfireDataModule(..., use_weighted_sampling=True, fire_sample_weight=5.0)` so that patches with fire are sampled **5×** more often (weights from `get_sample_weights()`).
- Implemented via **WeightedRandomSampler** in the training DataLoader.

### 3.6 Optional normalization

- **After** augmentation and conversion to tensor, a **torchvision** `Normalize(mean, std)` can be applied (e.g. from `compute_dataset_statistics()` over the training set).
- Not applied by default; useful if you want per-channel normalization.

**Talking point**: “Augmentation is geometric and spectral, applied consistently to image and mask; we can optionally treat fire patches with stronger augmentation and oversample them.”

---

## 4. End-to-end flow

1. **Offline**: Extract or clone CEMS data → run `run_pipeline.py` → get `patches/{train,val,test}/` with `.npy` and `metadata.csv`.
2. **Training**: `WildfireDataModule(patches_root, ...)` or `WildfirePatchDataset(patches/train, augment=..., fire_augment=..., ...)` → DataLoader (optionally with weighted sampler) → model.
3. **Inference**: Load full tile or pre-cut patches with **stride 256** (no overlap); no augmentation.

---

## 5. Key files

| File | Role |
|------|------|
| `run_pipeline.py` | CLI entry; extraction + patch generation. |
| `patch_generator.py` | `PatchGenerator`, `PatchConfig`; sliding window, band selection, cloud filter, save .npy + metadata. |
| `constants.py` | `BAND_INDICES`, `DEFAULT_PATCH_SIZE`, `DEFAULT_STRIDE_TRAIN`, `DEFAULT_STRIDE_INFERENCE`, `DEFAULT_MAX_CLOUD_COVER`. |
| `dataset.py` | `WildfirePatchDataset`, `WildfireDataModule`, `get_training_augmentation()`, `get_strong_augmentation()`, weighted sampling. |
| `PATCHES.md` | User-facing doc on patch format, metadata, and augmentation. |

---

## 6. Summary table

| Aspect | Patch generation | Training (data) |
|--------|------------------|------------------|
| **Input** | S2L2A + DEL/GRA + CM GeoTIFFs | `*_image.npy`, `*_mask.npy` |
| **Output** | .npy + metadata.csv | Batches (image, mask) tensors |
| **Size** | 256×256, 7 ch | Same (optionally normalized) |
| **Stride** | 224 (train) / 256 (inf) | N/A |
| **Filter** | Cloud ≤ 50% | Optional weighted sampling |
| **Augmentation** | None | Standard + optional strong (fire-aware) |

---

*Short slide version: [Pipeline_Data_Transformation_Summary.md](Pipeline_Data_Transformation_Summary.md).*
