# Data Transformation Pipeline — Summary (Slide)

**One-slide overview: patch generation and training-time augmentation (CEMS Wildfire).**

---

## Pipeline at a Glance

| Stage | What it does |
|-------|----------------|
| **Input** | Raw CEMS tiles: 12-band S2L2A GeoTIFF, DEL/GRA mask, cloud mask (CM). |
| **Patch generation** | Sliding 256×256 window; 7 bands selected; cloudy patches rejected; output `.npy` + `metadata.csv`. |
| **Training load** | Load patches → optional augmentation → tensor (7, 256, 256) + mask (256, 256). |
| **Augmentation** | On-the-fly (Albumentations): geometric + spectral; optional fire-aware and weighted sampling. |

---

## Patch generation (offline)

- **Patch size**: **256×256** pixels (fixed for GPU and batch training).
- **Stride**: **224** for training (32 px overlap → more samples); **256** for inference (no overlap, clean tiling).
- **Bands**: **7** from 12-band S2L2A: B02, B03, B04, B08, B8A, B11, B12 (Blue, Green, Red, NIR, NIR-narrow, SWIR1, SWIR2). Values clipped to [0, 1], float32.
- **Cloud filter**: Patches with **>50%** cloudy pixels (from CM) are **skipped**.
- **Output**: `{patch_id}_image.npy` (256, 256, 7), `{patch_id}_mask.npy` (256, 256) uint8, plus `metadata.csv` (patch_id, source_image, row, col, bounds, cloud_cover, burn_fraction).

---

## Training pipeline (data side)

- **Loader**: `WildfirePatchDataset` / `WildfireDataModule` from `dataset.py`; reads `.npy` pairs, returns image (C, H, W) and mask (H, W).
- **Augmentation** (on-the-fly, Albumentations):
  - **Standard**: horizontal/vertical flip (p=0.5), random 90° rotation (p=0.5), brightness/contrast (±10%, p=0.3), Gaussian noise (p=0.3).
  - **Strong** (optional, for fire patches): same + elastic deformation, higher probabilities, ±15% brightness/contrast.
- **Fire-aware**: Patches with **>1%** fire pixels can use **strong** augmentation; others use standard (reduces class imbalance effect).
- **Weighted sampling**: Optional oversampling of fire patches (e.g. 5×) via `WeightedRandomSampler` to balance fire vs no-fire.

---

## Flow (high level)

```
GeoTIFFs (S2L2A, DEL, CM)  →  sliding window  →  cloud filter  →  save .npy + metadata
                                                                        ↓
Training: load .npy  →  augment (optional)  →  to tensor  →  DataLoader  →  model
```

---

## Quick reference

| Item | Value |
|------|--------|
| Patch size | 256×256 |
| Stride (train) | 224 (overlap 32 px) |
| Stride (inference) | 256 (no overlap) |
| Channels | 7 (B02, B03, B04, B08, B8A, B11, B12) |
| Max cloud cover | 50% (configurable) |
| Mask types | DEL (binary) or GRA (5-class) |

---

*For details and talking points, see [Pipeline_Data_Transformation_Expanded.md](Pipeline_Data_Transformation_Expanded.md).*
