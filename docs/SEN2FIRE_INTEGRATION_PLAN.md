# Sen2Fire Dataset Integration Plan

This document outlines how to add the **Sen2Fire** dataset (Australia, binary fire only, no cloud masks) to the existing CEMS-based pipeline, and how to handle clouds and training strategy.

**Note:** CEMS pipeline now uses **8 input channels** (7 spectral bands + NDVI) by default. When integrating Sen2Fire, use the same 8-channel input (compute NDVI from Sen2Fire Red/NIR) so one model handles both datasets. See `fire-pipeline/README.md` and `fire-pipeline/PATCHES.md` for NDVI and `in_channels` options.

---

## Implementation status (done)

| Component | Location | Notes |
|-----------|----------|--------|
| **Dual-head model** | `fire-pipeline/model.py` | `FireDualHeadModel`: binary head (2 classes) + severity head (5). `freeze_severity_head()` for Sen2Fire fine-tune. |
| **CEMS dual-head training** | `fire-pipeline/train.py` | `--dual-head` (requires `--num-classes 5` and GRA patches). Trains both heads; saves `dual_head: true` in config. |
| **Sen2Fire dataset** | `fire-pipeline/sen2fire_dataset.py` | `Sen2FireDataset`: loads .npz, 7/8 ch, center-crop 256. **Cloud filter**: s2cloudless by default when installed, else rule-based (`max_cloud_score`). `use_s2cloudless=True` (default); `--no-s2cloudless` in fine-tune script for rule-based only. Splits: train=scene1+2, val=scene3, test=scene4. |
| **Cloud detection** | `fire-pipeline/cloud_detection.py` | Rule-based `cloud_score_*`; `get_cloud_fraction_s2cloudless()` and `get_cloud_fraction_sen2fire()` (10-band subset for s2cloudless). Optional dep: `s2cloudless` in `train` / `all` extras. |
| **Sen2Fire fine-tune** | `fire-pipeline/train_sen2fire_finetune.py` | Loads CEMS checkpoint (dual or single-head), freezes severity head, trains binary on Sen2Fire. `--checkpoint`, `--sen2fire-dir`, `--output-dir`, `--max-cloud-score`, `--no-s2cloudless`, `--no-ndvi`. |
| **Inference dual-head** | `fire-pipeline/inference.py` | Loads `FireDualHeadModel` when `config.dual_head`; returns `InferenceResult` with `binary_segmentation`, `severity_segmentation`, `binary_probabilities`, `severity_probabilities`. |
| **App: two layers** | `fire-pipeline/app.py` | When `result.dual_head`, two checkboxes: "Show binary fire map", "Show severity map"; each toggles a separate overlay. |
| **History (dual-head)** | `fire-pipeline/storage.py` | Saves/loads `binary_segmentation`, `severity_segmentation`, `binary_probabilities`, `severity_probabilities` in result npz; `metadata.dual_head` so history view shows both layers. |
| **Tests** | `fire-pipeline/tests/` | `test_cloud_detection.py`, `test_sen2fire_dataset.py`, `test_model_dual_head.py`; dual-head save/load in `test_storage.py`; dual-head `InferenceResult` and `create_visualization_from_segmentation` in `test_inference.py`. |

**Quick start (Sen2Fire fine-tune):**
```bash
# 1. Train dual-head on CEMS (GRA patches, NDVI default)
uv run python train.py --patches-dir ./patches --output-dir ./output/dual --num-classes 5 --dual-head

# 2. Fine-tune on Sen2Fire (binary only, severity frozen; s2cloudless used by default)
uv run python train_sen2fire_finetune.py \
  --checkpoint ./output/dual/checkpoints/best_model.pt \
  --sen2fire-dir ../data-sen2fire \
  --output-dir ./output/sen2fire_ft
```

See `fire-pipeline/README.md` section **Full workflow: CEMS + Sen2Fire** for step-by-step run instructions (process CEMS, process Sen2Fire, train, fine-tune) and defaults (NDVI, s2cloudless, dual heads).

---

## 1. Sen2Fire dataset summary

| Aspect | Detail |
|--------|--------|
| **Source** | [Zenodo 10881058](https://zenodo.org/records/10881058), paper: [Sen2Fire arXiv:2403.17884](https://arxiv.org/abs/2403.17884) |
| **Content** | 2,466 patches from 4 bushfire scenes, NSW Australia, 2019–2020 season |
| **Patch size** | 512×512 (vs CEMS 256×256) |
| **Bands** | 12 Sentinel-2 L2A + 1 Sentinel-5P aerosol index (13 channels) |
| **Labels** | Binary fire mask only (0/1), from MOD14A1 V6.1 — **no severity (GRA)** |
| **Splits** | **Train**: scene1 + scene2 (1,458 patches). **Val**: scene3 (504). **Test**: scene4 (504). |
| **Format** | `.npz` per patch: `image` (12, 512, 512) int16, `aerosol` (512, 512), `label` (512, 512) uint8 |
| **Values** | Reflectance-style DN (e.g. ×10000); need to normalize to [0, 1] or similar |
| **Cloud mask** | **None** — we need to infer or filter cloudy patches |

**Band order in Sen2Fire** (from paper Table 1): B1–B12 in array order (B1=Coastal, B2=Blue, B3=Green, B4=Red, B5–B7=Red edge, B8=NIR, B9=Red edge 0.865, B10=Water vapour, B11=SWIR, B12=SWIR). So our 7-channel subset uses indices `[1,2,3,7,8,10,11]` → B02, B03, B04, B08, B9(≈B8A), B11, B12 (B10 in 0-index = water vapour; B8A is band 8 in their table = index 8).

---

## 2. Training strategy: CEMS first, then Sen2Fire

**Recommendation: train on CEMS → fine-tune on Sen2Fire** (with optional joint training variant).

| Approach | Pros | Cons |
|----------|------|------|
| **CEMS only** | Already implemented | No Australia / geographic diversity |
| **Sen2Fire only** | Simple | Small (1,458 train), single region, no severity |
| **Joint from scratch** | One pipeline | Different formats/sizes, label mismatch (severity vs binary), risk of CEMS dominance |
| **CEMS train → Sen2Fire fine-tune** ✓ | Reuse CEMS representation; add Australian data; binary head matches Sen2Fire | Need a clear fine-tune protocol |
| **Joint with two loaders** | Can balance datasets | More complex; need to align input channels and loss (e.g. binary-only on Sen2Fire) |

**Concrete plan:**

1. **Phase 1 – CEMS (current)**  
   - Train segmentation model on CEMS (binary DEL or 5-class GRA, as now).  
   - Output: pretrained backbone + binary (and optionally severity) head.

2. **Phase 2 – Fine-tune on Sen2Fire**  
   - Use **binary fire only** (Sen2Fire has no severity).  
   - Options:  
     - **A)** Freeze encoder, train only decoder/head on Sen2Fire (fast, preserves CEMS features).  
     - **B)** Full fine-tune with small LR (e.g. 1e-5) to adapt to Australian conditions.  
   - Input: map Sen2Fire 12 (+ aerosol) to same **7 or 8 channels** as CEMS (same 7 bands; add NDVI for 8-channel to match current CEMS default).  
   - Patch size: either crop 512→256 from center (or multiple 256×256 crops per 512 patch) so the model sees 256×256 as in CEMS.

3. **Evaluation**  
   - CEMS val/test for European performance.  
   - Sen2Fire val (scene3) during fine-tune; Sen2Fire test (scene4) for final Australian / out-of-region report.

4. **Optional later**  
   - Joint training: two dataloaders (CEMS + Sen2Fire), binary loss on both; severity loss only on CEMS (ignore severity on Sen2Fire). Requires consistent 7-channel input and 256×256 patches from both.

---

## 3. Cloud handling for Sen2Fire (no cloud masks)

Because Sen2Fire has **no cloud mask**, we have three practical options.

### Option A – Rule-based cloud screening (recommended first)

Use spectral heuristics to flag “likely cloudy” patches and **exclude them from training** (and optionally from val/test).

- **Idea**: Clouds are bright in visible (e.g. Blue, Green), often flat in NIR, and we have Water Vapour (B10) which is high over clouds.  
- **Simple proxy**: e.g. mean reflectance in B02 (Blue) or (B02+B03)/2 above a threshold → “cloudy”.  
- **Alternative**: Compute a **simple cloud score** per patch, e.g.  
  - `cloud_score = mean(B02 + B03) / (1 + mean(B08))`  
  - or use B10 (water vapour) if available in the 12-band cube.  
- **Action**: Exclude patches with `cloud_score > threshold` (tune threshold on a few manual checks or on CEMS by comparing to real CM).  
- **Pros**: No extra model, no external API, fast.  
- **Cons**: Heuristic; may drop some usable patches or keep some cloudy ones.

### Option B – s2cloudless (or similar) cloud mask

- Use **s2cloudless** (Sentinel Hub) or another Sentinel-2 cloud detector to predict a cloud mask per patch.  
- Then: either **drop patches** with cloud fraction &gt; X%, or **mask out** cloudy pixels in the loss (same idea as CEMS).  
- **Pros**: Better accuracy than a simple rule.  
- **Cons**: Extra dependency; possibly different resolution; need to run it over all Sen2Fire patches once and save masks or exclude list.

### Option C – Cloud classifier trained on CEMS

- Train a small **binary cloud classifier** on CEMS (input: 7-channel patch, label: from CEMS CM).  
- Run this classifier on Sen2Fire patches; treat “cloud probability” as cloud fraction and exclude (or mask) high-cloud patches.  
- **Pros**: Same sensor/bands as your pipeline; no external service.  
- **Cons**: Need to implement and train; CEMS cloud distribution may differ from Australia.

**Recommendation:** Start with **Option A** (rule-based) to get a clean subset quickly. If you see obvious cloud contamination in visual checks or validation metrics, add **Option B** or **Option C**.

---

## 4. Implementation checklist

### 4.1 Data loading and normalization

- [x] **Sen2Fire reader** (`fire-pipeline/sen2fire_dataset.py`):
  - Load `.npz`: `image` (12, 512, 512), `label` (512, 512).
  - Normalize `image`: `img = np.clip(img.astype(np.float32) / 10000., 0, 1)`.
  - Map to **7 channels** (or 8 with NDVI): indices `[1,2,3,7,8,10,11]`; optionally add NDVI from Red/NIR.
  - Output: **(8, 256, 256)** by center-cropping 256 from 512 (or 7 ch if `include_ndvi=False`).

### 4.2 Patch size alignment (256 vs 512)

- [x] **Center-crop to 256×256**: From each 512×512 patch, center 256×256 (image + label). Same architecture as CEMS.

### 4.3 Cloud screening

- [x] **Rule-based cloud score** (`fire-pipeline/cloud_detection.py`): `cloud_score_sen2fire_12band()`; in `Sen2FireDataset`, skip patches with `cloud_score > max_cloud_score`.
- [ ] Optional: **s2cloudless** — `get_cloud_fraction_s2cloudless()` implemented; not wired into dataset (use when dependency added).

### 4.4 Dataset and split mapping

- [x] **Split mapping**: Train: scene1+scene2; Val: scene3; Test: scene4.
- [x] **Dataset class**: `Sen2FireDataset(root_dir, split, max_cloud_score=..., include_ndvi=True)` returns `(image_tensor, mask_tensor)`.

### 4.5 Training / fine-tuning

- [x] **Fine-tune script** (`train_sen2fire_finetune.py`): Load CEMS checkpoint, freeze severity head, train binary on Sen2Fire; validate on scene3, optional test on scene4.

### 4.6 Severity (GRA) handling

- [x] **Dual-head**: Binary head (2) + severity head (5). Fine-tune on Sen2Fire with severity head **frozen**; only binary loss. App shows two toggleable layers (binary + severity).

---

## 5. Summary

| Question | Answer |
|----------|--------|
| **Use Sen2Fire?** | Yes — for geographic diversity (Australia) and extra binary fire data. |
| **How to use it?** | Train on CEMS first; **fine-tune on Sen2Fire** with binary-only loss; evaluate on CEMS (EU) and Sen2Fire test (Australia). |
| **No severity in Sen2Fire** | Use binary head only when training/fine-tuning on Sen2Fire; severity head only on CEMS if doing joint training. |
| **No cloud masks** | **Rule-based cloud score** (e.g. B02/B03 brightness + optional B10) and exclude high-cloud patches; optionally add s2cloudless or CEMS-trained cloud classifier later. |
| **Patch size** | Prefer **center-crop 512→256** so the rest of the pipeline stays 256×256. |
| **Bands** | Map Sen2Fire 12 bands to **same 7 as CEMS** (B02, B03, B04, B08, B8A, B11, B12); optionally add aerosol as 8th channel for Sen2Fire. |

Next implementation steps: add Sen2Fire loader + cloud filter, then a small fine-tune script that loads CEMS checkpoint and trains on Sen2Fire train/val/test with the above split and options.
