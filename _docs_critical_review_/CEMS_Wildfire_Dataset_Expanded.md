# CEMS Wildfire Dataset — Expanded Summary (Talking Points)

**1–2 page expanded summary for talks and deep-dive discussions.**

---

## 1. What the dataset is

- **CEMS Wildfire Dataset** combines **Sentinel-2 L2A** multispectral imagery with **CEMS Rapid Mapping** expert delineations of burned area.
- **Goal**: Train models for **burned area delineation** (binary) and optionally **damage severity** (5 levels).
- **Source**: [Copernicus Emergency Management Service](https://emergency.copernicus.eu/) activations (EMSR codes) and [ESA Sentinel-2](https://sentinels.copernicus.eu/) L2A products; dataset version used in e.g. Arnaudo et al., ECML-PKDD 2023.

**Talking point**: Labels are **human-expert** delineations from CEMS, not algorithmic; this makes the dataset suitable for supervised learning and benchmarking.

---

## 2. Scale and scope

| Aspect | Detail |
|--------|--------|
| **Fire events** | ~275 unique EMSR activations |
| **Total tiles** | 500+ georeferenced image tiles (~560 in main metadata) |
| **Time span** | June 2017 – April 2023 |
| **Geography** | Europe, 19 countries; emphasis on Mediterranean (Portugal, Spain, Italy, Greece, France, etc.) |
| **Splits** | Pre-defined train / val / test; optional regional hold-out (e.g. Catalonia in `cat/`) |

**Talking point**: The dataset is **European, post-fire** (acquisitions after the event), and **multi-country**, so models see varied terrain and climate (Köppen metadata available).

---

## 3. Image specifics (what’s in each tile)

### 3.1 Satellite imagery (S2L2A)

- **File**: `EMSR{XXX}_AOI{YY}_{ZZ}_S2L2A.tif`
- **Content**: 12-band Sentinel-2 **Level-2A** (bottom-of-atmosphere reflectance).
- **Data type**: Float32.
- **Value range**: 0.0–1.0 (surface reflectance).
- **CRS**: EPSG:4326 (WGS84).
- **Resolution**: ~10 m for 10 m bands; 20 m and 60 m bands are present and may be resampled in the tile.

**Band layout (12 bands)**:

| Index | Band | Wavelength | Resolution | Role in fire |
|-------|------|------------|------------|--------------|
| 0 | B01 | 443 nm (Coastal) | 60 m | Aerosol |
| 1 | B02 | 490 nm (Blue) | 10 m | RGB, water |
| 2 | B03 | 560 nm (Green) | 10 m | RGB, vegetation |
| 3 | B04 | 665 nm (Red) | 10 m | RGB, chlorophyll |
| 4 | B05 | 705 nm (Red Edge 1) | 20 m | Vegetation |
| 5 | B06 | 740 nm (Red Edge 2) | 20 m | Vegetation |
| 6 | B07 | 783 nm (Red Edge 3) | 20 m | Vegetation |
| 7 | B08 | 842 nm (NIR) | 10 m | **Vegetation health (drops after fire)** |
| 8 | B8A | 865 nm (NIR narrow) | 20 m | **Vegetation / burn** |
| 9 | B09 | 945 nm (Water vapour) | 60 m | Atmosphere |
| 10 | B11 | 1610 nm (SWIR1) | 20 m | **Fire / burn (strong signal)** |
| 11 | B12 | 2190 nm (SWIR2) | 20 m | **Fire / burn (strong signal)** |

**Modelling convention**: We typically use **7 channels**: B02, B03, B04, B08, B8A, B11, B12 (indices 1,2,3,7,8,10,11). Burned area is characterised by **low NIR (B08, B8A)** and **high SWIR (B11, B12)**.

**Talking point**: Emphasise that the **physical basis** for detection is spectral: vegetation loses NIR reflectance and increases SWIR when burned; the model learns this from the 7-band input.

### 3.2 Labels and auxiliary rasters

- **DEL** (`*_DEL.tif`): Binary delineation (0 = not burned, 1 = burned). **Primary target** for binary segmentation.
- **GRA** (`*_GRA.tif`): Severity grading 0–4 (no damage → destroyed). **Optional** target; not available for every tile.
- **CM** (`*_CM.tif`): Cloud mask (0=clear, 1=cloud, 2=light cloud/haze, 3=shadow). Used to **mask out** unreliable pixels in training/inference.
- **Land cover** (e.g. `*_ESA_LC.tif`, `*_Esri10_LC.tif`, `*_Annual9_LC.tif`): Optional for stratification or multi-task use.

**Talking point**: DEL is always the main task; GRA and land cover support richer analyses and optional multi-task or stratified setups.

---

## 4. Visual examples (layers in the dataset)

Below are example visuals for **one tile** (EMSR382, AOI01, 01). They show what the dataset “looks like” in practice.

### 4.1 True-color (RGB) satellite image

The standard visualisation of the multispectral image (B04, B03, B02 → R, G, B).

![S2L2A true-color](../assets/sample/EMSR382/AOI01/EMSR382_AOI01_01/EMSR382_AOI01_01_S2L2A.png)

### 4.2 DEL mask (binary burned area)

Expert-drawn delineation: red = burned, background = not burned.

![DEL mask](../assets/sample/EMSR382/AOI01/EMSR382_AOI01_01/EMSR382_AOI01_01_DEL.png)

### 4.3 GRA mask (severity)

Five levels from no damage to destroyed; only available for a subset of tiles.

![GRA mask](../assets/sample/EMSR382/AOI01/EMSR382_AOI01_01/EMSR382_AOI01_01_GRA.png)

### 4.4 Cloud mask (CM)

Used to exclude cloudy and shadowed pixels from training and evaluation.

![Cloud mask](../assets/sample/EMSR382/AOI01/EMSR382_AOI01_01/EMSR382_AOI01_01_CM.png)

### 4.5 Land cover (optional layers)

ESA WorldCover and Esri 10-class land use/land cover — useful for stratification or multi-task learning.

![ESA WorldCover](../assets/sample/EMSR382/AOI01/EMSR382_AOI01_01/EMSR382_AOI01_01_ESA_LC.png)

![Esri 10-class LC](../assets/sample/EMSR382/AOI01/EMSR382_AOI01_01/EMSR382_AOI01_01_Esri10_LC.png)

### 4.6 Geographic distribution

Fire events are spread across Europe; Mediterranean regions dominate.

![Distribution Europe](../assets/images/distribution_Europe_Map_DEL.png)

**Talking point**: When presenting, walk through: “Input is 7-channel reflectance; labels are DEL (and optionally GRA); we use the cloud mask to avoid learning from obscured pixels.”

---

## 5. From tiles to training (patches)

- **Tile size**: Variable (e.g. hundreds to ~2000 pixels per side); exact size per tile in `satelliteData.csv` (`height`, `width`).
- **Patch size**: **256×256** pixels.
- **Patch format**: NumPy `.npy` — image `(256, 256, 7)` float32, mask `(256, 256)` uint8.
- **Bands in patches**: Same 7 bands (B02, B03, B04, B08, B8A, B11, B12), normalised to [0, 1].
- **Stride**: Often 50% overlap (e.g. stride 128) for training to increase samples; no overlap for clean inference tiling.

**Talking point**: We don’t feed full tiles to the model; we slice them into 256×256 patches and optionally filter by cloud cover and burn fraction (see `metadata.csv` in each split).

---

## 6. Metadata and filtering

- **satelliteData.csv**: Per-tile metadata — EMSR, AOI, country, dates, bounding box, resolution, `height`, `width`, `pixelBurned`, Köppen, etc.
- **cloudCoverage.csv**: Per-image cloud and burn statistics (`percentageCloud`, `percentageOverlap`, etc.).
- **Filtering**: Common filters include max cloud cover (e.g. &lt;50% for patches), minimum burn fraction for oversampling, and country/date for regional or temporal studies.

**Talking point**: Class imbalance is typical (few fire pixels overall); strategies include weighted sampling, weighted loss, and fire-aware augmentation (see project `dataset.py` and `analyze_patches.py`).

---

## 7. Intended use and limitations

- **Intended**: Binary (and optionally severity) segmentation of **burned area** from Sentinel-2 L2A; research and benchmarking; possible extension to near-real-time pipelines with same sensor.
- **Limitations**: (1) European, post-fire imagery — not global and not “active fire”; (2) GRA and some land cover products not on all tiles; (3) ~4% of images have no burned pixels (edge tiles); (4) cloud mask must be respected to avoid biased learning.

**Talking point**: “This is a **delineation** dataset (where did it burn?), not an **active fire** dataset (where is it burning right now?).”

---

## 8. References and links

- **Data**: [HuggingFace – links-ads/wildfires-cems](https://huggingface.co/datasets/links-ads/wildfires-cems); [GitHub – CEMS-Wildfire-Dataset](https://github.com/MatteoM95/CEMS-Wildfire-Dataset).
- **Citation**: Arnaudo et al., “Robust Burned Area Delineation through Multitask Learning,” ECML-PKDD 2023.
- **Project docs**: `DATA_REFERENCE.md`, `fire-pipeline/PATCHES.md`, `AIDL_PROJECT_GUIDE.md`, `AGENTS.md`.

---

*Short slide version: [CEMS_Wildfire_Dataset_Summary.md](CEMS_Wildfire_Dataset_Summary.md).*
