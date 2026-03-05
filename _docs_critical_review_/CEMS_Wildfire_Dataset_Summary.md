# CEMS Wildfire Dataset — Summary (Slide)

**One-slide overview for presentations.**

---

## CEMS Wildfire Dataset at a Glance

| | |
|---|---|
| **Name** | CEMS Wildfire Dataset (Sentinel-2 L2A + CEMS Rapid Mapping) |
| **Task** | Burned area delineation (binary) and severity grading (optional) |
| **Source** | Copernicus Emergency Management Service (CEMS) + ESA Sentinel-2 |
| **Size** | ~560 georeferenced tiles; ~275 fire events; 500+ images (train/val/test) |
| **Period** | June 2017 – April 2023 |
| **Coverage** | Europe (19 countries), mainly Mediterranean (Portugal, Spain, Italy, Greece, etc.) |

---

## Image specifics

- **Sensor**: Sentinel-2 **Level-2A** (atmospherically corrected surface reflectance).
- **Resolution**: ~10 m per pixel (for 10 m bands); some bands at 20 m or 60 m, resampled to tile grid.
- **Format**: GeoTIFF (`.tif`). Each **input image** is 12-band Float32, values in **[0, 1]**; CRS **EPSG:4326**.
- **Bands**: 12 spectral bands; for modelling we typically use **7**: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B8A (NIR narrow), **B11 (SWIR1)**, **B12 (SWIR2)**. SWIR bands are key for fire/burn detection.
- **Tile size**: Variable per scene (e.g. ~500–2000 px height/width); processed as **256×256 patches** (7-channel float32) for training.

**Labels (per tile)**  
- **DEL** (delineation): binary mask (0 = not burned, 1 = burned); expert-drawn.  
- **GRA** (grading): 5-level severity (0–4); optional, not for all tiles.  
- **CM** (cloud mask): clear / cloud / light cloud / shadow — used to filter unreliable pixels.

---

## Sample imagery (dataset layers)

*Example tile: EMSR382, AOI01, tile 01.*

| Layer | Description |
|-------|-------------|
| **Satellite (RGB preview)** | True-color composite from the multispectral image. |
| **DEL mask** | Binary burned area (red = burned). |
| **GRA mask** | Severity levels (no damage → destroyed). |
| **Cloud mask (CM)** | Clear vs cloud/shadow. |

![S2L2A true-color preview](../assets/sample/EMSR382/AOI01/EMSR382_AOI01_01/EMSR382_AOI01_01_S2L2A.png)

*Sentinel-2 L2A true-color preview (RGB) — example tile.*

![DEL mask](../assets/sample/EMSR382/AOI01/EMSR382_AOI01_01/EMSR382_AOI01_01_DEL.png)

*DEL (delineation) mask: binary burned area.*

![GRA mask](../assets/sample/EMSR382/AOI01/EMSR382_AOI01_01/EMSR382_AOI01_01_GRA.png)

*GRA (grading) mask: severity levels.*

![Cloud mask](../assets/sample/EMSR382/AOI01/EMSR382_AOI01_01/EMSR382_AOI01_01_CM.png)

*Cloud mask (CM): clear vs cloud/shadow.*

---

## Use case

- **Binary segmentation**: pixel-level “burned vs not burned” from DEL.  
- **Severity segmentation**: 5-class damage grading from GRA where available.  
- **Pipeline**: Raw GeoTIFFs → 256×256 patches (7 bands, normalized) + DEL/GRA patches; optional cloud filtering via CM; train/val/test splits and optional Catalonia-held-out set.

---

## Quick stats

| Metric | Value |
|--------|--------|
| Train / Val / Test | Pre-split (~400 / ~80 / ~80 images) |
| Patch size | 256×256 |
| Input channels | 7 (B02, B03, B04, B08, B8A, B11, B12) |
| DEL classes | 2 (binary) |
| GRA classes | 5 (severity) |

---

*For details and talking points, see [CEMS_Wildfire_Dataset_Expanded.md](CEMS_Wildfire_Dataset_Expanded.md).*
