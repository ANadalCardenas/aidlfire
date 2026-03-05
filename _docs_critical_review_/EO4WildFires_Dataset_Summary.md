# EO4WildFires Dataset — Summary (Slide)

**One-slide overview for presentations.**

---

## EO4WildFires at a Glance

| | |
|---|---|
| **Name** | EO4WildFires (Earth Observation multi-sensor, time-series benchmark) |
| **Task** | Wildfire **severity prediction** — size and shape of burned area (not ignition) |
| **Source** | EFFIS + Sentinel-2 + Sentinel-1 + NASA POWER |
| **Size** | **31,730** wildfire events; **31,740** samples (train/val/test); 45 countries |
| **Period** | 2018 – 2022 |
| **Coverage** | Global (45 countries); strong presence in Ukraine, Turkey, Algeria, Spain, Romania, Mediterranean |

---

## Image and data specifics

- **Multi-sensor data cube per event**: Each wildfire event is one **NetCDF** (or equivalent) data cube combining:
  - **Sentinel-2 (S2A)**: Multispectral imagery, **10 m** resolution (dataset resamples all S2 bands to 10 m; native S2 has 10 m / 20 m / 60 m bands). **Monthly composite** (best pixels from last 30 days before event) to reduce clouds; via Sentinel Hub (L1C TOA or similar).
  - **Sentinel-1 (S1)**: SAR (C-band), **10 m**. Most recent acquisition before event; **ascending and descending** orbits. All-weather.
  - **NASA POWER**: Meteorological parameters (temperature, precipitation, wind, soil moisture, etc.) for the **30 days before** the event at the event’s central point.
- **Labels**: **Burned area mask** from **EFFIS** (European Forest Fire Information System): vector data **rasterized** onto the Sentinel-2 grid → **binary mask** (burned / not burned).
- **Sample size in experiments**: Resized to **224×224** pixels; events with &lt;10 burned pixels (≈&lt;1 km²) can be excluded for severity-focused experiments.
- **Format on HuggingFace**: Loaded via `datasets`; keys include `S2A`, `S1`, `burned_mask`, plus metadata; use `datasets==3.6.0` (incompatible with 4.x).

---

## Sample imagery (dataset layers)

*Examples: EFFIS burned mask (left) and Sentinel-2 RGB (right) for three validation samples. Source: [how-to-use-eo4wildfires.ipynb](https://huggingface.co/datasets/AUA-Informatics-Lab/eo4wildfires/blob/main/how-to-use-eo4wildfires.ipynb).*

| Layer | Description |
|-------|-------------|
| **Left column** | Binary EFFIS burned-area mask (yellow = burned, purple = not burned). |
| **Right column** | Sentinel-2 true-color composite (bands 3:0:-1 → R,G,B), ~224×224 px. |

![EO4WildFires sample grid: masks and S2 RGB](images/eo4wildfires/sample_masks_and_s2.png)

*Three sample pairs: burned mask (left) and pre-fire S2 RGB (right). Black areas in the satellite images are NoData / outside the region of interest.*

*To generate additional figures locally, run: `pip install "datasets==3.6.0" matplotlib numpy` then `python _docs_/scripts/export_eo4wildfires_samples.py`.*

---

## Use case

- **Severity prediction**: Predict the **size and shape** of the burned area (if a fire occurs) using pre-fire conditions: S2 + S1 + meteorology.
- **Not** ignition prediction: the dataset is designed for “how bad will it be?” not “will it start?”.
- **Typical pipeline**: Load data cube → (optional) exclude empty/very-small events → train segmentation (e.g. LinkNet, U-Net) or other models; metrics: F1, IoU, Average Percentage Difference (aPD).

---

## Quick stats

| Metric | Value |
|--------|--------|
| Events | 31,730 |
| Samples (train/val/test) | 31,740 total |
| Spatial resolution | 10 m (S2, S1); meteorology at point/region |
| Patch size (paper) | 224×224 |
| Label source | EFFIS (rasterized) |
| Median fire size | 31 ha; average 128.77 ha |

---

*For details and talking points, see [EO4WildFires_Dataset_Expanded.md](EO4WildFires_Dataset_Expanded.md).*

*References: [HuggingFace](https://huggingface.co/datasets/AUA-Informatics-Lab/eo4wildfires), [Zenodo 7762564](https://zenodo.org/records/7762564), [MDPI Fire 2024](https://www.mdpi.com/2571-6255/7/11/374).*
