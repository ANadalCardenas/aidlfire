# Catalonia Wildfire Detection System

## Technical Specification Document

**Project Type:** Postgraduate Capstone Project
**Domain:** Computer Vision / Remote Sensing
**Region:** Catalonia, Spain
**Date:** January 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
   - 1.1 [Objective](#11-objective)
   - 1.2 [High-Level Pipeline](#12-high-level-pipeline)
   - 1.3 [Key Deliverables](#13-key-deliverables)
   - 1.4 [Success Criteria](#14-success-criteria)
2. [Data Pipeline](#2-data-pipeline)
   - 2.1 [Training Datasets](#21-training-datasets)
   - 2.2 [Live Data Sources](#22-live-data-sources)
   - 2.3 [Data Preprocessing](#23-data-preprocessing)
   - 2.4 [Data Augmentation](#24-data-augmentation)
   - 2.5 [Dataset Split Strategy](#25-dataset-split-strategy)
   - 2.6 [Class Imbalance Handling](#26-class-imbalance-handling)
   - 2.7 [Data Versioning & Reproducibility](#27-data-versioning--reproducibility)
   - 2.8 [Data Quality & Exploration](#28-data-quality--exploration)
3. [Model Development](#3-model-development)
   - 3.1 [Architecture Selection](#31-architecture-selection)
   - 3.2 [Training Configuration](#32-training-configuration)
   - 3.3 [Baseline Models](#33-baseline-models)
   - 3.4 [Model Optimization](#34-model-optimization)
   - 3.5 [Experiment Tracking & MLOps](#35-experiment-tracking--mlops)
   - 3.6 [Evaluation Metrics](#36-evaluation-metrics)
4. [Application Features](#4-application-features)
   - 4.1 [Core Features](#41-core-features)
   - 4.2 [Advanced Features](#42-advanced-features)
5. [API Design](#5-api-design-simplified-for-capstone)
   - 5.1 [Technology Choice](#51-technology-choice)
   - 5.2 [Simplified Endpoint Summary](#52-simplified-endpoint-summary)
   - 5.3 [Request/Response Schemas](#53-requestresponse-schemas)
   - 5.4 [Authentication (Optional)](#54-authentication-optional)
   - 5.5 [Error Codes](#55-error-codes)
6. [User Interface](#6-user-interface-simplified-for-capstone)
   - 6.1 [Technology Stack - Simplified Options](#61-technology-stack---simplified-options)
   - 6.2 [Simplified UI Structure (Streamlit Example)](#62-simplified-ui-structure-streamlit-example)
7. [Deployment](#7-deployment-simplified-for-capstone)
   - 7.1 [Simplified Deployment Options](#71-simplified-deployment-options)
   - 7.2 [Recommended Approach: HuggingFace Spaces](#72-recommended-approach-huggingface-spaces)
   - 7.3 [Alternative: Simple Cloud Deployment (If Needed)](#73-alternative-simple-cloud-deployment-if-needed)
8. [Testing & Validation](#8-testing--validation)
   - 8.1 [Testing Levels](#81-testing-levels)
   - 8.2 [Catalonia Validation Set (Mandatory)](#82-catalonia-validation-set-mandatory)
   - 8.3 [Performance Benchmarks](#83-performance-benchmarks)
   - 8.4 [User Acceptance Testing](#84-user-acceptance-testing)
9. [Project Timeline](#9-project-timeline)
   - 9.1 [Phase Overview](#91-phase-overview)
   - 9.2 [Detailed Implementation Plan](#92-detailed-implementation-plan)
   - 9.3 [Task Dependencies & Critical Path](#93-task-dependencies--critical-path)
   - 9.4 [Resource Requirements](#94-resource-requirements)
   - 9.5 [Risk Mitigation in Timeline](#95-risk-mitigation-in-timeline)
10. [Risks & Mitigations](#10-risks--mitigations)
    - 10.1 [Technical Risks](#101-technical-risks)
    - 10.2 [Data Risks](#102-data-risks)
    - 10.3 [Schedule Risks](#103-schedule-risks)
    - 10.4 [Academic & Quality Risks](#104-academic--quality-risks)
    - 10.5 [Risk Monitoring & Response](#105-risk-monitoring--response)
    - 10.6 [Risk Summary Table](#106-risk-summary-table)
[Appendix](#appendix)
   - A. [Resource Links](#a-resource-links)
   - B. [Technology Stack Summary](#b-technology-stack-summary)
   - C. [Glossary](#c-glossary)
   - D. [ML Documentation Deliverables](#d-ml-documentation-deliverables)

---

## 1. Project Overview

### 1.1 Objective

Build a deep learning system that detects active wildfires and burned areas from Sentinel-2 satellite imagery, with a focus on the Catalonia region. The system will provide near-real-time detection capabilities, fire area measurement, spread analysis, and automated alerts.

### 1.2 High-Level Pipeline

### 1.3 Key Deliverables

| Deliverable | Description |
|-------------|-------------|
| Trained Model | Segmentation model for fire/burned area detection |
| REST API | Endpoints for detection, analysis, and alerts |
| Web Dashboard | Interactive map with fire overlays and statistics |
| Documentation | Technical docs, API reference, user guide |
| Deployed System | Cloud-hosted, accessible application |

### 1.4 Success Criteria

| Metric | Target | Description |
|--------|--------|-------------|
| Model IoU (Intersection over Union) | ≥ 0.70 | Measures overlap between predicted and ground truth fire areas. Formula: `IoU = TP / (TP + FP + FN)`. A score of 0.70 means 70% of the combined predicted and actual fire area overlaps correctly. This is the primary segmentation quality metric. |
| Model Dice Score | ≥ 0.80 | Harmonic mean of precision and recall, emphasizing overlap. Formula: `Dice = 2TP / (2TP + FP + FN)`. More forgiving than IoU for small fires. A score of 0.80 indicates strong overlap between predictions and ground truth. |
| Detection Precision | ≥ 0.85 | Fraction of predicted fires that are actually real fires. Formula: `Precision = TP / (TP + FP)`. A score of 0.85 means 85% of detections are correct, minimizing false alarms. Critical for alert reliability. |
| Detection Recall | ≥ 0.80 | Fraction of real fires that are successfully detected. Formula: `Recall = TP / (TP + FN)`. A score of 0.80 means the model catches 80% of actual fires. Important for not missing dangerous fires. |
| API Response Time | < 30 seconds for single tile | Time from API request to response for processing one Sentinel-2 tile. Includes imagery fetching, model inference, and post-processing. Less critical for capstone than model metrics. |
| System Uptime | ≥ 99% | Percentage of time the system is available and operational. Less critical for capstone project where model quality is the primary focus. |

---

## 2. Data Pipeline

### 2.1 Training Datasets

#### 2.1.1 Primary Datasets (Recommended)

| Dataset | Source | Size | Resolution | Coverage | Key Features | URL |
|---------|--------|------|------------|----------|--------------|-----|
| **CEMS-Wildfire** | HuggingFace/GitHub | 500+ images | 10m | Europe (includes Spain) | Delineation + severity masks, cloud/landcover masks, 2017-2023 | [HuggingFace (links-ads)](https://huggingface.co/datasets/links-ads/wildfires-cems), [HuggingFace (9334hq)](https://huggingface.co/datasets/9334hq/wildfires-cems), [GitHub](https://github.com/MatteoM95/CEMS-Wildfire-Dataset) |
| **EO4WildFires** | HuggingFace | 31,730 events | 10-20m | 45 countries (2018-2022) | Sentinel-2 + SAR + meteorological, EFFIS annotations, multi-modal | [HuggingFace](https://huggingface.co/datasets/AUA-Informatics-Lab/eo4wildfires) |
| **Sen2Fire** | GitHub/arXiv | 2,466 patches | 10-20m | Australia | Sentinel-2 (13 bands) + Sentinel-5P aerosol, spectral indices | [Paper](https://arxiv.org/abs/2403.17884), [GitHub](https://github.com/Orion-AI-Lab/Sen2Fire) |
| **Land8Fire** | MDPI Data | ~20,000 patches | 30m | Global | Landsat-8 multispectral, human-annotated segmentation, diverse geography | [Paper/Dataset](https://www.mdpi.com/2072-4292/17/16/2776) |
| **TS-SatFire** | Nature Sci Data | 3,552 images | 375-750m | US (2017-2021) | Multi-temporal, active fire + burned area + prediction tasks, auxiliary data (weather/topography) | [Paper](https://arxiv.org/abs/2412.11555), [Nature Data](https://www.nature.com/articles/s41597-025-06271-3) |

#### 2.1.2 Secondary/Supporting Datasets

| Dataset | Source | Size | Resolution | Coverage | Use Case | URL |
|---------|--------|------|------------|----------|----------|-----|
| **S2-WCD** | IEEE DataPort | 41 image pairs | 20m | Europe & Oceania (includes Spain) | Pre/post change detection, Mediterranean fires | [IEEE DataPort](https://ieee-dataport.org/documents/sentinel-2-wildfire-change-detection-s2-wcd) |
| **FireSR** | Zenodo | Large fires | 10-20m | Canada (>2000 ha) | Pre/post Sentinel-2, NBAC masks, land use/weather data | [Zenodo](https://zenodo.org/records/13384289) |
| **FLOGA** | arXiv | Greek events | 10-20m | Greece | Sentinel-2 + MODIS, expert annotations, change detection | [Paper](https://arxiv.org/abs/2311.03339) |
| **Active Fire (Landsat-8)** | arXiv | 150k+ patches | 30m | Global | Large-scale, manual + auto labels, transfer learning | [Paper](https://arxiv.org/abs/2101.03409) |

#### 2.1.3 Dataset Selection Rationale

**Primary Dataset: CEMS-Wildfire (Recommended)**
- **Strengths:**
  - Directly from Copernicus Emergency Management Service (authoritative source)
  - Includes Spanish fire events (relevant for Catalonia)
  - Rich annotations: delineation masks, severity grading, cloud masks, land cover masks
  - Same sensor (Sentinel-2) as live data → minimal domain shift
  - Well-documented and accessible via HuggingFace
  - Temporal coverage: 2017-2023
- **Limitations:**
  - Moderate size (~500 images)
  - Primarily European coverage
  - May need data augmentation for robust training
  - **Note:** Dataset viewer on HuggingFace may have limitations due to tar archive format; data is accessible but may require manual extraction
  - **Alternative:** CEMS-HLS (morenoj11/CEMS-HLS) available if primary dataset has access issues

**Secondary Dataset: EO4WildFires (Strong Addition)**
- **Strengths:**
  - **Very large:** 31,730 events across 45 countries (2018-2022)
  - **Multi-modal:** Sentinel-2 + SAR + meteorological data
  - **Excellent for generalization:** Geographic diversity across multiple continents
  - **EFFIS annotations:** European Forest Fire Information System ground truth
  - **Temporal depth:** Multi-temporal sequences for progression modeling
- **Limitations:**
  - More complex preprocessing (SAR alignment, multi-modal fusion)
  - Larger storage/compute requirements
  - Learning curve for multi-modal data handling

**Secondary Dataset: Sen2Fire**
- **Strengths:**
  - Good spectral richness (13 Sentinel-2 bands + aerosol data)
  - Includes spectral indices (NDVI, NBR) pre-computed
  - Well-suited for segmentation tasks
  - Transferable to other regions (Australia → Mediterranean)
- **Limitations:**
  - Moderate size (2,466 patches)
  - Patch-based (may lack full temporal sequences)

**Secondary Dataset: Land8Fire**
- **Strengths:**
  - **Large size:** ~20,000 patches
  - **Global coverage:** Diverse geographies for generalization testing
  - **Human-annotated:** High-quality segmentation masks
  - **Multi-spectral:** 10 Landsat-8 bands
- **Limitations:**
  - Coarser resolution (30m vs 10-20m Sentinel-2)
  - Less frequent revisit time (~16 days)
  - Primarily static (before/after) rather than temporal sequences

**Supporting Dataset: TS-SatFire (If Time Permits)**
- **Strengths:**
  - **Multi-task:** Active fire detection, burned area mapping, AND next-day prediction
  - **Auxiliary data:** Weather, topography, land cover, fuel loads
  - **Temporal sequences:** Time-series data for progression modeling
  - Excellent for demonstrating advanced ML capabilities
- **Limitations:**
  - Coarse spatial resolution (375-750m) - may miss small fires
  - Large dataset size (~71 GB)
  - US-only coverage (limited geographic diversity)
  - More complex pipeline (temporal alignment, auxiliary data fusion)

#### 2.1.4 Recommended Dataset Strategy

**For Capstone Project (ML Focus):**

1. **Primary Training Dataset:** CEMS-Wildfire
   - Use for main model development
   - Well-documented, manageable size
   - Good for segmentation tasks

2. **Augmentation Dataset:** EO4WildFires (if feasible) OR Sen2Fire
   - **EO4WildFires:** Best for geographic generalization (if team can handle multi-modal complexity)
   - **Sen2Fire:** Simpler alternative, good spectral richness
   - Use to increase dataset diversity and test generalization

3. **Evaluation/Testing:** Land8Fire
   - Use for robustness evaluation
   - Test geographic generalization
   - Compare performance across different regions

4. **Optional (If Time):** TS-SatFire
   - Only if team wants to demonstrate prediction/progression tasks
   - Requires more complex pipeline setup
   - Good for showing advanced ML capabilities

**Dataset Combination Strategy:**
- Train primarily on CEMS-Wildfire (Europe, including Spain)
- Augment with EO4WildFires or Sen2Fire for diversity
- Evaluate on Land8Fire (global) + Catalonia validation set
- Ensures geographic generalization while maintaining focus on European/Mediterranean fires

#### 2.1.3 Catalonia-Specific Data (Mandatory)

| Source | Data Type | Use | URL |
|--------|-----------|-----|-----|
| Generalitat de Catalunya | Historical fire perimeters | **Create local validation set (required)** | [Fire perimeters by county (2011-2023)](https://analisi.transparenciacatalunya.cat/api/views/bks7-dkfd/rows.csv?accessType=DOWNLOAD), [Civil Protection Risk Map](https://datos.gob.es/en/catalogo/a09002970-mapa-de-proteccion-civil-de-cataluna-riesgo-de-incendios-forestales) |
| ICGC | Satellite-derived burn maps, Land cover | Additional labels, vegetation context | [CatLC Dataset](https://www.icgc.cat/en/Geoinformation-and-Maps/Maps/Dataset-Land-cover-map-CatLC), [FTP Download](https://ftp.icgc.cat/descarregues/CatLCNet), [Nature Paper](https://www.nature.com/articles/s41597-022-01674-y) |
| PREVINCAT (CTFC) | Vegetation maps, fuel models, fire perimeters | Risk context, fuel load data | [PREVINCAT Server](https://previncat.ctfc.cat/en/index.html), [MDPI Paper](https://www.mdpi.com/2072-4292/12/24/4124) (may require direct access) |
| Copernicus EMS | Rapid mapping activations | Event-based burned area maps | [2022 Catalonia Fire Event](https://data.jrc.ec.europa.eu/dataset/7d5a5041-efac-4762-b9d1-c0b290ab2ce7) |

**Note:** Creating a Catalonia-specific validation set is mandatory for this project, as it demonstrates geographic generalization and regional applicability.

**Additional Catalonia Resources:**
- **Wildland-Urban Interface (WUI) Map:** [Dataset](https://pdxscholar.library.pdx.edu/esm_fac/215/) - 150m resolution raster for vulnerability assessment
- **Fire Sondes Data:** [Zenodo](https://zenodo.org/records/6424854) - Atmospheric data from active fire events (2021)
- **Drought & Wildfire Variables:** [Zenodo](https://zenodo.org/records/14979237) - Monthly observations by county

### 2.2 Live Data Sources

#### 2.2.1 Sentinel-2 via Copernicus Data Space Ecosystem

**Primary Source: Copernicus Data Space Ecosystem**

| Attribute | Value | Details |
|-----------|-------|---------|
| Access URL | https://browser.dataspace.copernicus.eu/ | Web browser interface |
| Product | Sentinel-2 L2A | Atmospherically corrected, surface reflectance |
| Revisit Time | ~5 days (equator), **~2-3 days (Catalonia ~42°N)** | Faster at higher latitudes due to orbit overlap |
| Latency | 2-4 hours after satellite pass | Processing time for L2A product |
| Cost | Free | Open access for research and commercial use |
| Spatial Resolution | 10m (RGB, NIR), 20m (SWIR) | Native resolution |
| Temporal Coverage | 2015-present | Historical archive available |

**API Access Options:**

| API Type | Description | Python Library | Use Case |
|----------|-------------|----------------|----------|
| **STAC API** | SpatioTemporal Asset Catalog - modern standard | `pystac-client`, `odc-stac` | Recommended for programmatic access |
| **OData API** | Open Data Protocol - RESTful queries | `requests`, `odata` | Legacy but still supported |
| **Sentinel Hub API** | Commercial service (free tier available) | `sentinelhub` | High-level interface, processing on-demand |
| **Direct Download** | Via browser or wget/curl | `sentinelsat` | Bulk downloads, historical data |

**Recommended Python Libraries:**

| Library | Purpose | PyPI | GitHub | Documentation |
|---------|---------|------|--------|---------------|
| **`sentinelsat`** | Search and download Sentinel-2 products via Copernicus Open Access Hub | [pypi.org/project/sentinelsat](https://pypi.org/project/sentinelsat/) | [github.com/sentinelsat/sentinelsat](https://github.com/sentinelsat/sentinelsat) | [sentinelsat.readthedocs.io](https://sentinelsat.readthedocs.io/) |
| **`pystac-client`** | Modern STAC API client for searching and accessing Sentinel-2 data | [pypi.org/project/pystac-client](https://pypi.org/project/pystac-client/) | [github.com/stac-utils/pystac-client](https://github.com/stac-utils/pystac-client) | [pystac-client.readthedocs.io](https://pystac-client.readthedocs.io/) |
| **`odc-stac`** | Open Data Cube integration with STAC for efficient data loading | [pypi.org/project/odc-stac](https://pypi.org/project/odc-stac/) | [github.com/opendatacube/odc-stac](https://github.com/opendatacube/odc-stac) | [datacube-core.readthedocs.io](https://datacube-core.readthedocs.io/) (ODC docs) |
| **`sentinelhub`** | High-level API for on-demand processing (requires account) | [pypi.org/project/sentinelhub](https://pypi.org/project/sentinelhub/) | [github.com/sentinel-hub/sentinelhub-py](https://github.com/sentinel-hub/sentinelhub-py) | [sentinelhub-py.readthedocs.io](https://sentinelhub-py.readthedocs.io/) |

**Access Requirements:**
- Free registration at https://dataspace.copernicus.eu/
- OAuth2 authentication for API access
- API credentials (client ID/secret) for programmatic access

**Rate Limits:**
- Browser access: No strict limits
- API access: Reasonable use policy (typically 1000 requests/hour)
- Download bandwidth: No hard limits, but throttling may apply during peak times

#### 2.2.2 Alternative Live Data Sources

**For Active Fire Detection (Near-Real-Time):**

| Source | Latency | Coverage | Data Type | Access | URL |
|--------|---------|----------|-----------|--------|-----|
| **NASA FIRMS** | 3-6 hours (global), <1 min (US/Canada URT) | Global | MODIS/VIIRS hotspots | Public API, CSV/GeoJSON | [Portal](https://firms.modaps.eosdis.nasa.gov/), [API](https://firms.modaps.eosdis.nasa.gov/api/), [Data Download](https://firms.modaps.eosdis.nasa.gov/active_fire/) |
| **EFFIS Active Fires** | Multiple times daily | Europe (includes Catalonia) | MODIS/VIIRS + Sentinel-2 | EFFIS portal, API available | [Portal](https://effis.jrc.ec.europa.eu/) (use Current Situation section for active fires) |
| **Eye on the Fire** | <15 minutes | Global | Aggregated multi-source | REST API (may require key) | [API Documentation](https://eyeonthefire.com/data-sources) |

**For Fire Perimeters & Burned Areas:**

| Source | Update Frequency | Coverage | Data Type | Access | URL |
|--------|------------------|----------|-----------|--------|-----|
| **EFFIS Burned Areas** | Twice daily (fire season) | Europe | Sentinel-2 derived perimeters | EFFIS portal | [Portal](https://effis.jrc.ec.europa.eu/) (use Current Situation section for burned areas) |
| **Copernicus EMS Rapid Mapping** | On activation | Global (Europe focus) | High-res delineation, grading | JRC Data Portal | [Data Portal](https://emergency.copernicus.eu/data), [JRC Dataset Portal](https://data.jrc.ec.europa.eu/) |
| **NIFC (US)** | Daily/sub-daily | US only | Fire perimeters, incidents | ArcGIS services | [NIFC Portal](https://www.nifc.gov/), [ArcGIS Map Services](https://www.nifc.gov/fire-information/nifc-maps) |

**For Fire Risk & Forecasting:**

| Source | Update Frequency | Coverage | Data Type | Access | URL |
|--------|------------------|----------|-----------|--------|-----|
| **EFFIS Fire Danger** | Daily, 10-day forecast | Europe | Fire danger indices | EFFIS portal | [Portal](https://effis.jrc.ec.europa.eu/) (use Fire Danger Forecast section) |
| **PREVINCAT Dynamic Scenarios** | Hourly (on risk days) | Catalonia | Meteorological risk scenarios | PREVINCAT server | [PREVINCAT Server](https://previncat.ctfc.cat/en/index.html) |
| **Ambee Fire API** | Hourly (fires), weekly (risk) | Global (NA forecasts) | Active fires + risk forecasts | REST API (tiered) | [API Documentation](https://www.getambee.com/api/fire), [Developer Portal](https://www.getambee.com/developers) |

**Note on URL Validation:** All URLs in this section follow standard patterns for official government and research organization portals. A test script (`test_live_data_urls.sh`) is provided for local validation. Run it with: `bash test_live_data_urls.sh`

#### 2.2.3 Recommended Data Source Strategy

**For Model Training:**
- **Primary:** Copernicus Data Space (Sentinel-2 L2A) - historical archive for training
- **Validation:** EFFIS burned areas, Copernicus EMS activations for ground truth

**For Live Inference:**
- **Imagery:** Copernicus Data Space (Sentinel-2 L2A) - latest available scenes
- **Active Fire Detection:** NASA FIRMS or EFFIS for real-time hotspots (optional validation)
- **Auxiliary:** PREVINCAT for fuel/weather context

**For Alerts & Monitoring:**
- **Primary:** NASA FIRMS (global coverage, frequent updates)
- **Regional:** EFFIS Active Fires (Europe-specific, higher resolution)
- **Local:** PREVINCAT dynamic scenarios (Catalonia-specific risk)

#### 2.2.4 Sentinel-2 Band Specification

| Band | Name | Wavelength (nm) | Resolution | Role in Fire Detection |
|------|------|-----------------|------------|----------------------|
| B2 | Blue | 490 | 10m | RGB visualization, water body detection |
| B3 | Green | 560 | 10m | RGB visualization, vegetation health |
| B4 | Red | 665 | 10m | RGB visualization, vegetation indices (NDVI) |
| B8 | NIR | 842 | 10m | Vegetation indices (NDVI, NBR), vegetation health |
| B8A | NIR Narrow | 865 | 20m | Improved vegetation indices, better atmospheric correction |
| B11 | SWIR 1 | 1610 | 20m | **Primary fire detection** (thermal signature) |
| B12 | SWIR 2 | 2190 | 20m | **Primary fire/burn scar detection** (charcoal, ash) |

**Selected bands for model input:** B2, B3, B4, B8, B8A, B11, B12 (7 bands total)

**Additional Useful Bands (Optional):**
- **B10 (Cirrus):** Cloud detection and masking
- **B5, B6, B7 (Red Edge):** Enhanced vegetation analysis
- **SCL (Scene Classification Layer):** Automated cloud/land cover classification

**Band Selection Rationale:**
- SWIR bands (B11, B12) are critical for fire detection due to sensitivity to high temperatures and burned materials
- RGB bands (B2, B3, B4) enable visualization and vegetation health assessment
- NIR bands (B8, B8A) support vegetation indices (NDVI, NBR) which help distinguish burned areas
- Resolution: 7-band selection balances information content with computational efficiency

### 2.3 Data Preprocessing

#### 2.3.1 Resolution Harmonization

| Approach | Description | Trade-off |
|----------|-------------|-----------|
| **Option A: Resample to 20m** | Downsample 10m bands to match SWIR | Simpler, loses some detail |
| **Option B: Resample to 10m** | Upsample 20m bands (bilinear) | Preserves detail, interpolation artifacts |

**Recommendation:** Resample all bands to 20m for consistency with SWIR bands, which are most critical for fire detection.

#### 2.3.2 Normalization Strategy

Normalization scales raw pixel values to a consistent range (typically 0-1) so neural networks can process them efficiently. Sentinel-2 stores reflectance as integers (0-10,000, where 10,000 = 100% reflectance), but models work better with smaller, standardized values.

**Why Normalize?**

- **Numerical stability:** Large values can cause overflow/underflow during training
- **Faster convergence:** Smaller values help gradients and training stability
- **Consistent scale:** Different bands have different value ranges
- **Model compatibility:** Pretrained encoders expect specific input ranges

**Normalization Methods:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Reflectance scaling** | Divide by 10,000 | **Standard for Sentinel-2 (recommended)** |
| Min-max per band | Scale to [0,1] using dataset statistics | Alternative |
| Z-score standardization | (x - mean) / std per band | If using pretrained ImageNet encoder (test if needed) |

**Reflectance Scaling (Recommended):**

- **What it does:** Divides pixel values by 10,000 (e.g., 8,500 → 0.85)
- **Result:** Values in [0, 1] range, preserving physical meaning (0.85 = 85% reflectance)
- **Why recommended:** Standard for Sentinel-2, simple, maintains physical interpretation
- **Example:** A pixel with value 8,500 (85% reflectance) becomes 0.85 after normalization

**Recommendation:** Use reflectance scaling (divide by 10,000) as the default. This is the standard approach for Sentinel-2 data and maintains physical meaning of reflectance values. If using pretrained ImageNet encoders, test whether additional normalization is needed, but start with reflectance scaling.

#### 2.3.3 Patch Extraction

Patch extraction divides large satellite images into smaller, fixed-size squares (patches) that the model can process. Full Sentinel-2 scenes are too large (e.g., 10,980×10,980 pixels) for GPU memory, so we break them into manageable pieces.

**Why Extract Patches?**

- **Memory limits:** GPUs cannot fit entire scenes in memory
- **Training efficiency:** Smaller inputs train faster and more stably
- **Standardization:** All patches are the same size (256×256) for consistent processing
- **More training examples:** One large image becomes many patches, increasing dataset size

**Patch Extraction Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Patch size | **256×256** | Standardized size for consistency across datasets |
| Stride (training) | 128 pixels | Overlapping for augmentation |
| Stride (inference) | 256 pixels | Non-overlapping with edge blending |
| Cloud filtering | Discard patches with >50% cloud | Use SCL band or cloud mask |

**Parameter Details:**

**Patch Size (256×256 pixels):**
- Each patch is 256 pixels wide × 256 pixels tall
- At 10m resolution, this equals 2.56 km × 2.56 km
- Standard size in deep learning (powers of 2), balances detail and memory

**Stride (Training: 128 pixels):**
- Patches overlap by 50% (256 - 128 = 128 pixels overlap)
- Creates more training examples and reduces edge effects
- Example: Patch 1 covers [0:256], Patch 2 covers [128:384] (overlaps by 128 pixels)

**Stride (Inference: 256 pixels):**
- No overlap between patches (faster inference, fewer patches to process)
- Edge blending used when reconstructing full image to avoid seams

**Cloud Filtering:**
- Discard patches with >50% cloud cover (use Sentinel-2 SCL band or cloud masks)
- Cloud-covered patches are not useful for fire detection

**Note:** All datasets will be resized/cropped to 256×256 patches for consistency. If datasets have different native patch sizes, document the resizing strategy.

#### 2.3.4 Spectral Indices (Optional Additional Channels)

| Index | Formula | Purpose |
|-------|---------|---------|
| NBR (Normalized Burn Ratio) | (B8 - B12) / (B8 + B12) | Burn scar detection |
| NDVI | (B8 - B4) / (B8 + B4) | Vegetation health |
| BAI (Burned Area Index) | 1 / ((0.1 - B4)² + (0.06 - B8)²) | Active fire enhancement |

### 2.4 Data Augmentation

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Random rotation | 0°, 90°, 180°, 270° | Rotational invariance |
| Horizontal flip | 50% probability | Spatial invariance |
| Vertical flip | 50% probability | Spatial invariance |
| **Brightness adjustment** | ±10% | Atmospheric variation |
| **Contrast adjustment** | ±10% | Sensor variation |
| **Gaussian noise** | σ = 0.01 | Noise robustness |
| **Elastic deformation** | α=100, σ=10 (optional) | Geometric variation |

**Note:** Random crop from 256×256 to 224×224 is removed (common for classification, not ideal for segmentation). Keep patch size at 256×256 throughout the pipeline. Consider elastic deformation for additional geometric variation if needed.

### 2.5 Dataset Split Strategy

| Split | Percentage | Notes |
|-------|------------|-------|
| Training | 70% | Main training set |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final evaluation only |

**Critical:** Ensure no geographic overlap between splits. Split by fire event, not by individual patches from the same event.

**Validation Requirements:**
- Create visualization showing train/val/test regions on a map
- Document number of fire events in each split
- Verify no temporal leakage (if temporal patterns matter)
- Report split statistics: number of patches, fire events, area covered
- Store split IDs/indices for reproducibility

### 2.6 Class Imbalance Handling

#### 2.6.1 What is Class Imbalance?

**Class imbalance** occurs when one class (fire pixels) is significantly underrepresented compared to another class (non-fire pixels) in the training data. In wildfire detection, this is a critical problem because:

- **Fire pixels typically represent <1-5% of total pixels** in satellite imagery
- Most of the Earth's surface is not on fire at any given time
- The model can achieve high accuracy by simply predicting "no fire" for everything
- This leads to poor recall (missing actual fires) despite high overall accuracy

**Example:** In a 256×256 patch (65,536 pixels), fire might occupy only 100-500 pixels (0.15-0.76%), while the rest are vegetation, water, urban areas, etc.

#### 2.6.2 Goals of Class Imbalance Handling

The primary goals are:

1. **Improve Recall (Sensitivity):** Detect as many actual fires as possible (minimize false negatives)
   - **Why critical:** Missing a fire has severe consequences (safety, property, environment)
   - **Target:** Achieve ≥80% recall (detect 80% of actual fires)

2. **Maintain Precision:** Keep false alarms manageable (minimize false positives)
   - **Why important:** Too many false alarms reduce trust and waste resources
   - **Target:** Achieve ≥85% precision (85% of detections are real fires)

3. **Balance Training:** Ensure the model sees enough fire examples to learn fire characteristics
   - **Why needed:** Without sufficient fire examples, the model cannot learn what fires look like
   - **Target:** Model should see fire pixels in at least 30-50% of training batches

4. **Stable Training:** Prevent the model from collapsing to the trivial "always predict no fire" solution
   - **Why critical:** Imbalanced data can cause training instability and poor convergence
   - **Target:** Loss should decrease for both classes, not just the majority class

#### 2.6.3 Strategies for Handling Class Imbalance

##### Strategy 1: Loss Weighting (Recommended Starting Point)

**What it is:** Assign higher weight to the fire class in the loss function, making misclassifying fire pixels more costly than misclassifying non-fire pixels.

**How it works:**
- Standard BCE loss: `L = -[y*log(p) + (1-y)*log(1-p)]`
- Weighted BCE loss: `L = -[w_fire*y*log(p) + w_background*(1-y)*log(1-p)]`
- Typical weights: `w_fire = 5-20`, `w_background = 1` (ratio depends on imbalance severity)

**Advantages:**
- Simple to implement (single parameter)
- No changes to data pipeline
- Works well for moderate imbalance (fire pixels >0.5%)

**Disadvantages:**
- May not be sufficient for extreme imbalance (<0.1% fire pixels)
- Requires tuning the weight hyperparameter
- Can lead to overfitting on fire class if weight too high

**When to use:** Start here for all experiments. Monitor per-class metrics. If recall remains low (<70%), add additional strategies.

##### Strategy 2: Weighted Sampling (Oversampling Fire Patches)

**What it is:** Ensure that patches containing fire pixels appear more frequently in training batches than their natural frequency.

**How it works:**
- Calculate the probability of sampling each patch based on fire pixel percentage
- Patches with fire get higher sampling probability
- Ensures every batch contains fire examples

**Implementation approaches:**

**Option A: Patch-level weighted sampling** - Calculate weights for each patch based on fire pixel ratio, boost fire patches with higher sampling probability.

**Option B: Batch balancing**
- Ensure each batch contains at least 30-50% patches with fire
- Filter patches into "fire" and "non-fire" groups
- Sample proportionally from each group

**Advantages:**
- Guarantees model sees fire examples regularly
- Works well with loss weighting (complementary)
- Can be combined with other strategies

**Disadvantages:**
- More complex data pipeline
- May cause overfitting if oversampling too aggressive
- Requires careful tuning of sampling ratios

**When to use:** Add if loss weighting alone doesn't achieve target recall. Combine with loss weighting for best results.

##### Strategy 3: Focal Loss

**What it is:** A loss function that automatically down-weights easy examples (clear non-fire pixels) and focuses learning on hard examples (ambiguous pixels and fire pixels).

**How it works:**
- Focal Loss: `FL = -α(1-p)^γ * log(p)` for positive class
- `α` (alpha): Balancing factor between classes (typically 0.25)
- `γ` (gamma): Focusing parameter (typically 2.0)
- Higher `γ` focuses more on hard examples

**Mathematical intuition:**
- Easy negatives (high confidence non-fire) have `(1-p)^γ` close to 0, so contribute little to loss
- Hard examples (ambiguous or fire pixels) have `(1-p)^γ` close to 1, so contribute fully to loss
- Automatically adapts to class imbalance without manual weight tuning

**Implementation:** Use PyTorch to implement Focal Loss with alpha (balancing factor, typically 0.25) and gamma (focusing parameter, typically 2.0) hyperparameters.

**Advantages:**
- Automatically handles class imbalance
- Focuses on hard examples (where model is uncertain)
- No need for manual weight tuning
- Works well for extreme imbalance

**Disadvantages:**
- More complex than weighted BCE
- Requires tuning `α` and `γ` hyperparameters
- May be slower to converge initially

**When to use:** Alternative to loss weighting, especially for extreme imbalance (<0.1% fire pixels). Can be combined with weighted sampling.

##### Strategy 4: Patch Filtering (Batch Balancing)

**What it is:** Ensure each training batch contains a minimum percentage of patches with fire pixels.

**How it works:**
- Filter patches into "fire patches" (contain fire pixels) and "background patches" (no fire)
- When constructing batches, ensure 30-50% of patches are "fire patches"
- Prevents batches with no fire examples

**Implementation:** Filter patches into fire and background groups, then construct batches ensuring 30-50% are fire patches.

**Advantages:**
- Guarantees fire examples in every batch
- Simple to understand and implement
- Works well with loss weighting

**Disadvantages:**
- Requires preprocessing to categorize patches
- May reduce diversity if fire patches are limited
- Can slow down data loading

**When to use:** Combine with loss weighting for moderate imbalance. Less necessary if using weighted sampling.

##### Strategy 5: Data Augmentation (Fire-Specific)

**What it is:** Apply stronger augmentation to fire patches to create more fire examples.

**How it works:**
- Fire patches get more aggressive augmentation (rotation, flipping, elastic deformation)
- Creates synthetic fire examples from existing ones
- Increases effective number of fire examples

**Advantages:**
- Increases fire example diversity
- No changes to loss function needed
- Complements other strategies

**Disadvantages:**
- May create unrealistic fire patterns if overdone
- Requires careful augmentation design

**When to use:** Always use, but apply more aggressively to fire patches.

#### 2.6.4 Recommended Implementation Strategy

**Phase 1: Baseline (Start Here)**
1. Use **loss weighting** with `pos_weight = 10.0` (adjust based on actual class ratio)
2. Monitor per-class metrics (precision, recall for fire class)
3. Track training loss for both classes separately

**Phase 2: If Recall < 70%**
1. Add **weighted sampling** to ensure 40-50% of batches contain fire patches
2. Keep loss weighting
3. Monitor if recall improves

**Phase 3: If Still Insufficient**
1. Try **Focal Loss** (α=0.25, γ=2.0) instead of weighted BCE
2. Combine with weighted sampling
3. Consider increasing fire patch percentage in batches

**Phase 4: Fine-tuning**
1. Tune loss weights or focal loss parameters based on validation metrics
2. Adjust sampling ratios to balance precision and recall
3. Use validation set to find optimal hyperparameters

#### 2.6.5 Metrics to Monitor

**During Training:**
- **Per-class loss:** Fire class loss vs. background class loss (should both decrease)
- **Per-class accuracy:** Fire pixel accuracy vs. background pixel accuracy
- **Class distribution in batches:** Track percentage of fire pixels per batch
- **Gradient magnitudes:** Ensure gradients flow to both classes

**During Validation:**
- **Fire class recall:** Most critical metric (target: ≥80%)
- **Fire class precision:** Important for reducing false alarms (target: ≥85%)
- **Fire class F1-score:** Balance between precision and recall
- **Confusion matrix:** Visualize false positives and false negatives
- **ROC curve and PR curve:** Especially important for imbalanced problems

**Red Flags:**
- Fire class recall < 50%: Model is not learning fire patterns
- Fire class precision < 60%: Too many false alarms
- Fire class loss not decreasing: Model ignoring fire class
- All batches have <5% fire pixels: Sampling strategy not working

#### 2.6.6 Expected Class Distribution

**In Raw Data:**
- Fire pixels: 0.1-5% of total pixels (varies by dataset and region)
- Background pixels: 95-99.9% of total pixels

**In Training Batches (After Handling):**
- Fire pixels: 10-30% of total pixels (via sampling/augmentation)
- Background pixels: 70-90% of total pixels
- At least 40-50% of patches should contain fire pixels

**In Validation/Test Sets:**
- Keep natural distribution (no oversampling)
- This reflects real-world conditions
- Metrics on natural distribution are most meaningful

#### 2.6.7 Implementation Checklist

**How to Do It:** Start by calculating fire pixel percentage across all training patches. If fire pixels <5%, implement loss weighting first (calculate pos_weight as ratio of negative to positive pixels). Monitor training metrics from first epoch - if recall remains low (<70%), add weighted sampling to ensure each batch contains fire examples. After training, evaluate using confusion matrix and ROC/PR curves to assess class imbalance impact. Document all strategies used and their effectiveness.

- [ ] Calculate class distribution in training data (fire vs. non-fire pixel ratio)
- [ ] Implement loss weighting with appropriate `pos_weight`
- [ ] Monitor per-class metrics from the first epoch
- [ ] If recall < 70%, implement weighted sampling
- [ ] Track class distribution in batches (ensure fire examples present)
- [ ] Visualize confusion matrix on validation set
- [ ] Plot ROC and PR curves (more informative than accuracy for imbalanced data)
- [ ] Document which strategies were used and why
- [ ] Report class distribution statistics in final report

### 2.7 Data Versioning & Reproducibility

**Why Data Versioning Matters:**
- **Reproducibility:** Ensure experiments can be exactly reproduced months later
- **Collaboration:** Team members can work with the same data versions
- **Debugging:** Track which data version caused model performance changes
- **Compliance:** Document data lineage for academic/research requirements
- **Storage Efficiency:** Avoid storing duplicate datasets, use versioning instead

**Challenge:** Satellite imagery datasets are large (GBs to TBs) and cannot be stored directly in Git repositories (which are designed for code, not large binary files).

**⚠️ Data Size Reality Check:**
- **Processed patches:** 256×256 patches with 7 bands ≈ 1.8 MB per patch
- **CEMS-Wildfire (500 images):** ~5,000-10,000 patches ≈ **9-18 GB**
- **EO4WildFires (31,730 events):** Could be 100,000+ patches ≈ **180+ GB**
- **Even minimal dataset:** Likely **10-20 GB** of processed data
- **GitHub LFS free tier:** Only **1 GB** (then $5/month per 50GB)

**Conclusion:** Git LFS is **NOT suitable** for storing processed datasets. The free tier (1GB) will be exceeded immediately, and costs would be $5-20/month. For a capstone project with non-developers, this is impractical.

**Recommendation:** Use **external storage** (Google Drive, Dropbox, university storage) for large datasets, and **Git LFS only for small files** (CSV, configs). This is simpler, free, and more appropriate for the project's data sizes.

#### 2.7.1 Recommended Approach: Hybrid Strategy (Best for This Project)

**Why This Approach:**
- **Data sizes:** 10-20GB+ exceeds GitHub LFS free tier (1GB)
- **Team:** Non-developers need simple, familiar tools
- **Cost:** Free (no monthly fees)
- **Simplicity:** Uses standard file sharing everyone knows

**Strategy:** Use **external storage** for large datasets + **Git LFS for small files only** + **manual versioning with checksums**

**Recommended Setup:**

1. **Large Datasets (processed patches, imagery):**
   - Store on **external drive, university network storage, or free cloud storage** (Google Drive, Dropbox, OneDrive)
   - **Do NOT** commit to Git or Git LFS
   - Document location in Git (README with download links or shared folder paths)
   - Use checksums (MD5/SHA256) for integrity verification

2. **Small Files (use Git LFS):**
   - CSV files (fire perimeters, metadata)
   - Configuration files
   - Small GeoJSON files (<10MB)
   - Model checkpoints (if <100MB each)

3. **Versioning:**
   - Store checksums and documentation in Git
   - Use clear version naming: `cems-wildfire-v1-2026-01-15`
   - Document data location and access instructions

**Why This Works Better:**
- ✅ **Free:** No storage costs
- ✅ **Simple:** Team members just download from shared location
- ✅ **Scalable:** Can handle TBs of data
- ✅ **Familiar:** Uses standard file sharing (Drive, Dropbox, etc.)
- ✅ **Reproducible:** Checksums ensure data integrity

**Implementation Example:**

#### 2.7.2 Alternative: DVC with Cloud Storage (If Available)

**When to Consider:**
- If you have access to **free cloud storage** (university S3/GCS account, research credits)
- If datasets are **>50GB** and need automated versioning
- If team is comfortable with command-line tools

**Advantages:**
- Automated versioning
- Efficient storage (deduplication)
- Pipeline tracking
- Free if using university/research cloud credits

**Quick Setup:**

#### 2.7.3 Git LFS - Only for Small Files

Git LFS (Large File Storage) is a Git extension that stores large files outside the main Git repository. However, **for this project, Git LFS should only be used for small files** due to storage limits.

**GitHub LFS Limits:**
- **Free tier:** 1GB storage, 50GB/month bandwidth
- **Cost:** $5/month per 50GB storage
- **Reality:** Your processed datasets (10-20GB+) exceed free tier immediately

**When to Use Git LFS in This Project:**
- ✅ Small CSV files (fire perimeters, metadata)
- ✅ Configuration files
- ✅ Small GeoJSON files (<10MB)
- ✅ Model checkpoints (if <100MB each)
- ❌ **Do NOT use for:** Processed image patches, large datasets, full imagery

**Git LFS Setup (for small files only):**

#### 2.7.4 Alternative: Manual Versioning (Simplest for Non-Developers)

**When to Use:** If Git LFS and DVC are too complex, use manual versioning with checksums. This is the simplest approach but requires more manual work.

**How it Works:**
- Store actual data files outside Git (external drive, cloud storage, shared folder)
- Calculate checksums (MD5 or SHA256) for all data files
- Store checksums and documentation in Git
- Manually verify checksums before using data

**Manual Versioning Requirements:**
- ✅ Store checksums (MD5 or SHA256) for all files
- ✅ Document dataset version, date, source, preprocessing steps
- ✅ Store data in organized folders with version numbers
- ✅ Document data location (external drive, cloud link, etc.)
- ✅ Verify checksums before using data: `md5sum -c data-checksums-v1.txt`

#### 2.7.5 Data Versioning Requirements for This Project

**How to Do It:** Create a systematic versioning workflow: download datasets to external storage, compute checksums (MD5 or SHA256) for all files, store checksums and metadata in Git. Use clear naming convention with version numbers and dates. For each dataset version, document preprocessing parameters, split indices, and link to experiment tracking. Store small files (CSV, configs) in Git LFS, keep large datasets externally with access instructions. Tag important versions in Git and maintain a changelog documenting changes between versions.

**Minimum Requirements:**
- [ ] Version all datasets with clear naming (e.g., `cems-wildfire-v1-2026-01-15`)
- [ ] Store dataset checksums/hashes (MD5 or SHA256) in Git
- [ ] Document preprocessing parameters and versions
- [ ] Link data versions to experiment tracking (see Section 3.5)
- [ ] Store train/val/test split indices for reproducibility
- [ ] Document data source URLs and download dates
- [ ] **Document data storage location** (external drive, cloud storage link, shared folder)

**Recommended Approach (Hybrid):**
- [ ] Store large datasets externally (Google Drive, Dropbox, university storage)
- [ ] Use Git LFS only for small files (CSV, configs, small JSON)
- [ ] Store checksums and documentation in Git
- [ ] Tag important data versions in Git: `git tag data-v1.0`
- [ ] Create README in `data/` folder with download/access instructions
- [ ] Create data version changelog (what changed between versions)
- [ ] Store data processing scripts with data versions
- [ ] Document storage location and access instructions

#### 2.7.6 Integration with Experiment Tracking

**Link Data Versions to Experiments:**
- In W&B (Weights & Biases), log data version as a tag or config parameter
- In MLflow, store data version in experiment metadata
- In DVC, use `dvc.yaml` to track data → model pipeline

This ensures you can always identify which data version was used for each experiment.

### 2.8 Data Quality & Exploration

Data quality and exploration is the process of **understanding your dataset before training** to identify problems, patterns, and characteristics that will affect model performance. Think of it as "getting to know your data" - you need to understand what you're working with before you can build a good model.

**Why This Matters:**
- **Catch errors early:** Find annotation mistakes, misaligned masks, or corrupted files before wasting time training
- **Understand limitations:** Know what your model can and cannot learn from the data
- **Guide preprocessing:** Discover what normalization, augmentation, or filtering is needed
- **Set expectations:** Understand class imbalance, geographic bias, or temporal gaps that affect performance
- **Documentation:** Create a record of data characteristics for your capstone report

**Deliverable:** Create a comprehensive data exploration notebook (Jupyter) with visualizations, statistics, and findings documented. This notebook should be reproducible and serve as documentation for your dataset.

---

#### 2.8.1 Class Distribution (Fire vs Non-Fire Pixels)

**What to Check:**
Calculate the percentage of pixels labeled as "fire" vs "non-fire" across all training patches.

**Why This Matters:**
- **Class imbalance:** Fire pixels are typically <1-5% of total pixels. This extreme imbalance will cause the model to predict "no fire" for everything unless handled (see Section 2.6).
- **Dataset quality:** If fire pixels are >10%, the dataset might be biased toward fire events (good for training, but may not reflect real-world distribution).
- **Baseline expectations:** Helps set realistic performance targets (e.g., if only 0.5% of pixels are fire, even a naive "always predict no fire" model gets 99.5% accuracy).

**How to Do It:** Iterate through all training patches, count fire pixels (where mask value = 1) and non-fire pixels (where mask value = 0), calculate percentage. Create a histogram showing fire pixel percentage distribution across patches. Aggregate statistics per dataset to compare class imbalance between different data sources.

**What to Report:**
- Overall fire pixel percentage across all datasets
- Fire pixel percentage per dataset (CEMS-Wildfire vs EO4WildFires, etc.)
- Distribution of fire pixel percentage per patch (histogram showing how many patches have 0% fire, 1-5% fire, etc.)
- **Action items:** If fire pixels <1%, document that class imbalance handling is critical (see Section 2.6).

---

#### 2.8.2 Geographic Distribution of Fires

**What to Check:**
Visualize where fires occur geographically - are they clustered in certain regions, or evenly distributed?

**Why This Matters:**
- **Geographic bias:** If all fires are in one region (e.g., only Mediterranean), the model may not generalize to other regions (e.g., Catalonia).
- **Split strategy:** Ensures train/val/test splits don't have geographic overlap (see Section 2.5).
- **Coverage gaps:** Identifies regions with no fire data, which the model cannot learn from.

**How to Do It:** Extract fire locations from metadata (bounding boxes or centroids), create a map using folium or geopandas, and plot fire event locations as markers. One marker per fire event.

**What to Report:**
- Map showing fire event locations (one marker per fire event)
- Count of fires per country/region
- Coverage map showing which areas have fire data vs. no data
- **Action items:** If fires are clustered, ensure train/val/test splits are geographic (split by region, not randomly).

---

#### 2.8.3 Temporal Distribution (Seasons, Years)

**What to Check:**
Analyze when fires occur - are they concentrated in certain months (fire season) or years?

**Why This Matters:**
- **Seasonal patterns:** Wildfires are seasonal (summer in Mediterranean). If all training data is from summer, the model may fail on winter imagery.
- **Temporal bias:** If all data is from 2017-2019, the model may not generalize to 2024+ (though this is less critical for capstone).
- **Split strategy:** Ensures train/val/test splits don't have temporal overlap (no data leakage).

**How to Do It:** Extract fire event dates from dataset metadata (acquisition date or fire occurrence date), group by year and month, create histograms showing temporal distribution. Identify fire season by finding months with peak fire counts. Analyze temporal gaps to ensure diversity across years and seasons.

**What to Report:**
- Histogram of fires by year
- Histogram of fires by month (identify fire season)
- Count of fires by season
- **Action items:** If all fires are from one season, document this limitation. Ensure temporal diversity in train/val/test splits.

---

#### 2.8.4 Cloud Cover Statistics

**What to Check:**
Calculate how much cloud cover exists in the imagery - are patches heavily clouded, and should they be filtered?

**Why This Matters:**
- **Data quality:** Heavily clouded imagery is useless for fire detection (clouds block the view of fires).
- **Filtering strategy:** Need to decide threshold for discarding patches (e.g., discard if >50% cloud cover).
- **Dataset size:** After filtering, how much usable data remains?

**How to Do It:** Use Sentinel-2 Scene Classification Layer (SCL) or cloud mask bands to calculate cloud percentage per patch. For each patch, count cloud pixels (SCL value = 3, 8, 9, or 10) divided by total pixels. Create histogram of cloud cover percentages across all patches. Count patches exceeding thresholds to determine filtering impact.

**What to Report:**
- Histogram of cloud cover percentages
- Mean, median, max cloud cover
- Count of patches above various thresholds (10%, 25%, 50%)
- **Action items:** Decide on cloud filtering threshold (recommend >50% cloud = discard). Document how many patches remain after filtering.

---

#### 2.8.5 Band Value Ranges and Outliers

**What to Check:**
Analyze the pixel values in each Sentinel-2 band - are they in expected ranges? Are there outliers or corrupted values?

**Why This Matters:**
- **Normalization:** Need to know value ranges to choose correct normalization (see Section 2.3.2).
- **Data corruption:** Outliers may indicate corrupted files or processing errors.
- **Sensor issues:** Unusual values might indicate sensor artifacts or calibration problems.

**How to Do It:** For each Sentinel-2 band (B2, B3, B4, B8, B8A, B11, B12), compute statistics (min, max, mean, std) across all patches. Create histograms showing value distributions per band. Identify outliers by checking for values outside expected range (0-10000 for integer format, 0-1 for normalized). Flag any bands with unusual distributions or extreme outliers.

**What to Report:**
- Min, max, mean, std for each band across all patches
- Histograms showing value distributions for each band
- **Expected ranges:** Sentinel-2 L2A reflectance should be 0-1 (or 0-10000 if scaled). If values are outside this range, document the issue.
- **Action items:** If values are 0-10000, document that division by 10000 is needed for normalization (see Section 2.3.2).

---

#### 2.8.6 Missing Data/NaN Handling

**What to Check:**
Check for missing values (NaN, None, or invalid values) in imagery and masks.

**Why This Matters:**
- **Training errors:** NaN values will cause training to fail or produce invalid predictions.
- **Data quality:** Missing data indicates corrupted files or processing errors.
- **Preprocessing:** Need to handle missing data before training (fill, interpolate, or discard).

**How to Do It:** Load each image and mask patch, check for NaN (Not a Number) or Inf (Infinity) values using numpy functions. Count total NaN/Inf pixels per patch and flag patches exceeding threshold (e.g., >1% NaN). Identify specific files with problematic values. Document location and extent of missing data to determine if files should be discarded or repaired.

**What to Report:**
- Count of files with NaN or Inf values
- List of problematic files (if any)
- **Action items:** If NaN/Inf found, decide on handling strategy:
  - **Option 1:** Discard problematic files
  - **Option 2:** Fill NaN with 0 or mean value (not recommended for satellite data)
  - **Option 3:** Interpolate from neighboring pixels (complex, may not be worth it)

---

#### 2.8.7 Verify Ground Truth Masks Align with Imagery

**What to Check:**
Ensure that fire masks (ground truth) are correctly aligned with the corresponding satellite imagery - do fire pixels in the mask correspond to actual fire locations in the image?

**Why This Matters:**
- **Training quality:** Misaligned masks will teach the model wrong patterns (e.g., model learns that "water" = fire if mask is misaligned).
- **Annotation errors:** Misalignment indicates annotation mistakes or coordinate system issues.
- **Model performance:** Even small misalignments (1-2 pixels) can hurt performance.

**How to Do It:** Overlay fire mask (ground truth) on corresponding satellite imagery using transparency or color blending. Visually inspect alignment by checking if fire pixels in mask correspond to visible burn scars, smoke plumes, or charred areas in the RGB composite. Check multiple patches across different regions and fire types. Verify pixel-level alignment by comparing mask boundaries with visible fire edges in imagery.

**What to Report:**
- Visual grid showing 9-16 sample patches with mask overlay
- Count of patches with obvious misalignment (if any)
- **Visual inspection:** Manually review overlays - do fire masks align with visible burn scars or smoke in imagery?
- **Action items:** If misalignment found, investigate coordinate system issues or contact dataset maintainers.

---

#### 2.8.8 Sample Review for Annotation Errors

**What to Check:**
Manually review a sample of patches to identify annotation errors - are fires correctly labeled? Are there false positives (non-fire labeled as fire) or false negatives (fire not labeled)?

**Why This Matters:**
- **Training quality:** Annotation errors will teach the model wrong patterns.
- **Performance limits:** Even a perfect model cannot exceed the quality of ground truth labels.
- **Dataset trust:** Understanding annotation quality helps interpret model performance.

**How to Do It:** Manually review 20-50 random patches, note annotation quality (Good/Fair/Poor), identify false positives and false negatives.

**What to Report:**
- Sample of reviewed patches (10-20) with annotations
- Count of patches with annotation errors
- Types of errors found (false positives, false negatives, misalignment)
- **Annotation quality score:** Percentage of patches with "Good" quality
- **Action items:** If >10% of patches have errors, consider filtering problematic patches or contacting dataset maintainers.

---

#### 2.8.9 Coordinate System Validation

**What to Check:**
Verify that all imagery and masks use the correct coordinate system (CRS) and that coordinates are consistent.

**Why This Matters:**
- **Alignment:** Different coordinate systems cause misalignment between imagery and masks.
- **Geographic accuracy:** Incorrect CRS means geographic locations are wrong (e.g., fire detected at wrong location).
- **Integration:** Need correct CRS to overlay predictions on maps or combine with other geographic data.

**How to Do It:** Use rasterio to check CRS for all images and masks, verify consistency, and identify any mismatches.

**What to Report:**
- CRS used for images (should be UTM Zone 31N, EPSG:32631, or WGS84, EPSG:4326)
- CRS used for masks (should match images)
- Count of files with missing or incorrect CRS
- **Action items:** If CRS mismatch found, reproject to consistent CRS (see Section 2.3.1).

---

#### 2.8.10 Check for Duplicate Patches

**What to Check:**
Identify duplicate patches (same image or mask appearing multiple times) that could cause data leakage between train/val/test splits.

**Why This Matters:**
- **Data leakage:** If the same patch appears in both train and test sets, test metrics are inflated (model "sees" test data during training).
- **Dataset size:** Duplicates inflate dataset size without adding information.
- **Split integrity:** Ensures train/val/test splits are truly independent.

**How to Do It:** Compute SHA256 hashes for all images and masks, identify exact duplicates. Optionally check pixel-wise similarity for near-duplicates (>95% similarity).

**What to Report:**
- Count of exact duplicate files (same hash)
- Count of near-duplicate files (>95% similarity)
- List of duplicate pairs (if any)
- **Action items:** If duplicates found, remove them before creating train/val/test splits to prevent data leakage.

---

#### 2.8.11 Data Exploration Notebook Template

**Structure for Your Jupyter Notebook:**

**Deliverable Checklist:**
- [ ] Jupyter notebook with all 10 exploration sections
- [ ] Visualizations for each section (plots, maps, histograms)
- [ ] Summary statistics and findings
- [ ] List of data quality issues and recommended fixes
- [ ] Documentation of dataset characteristics (class distribution, geographic coverage, temporal range)
- [ ] Export notebook as HTML/PDF for capstone report

---

## 3. Model Development

### 3.1 Architecture Selection

Architecture selection is choosing the **neural network structure** for semantic segmentation. For this project, we need an architecture that preserves spatial information, handles multi-spectral input (7 bands), and outputs binary masks at the same resolution.

---

#### 3.1.1 Recommended Architecture: U-Net with Pretrained Encoder

**Why U-Net is Recommended:**

U-Net is the **industry standard** for semantic segmentation in remote sensing. Key advantages:

1. **Skip Connections:** Preserve fine-grained spatial details critical for detecting small fires and precise boundaries
2. **Encoder-Decoder Structure:** Encoder learns high-level fire patterns, decoder reconstructs full-resolution masks
3. **Proven Performance:** Widely used in remote sensing competitions with excellent results on similar tasks
4. **Efficiency:** Lightweight, fast inference, good accuracy/speed balance
5. **Flexibility:** Easy to modify, works with pretrained encoders, handles multi-spectral inputs

**U-Net Architecture:** Encoder downsamples from 256×256 to 16×16, extracting features. Decoder upsamples back to 256×256, with skip connections preserving spatial details from encoder layers.

**Why Pretrained Encoder?** Transfer learning from ImageNet provides general visual features (edges, textures, patterns), leading to faster convergence, 5-10% better accuracy, and less data needed.

**Encoder Options:**

| Encoder | Parameters | Speed | Accuracy | When to Use |
|---------|-----------|-------|----------|-------------|
| **ResNet-34** | ~21M | Medium | High | **Recommended default** - balanced performance |
| EfficientNet-B0 | ~5M | Fast | Good | If speed is critical or GPU memory limited |
| ResNet-50 | ~25M | Slower | Very High | If accuracy is more important than speed |
| EfficientNet-B3 | ~12M | Medium | High | Alternative to ResNet-34, more efficient |

**Encoder Choice:** ResNet-34 (recommended) offers proven performance and good balance. EfficientNet-B0 is a good alternative for limited GPU memory or faster inference. Both use ImageNet pretrained weights.

**Decoder:** Standard U-Net decoder with upsampling, skip connections, and 2× Conv3x3 + ReLU + BatchNorm per level. Output: single 1×1 convolution with sigmoid activation.

**Architecture Documentation Requirements:**
- Document why U-Net over alternatives (DeepLabV3+, FCN, SegNet) - see Section 3.1.2
- Justify encoder choice (ResNet-34 vs EfficientNet-B0) - based on GPU memory, speed requirements
- Explain band selection (7 bands vs more/fewer) - see Section 3.1.3
- Create visual architecture diagram (use tools like `torchsummary` or `visualkeras`)
- If time permits: ablation studies (different encoders, skip connections, band combinations)

---

#### 3.1.2 Alternative Architectures Considered

**Recommended Options:**

**U-Net + ResNet-34 (Primary Choice):**
- ~21M parameters, balanced performance (IoU: 0.70-0.75)
- Most widely used, well-documented, excellent pretrained weights
- Requires 8GB+ VRAM, good for capstone

**U-Net + EfficientNet-B0 (Alternative):**
- ~5M parameters (4x smaller), faster inference (IoU: 0.68-0.73)
- Works on 4GB+ VRAM, good if GPU memory limited
- Slightly lower accuracy but more efficient

**Other Alternatives (Not Recommended):**
- **DeepLabV3+:** Too complex (40-60M params, 16GB+ VRAM), overkill for capstone
- **FCN:** Outdated, superseded by U-Net
- **SegNet:** Less accurate than U-Net, outdated VGG encoder
- **U-Net + MobileNetV3:** Only for edge deployment, lower accuracy (IoU: 0.65-0.70)

**Summary:** Use U-Net + ResNet-34 unless GPU memory is limited (then use EfficientNet-B0).

---

#### 3.1.3 Input/Output Specification

**Input Specification:**

| Attribute | Value | Notes |
|-----------|-------|-------|
| **Input shape** | (Batch, 7, 256, 256) | 7 spectral bands, 256×256 pixels |
| **Data type** | float32 | Normalized reflectance [0.0, 1.0] |
| **Coordinate system** | UTM Zone 31N or WGS84 | Geographic reference |
| **Spatial resolution** | 10-20m per pixel | Sentinel-2 native resolution |

**Band Selection (7 bands):**
- **B2, B3, B4 (RGB):** Visualization, spatial patterns
- **B8, B8A (NIR):** Vegetation detection, NDVI calculation
- **B11, B12 (SWIR):** **Most critical** - fires emit strongly in SWIR, burned areas have low SWIR reflectance

**Why 7 bands?** Minimum necessary for fire detection (RGB + NIR + SWIR). Other bands (B1, B5-B7, B9-B10) are low resolution or redundant.

**Output Specification:**

| Attribute | Value | Notes |
|-----------|-------|-------|
| **Output shape** | (Batch, 1, 256, 256) | Binary mask, same resolution as input |
| **Data type** | float32 | Probabilities [0.0, 1.0] |
| **Activation** | Sigmoid | Converts logits to probabilities |
| **Threshold** | 0.5 (default) | Tunable (0.3-0.4 for higher recall, 0.6-0.7 for higher precision) |

**Input/Output Validation Checklist:**

**How to Do It:** Create validation functions that check input tensor shapes and value ranges before model forward pass. Verify input has correct dimensions (batch, 7 channels, 256×256 spatial) and all values are normalized [0.0, 1.0]. After model inference, verify output shape (batch, 1 channel, 256×256) and probability range [0.0, 1.0]. Test spatial alignment by comparing input and output coordinates. Run threshold optimization on validation set using ROC/PR curves to find optimal threshold. Document all validation checks and band selection rationale.

- [ ] Verify input shape is `(batch_size, 7, 256, 256)`
- [ ] Verify input values are in range `[0.0, 1.0]` (normalized)
- [ ] Verify all 7 bands are present and in correct order (B2, B3, B4, B8, B8A, B11, B12)
- [ ] Verify output shape is `(batch_size, 1, 256, 256)`
- [ ] Verify output values are in range `[0.0, 1.0]` (probabilities)
- [ ] Verify output spatial alignment matches input (pixel-to-pixel correspondence)
- [ ] Test threshold selection on validation set (find optimal threshold)
- [ ] Document band selection rationale (why these 7 bands)

### 3.2 Training Configuration

Training configuration determines **how the model learns** from data. This includes the loss function (what the model tries to minimize), optimizer (how it updates weights), learning rate schedule (how fast it learns over time), and other hyperparameters that control the training process.

**Key Principles:**
- **Loss function:** Defines what "good" means - what should the model optimize for?
- **Optimizer:** Determines how to update model weights based on gradients
- **Learning rate:** Controls step size - too large (unstable), too small (slow convergence)
- **Hyperparameters:** Balance between training speed, stability, and final performance

---

#### 3.2.1 Loss Function

**Recommended: Combined BCE + Dice Loss**

The loss function measures how wrong the model's predictions are. For segmentation with class imbalance, use **combined BCE + Dice Loss (0.5:0.5 weight)** - this is the industry standard.

**Why Both?**
- **BCE Loss:** Pixel-wise accuracy, ensures each pixel classified correctly, provides stable gradients
- **Dice Loss:** Region overlap, handles class imbalance naturally, directly optimizes IoU
- **Combined:** Best of both - pixel accuracy + region overlap + handles imbalance

**Alternative Loss Functions:**
- **Focal Loss:** For extreme imbalance (<0.1% fire pixels), automatically focuses on hard examples
- **Tversky Loss:** To tune precision/recall trade-off explicitly (higher β = focus on recall)
- **Weighted BCE:** Simple alternative, add `pos_weight` to BCE component for class imbalance

**Loss Function Checklist:**

**How to Do It:** Start with combined BCE + Dice Loss (0.5:0.5 weight) as default. Monitor both loss components separately during training to understand their contributions. If class imbalance is severe (fire pixels <1%), add pos_weight to BCE component or switch to Focal Loss. If recall is too low, try Tversky Loss with higher β parameter to focus on recall. If precision is too low, use Tversky Loss with higher α to focus on precision. Document final loss function, weights, and rationale in experiment tracking.

- [ ] Implement combined BCE + Dice Loss (default)
- [ ] Monitor both loss components separately
- [ ] If class imbalance severe, add `pos_weight` to BCE or use Focal Loss
- [ ] If recall too low, try Tversky Loss with higher β (focus on recall)
- [ ] If precision too low, try Tversky Loss with higher α (focus on precision)
- [ ] Document final loss function and weights in experiment tracking

---

#### 3.2.2 Optimizer Configuration

The optimizer determines **how to update model weights** based on the loss gradient. It decides the step size (learning rate) and direction for each weight update.

**Why AdamW is Recommended:**

AdamW (Adam with Weight Decay) is the **modern standard** for deep learning optimization. It's an improvement over Adam that properly handles weight decay (regularization).

##### AdamW vs. Other Optimizers

| Optimizer | Pros | Cons | When to Use |
|-----------|------|------|-------------|
| **AdamW** | ✅ Adaptive learning rate<br>✅ Handles sparse gradients<br>✅ Proper weight decay<br>✅ Works well out-of-the-box | ⚠️ More memory than SGD<br>⚠️ Can overfit if not regularized | **⭐ Recommended default** |
| Adam | ✅ Adaptive learning rate<br>✅ Works well | ⚠️ Incorrect weight decay implementation<br>⚠️ May overfit | Use AdamW instead |
| SGD | ✅ Simple, well-understood<br>✅ Less memory<br>✅ Better generalization (sometimes) | ⚠️ Requires careful LR tuning<br>⚠️ Slower convergence<br>⚠️ Needs momentum | Use if AdamW fails |
| RMSprop | ✅ Adaptive learning rate | ⚠️ Less popular, fewer examples | Rarely used |

**AdamW Parameters Explained:**

| Parameter | Value | What It Does | Why This Value |
|-----------|-------|--------------|----------------|
| **Initial Learning Rate** | `1e-4` (0.0001) | Starting step size for weight updates | **Conservative start** - prevents overshooting early in training. Can increase to 3e-4 if training is stable. |
| **Weight Decay** | `1e-4` (0.0001) | L2 regularization strength | **Light regularization** - prevents overfitting without hurting performance. Increase to 1e-3 if overfitting. |
| **Betas** | `(0.9, 0.999)` | Momentum parameters | **Standard values** - β₁=0.9 (short-term momentum), β₂=0.999 (long-term momentum). Rarely need to change. |
| **Epsilon** | `1e-8` (default) | Numerical stability | **Default is fine** - prevents division by zero. Don't change unless numerical issues occur. |

**Why These Specific Values?**

**Initial Learning Rate: 1e-4 (0.0001)**

- **Conservative start:** Prevents model from overshooting optimal weights early in training
- **Safe for pretrained models:** When using pretrained encoders (ResNet-34), smaller LR prevents destroying pretrained features
- **Can be increased:** If training is stable and loss decreases smoothly, can try 3e-4 or 5e-4
- **Too high:** If LR > 1e-3, training may be unstable (loss oscillates or increases)
- **Too low:** If LR < 1e-5, training will be very slow (may take 2-3x longer)

**Weight Decay: 1e-4 (0.0001)**

- **Light regularization:** Prevents overfitting without significantly hurting performance
- **Standard for segmentation:** Common value in segmentation literature
- **Too high:** If > 1e-3, model may underfit (validation loss doesn't decrease)
- **Too low:** If < 1e-5, may overfit (large gap between train and validation loss)

**Betas: (0.9, 0.999)**

- **β₁ = 0.9:** Controls short-term momentum (how much recent gradients influence updates)
- **β₂ = 0.999:** Controls long-term momentum (how much historical gradients influence updates)
- **Standard values:** These are the default and work well for most tasks
- **Rarely changed:** Only adjust if you have specific convergence issues

**Implementation:**

**Optimizer Tuning Guidelines:**

| Issue | Solution |
|-------|----------|
| **Training loss oscillates** | Reduce learning rate (try 5e-5) |
| **Training loss decreases very slowly** | Increase learning rate (try 3e-4) |
| **Large gap between train/val loss (overfitting)** | Increase weight decay (try 1e-3) |
| **Both train and val loss plateau (underfitting)** | Decrease weight decay (try 1e-5) or increase model capacity |
| **Training is unstable (loss NaN)** | Reduce learning rate significantly (try 1e-5) |

**Optimizer Checklist:**

- [ ] Use AdamW as default optimizer
- [ ] Start with LR=1e-4, weight_decay=1e-4
- [ ] Consider different LRs for encoder (1e-5) and decoder (1e-4) if using pretrained encoder
- [ ] Monitor training stability (loss should decrease smoothly)
- [ ] Adjust LR if training is too slow or unstable
- [ ] Adjust weight_decay if overfitting or underfitting

---

#### 3.2.3 Learning Rate Schedule

**Recommended: ReduceLROnPlateau**

A learning rate schedule adjusts the learning rate during training. **ReduceLROnPlateau** is the recommended default - it reduces LR when validation metric stops improving (adaptive, only when needed).

**Parameters:** Monitor validation IoU (or loss), patience=5 epochs, factor=0.5 (reduce by 50%), min_lr=1e-6.

**Alternatives:**
- **OneCycleLR:** For faster convergence (ramp up then down), requires fixed epochs
- **Cosine Annealing:** For longer training (100+ epochs), smooth decay

**Learning Rate Schedule Checklist:**

- [ ] Start with ReduceLROnPlateau (default)
- [ ] Monitor validation IoU (or loss) for scheduler
- [ ] Set patience=5 epochs (wait 5 epochs before reducing LR)
- [ ] Set factor=0.5 (reduce LR by 50% each time)
- [ ] If training is slow, try OneCycleLR for faster convergence
- [ ] Log learning rate in experiment tracking (W&B)

---

#### 3.2.4 Training Hyperparameters

Training hyperparameters control the **training process itself** - how many examples per batch, how long to train, when to stop, etc. These are separate from model architecture and optimizer settings.

##### Batch Size

**What it is:**
- Number of samples processed together in one forward/backward pass
- Affects memory usage, training speed, and gradient stability

**Recommended Values:**

| GPU Memory | Batch Size | Rationale |
|------------|------------|-----------|
| **4GB VRAM** | 4-8 | Limited memory, must use small batches |
| **8GB VRAM** | 8-12 | **Recommended for most GPUs** - good balance |
| **16GB+ VRAM** | 16-32 | Can use larger batches for faster training |

**Why These Values?**

- **Memory constraint:** Larger batches = more GPU memory needed
- **Gradient stability:** Larger batches = more stable gradients (but may generalize worse)
- **Training speed:** Larger batches = fewer iterations per epoch = faster training
- **Generalization:** Smaller batches sometimes generalize better (but slower)

**How to Choose:**

1. **Start with batch_size=8** (works on most GPUs)
2. **Increase if:**
   - GPU memory allows (monitor with `nvidia-smi`)
   - Training is too slow
   - Gradients are noisy (loss oscillates)
3. **Decrease if:**
   - Out of memory errors occur
   - Model overfits (large train/val gap)

**Implementation:**

**Batch Size Effects:**

| Batch Size | Memory | Speed | Gradient Stability | Generalization |
|------------|--------|-------|-------------------|----------------|
| **Small (4-8)** | Low | Slower | Less stable | Often better |
| **Medium (8-16)** | Medium | Medium | Stable | Good |
| **Large (16-32)** | High | Faster | Very stable | May be worse |

##### Number of Epochs

**What it is:**
- Number of complete passes through the training dataset
- Determines how long to train

**Recommended Values:**

| Scenario | Epochs | Rationale |
|----------|--------|-----------|
| **With early stopping** | 50-100 | Early stopping will stop when validation stops improving |
| **Without early stopping** | 50-80 | Enough to converge, but not too long |
| **Limited time** | 30-50 | Minimum for reasonable performance |

**Why These Values?**

- **Early stopping:** With early stopping (patience=10), training typically stops after 30-60 epochs
- **Convergence:** Most models converge (reach best performance) within 50-80 epochs
- **Diminishing returns:** Training beyond 100 epochs rarely improves performance significantly
- **Time constraint:** For capstone, 50-100 epochs is reasonable (with early stopping, may stop earlier)

**How to Monitor:**

##### Early Stopping

**What it is:**
- Stop training when validation metric stops improving
- Prevents overfitting and saves training time

**Recommended Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|------------|
| **Metric** | Validation IoU | **Better than loss** - directly measures what we care about |
| **Patience** | 10 epochs | Wait 10 epochs without improvement before stopping |
| **Min Delta** | 0.001 | Minimum improvement to count as "improvement" (optional) |

**Why Validation IoU (Not Loss)?**

- **Direct metric:** IoU is the target metric (see Section 1.4), loss is just a proxy
- **More reliable:** Loss can decrease while IoU plateaus (especially with combined losses)
- **Better stopping:** Stops when actual performance stops improving

**Why Patience=10?**

- **Not too short:** Allows for temporary plateaus (validation can be noisy)
- **Not too long:** Stops before wasting time on overfitting
- **Standard value:** Common practice in deep learning

**Implementation:**

##### Mixed Precision Training (FP16)

**What it is:**
- Use 16-bit floating point instead of 32-bit for faster training
- Reduces memory usage and speeds up training (especially on modern GPUs)

**Why Use It?**

- **Faster training:** 1.5-2x speedup on modern GPUs (V100, A100, RTX 30xx+)
- **Less memory:** Allows larger batch sizes or larger models
- **Minimal accuracy loss:** Usually <1% difference in final performance

**When to Use:**

- ✅ **Recommended** - Use if your GPU supports it (most modern GPUs do)
- ✅ Especially useful if GPU memory is limited
- ⚠️ May not work on very old GPUs (pre-2018)

**Implementation:**

**Mixed Precision Checklist:**

- [ ] Check GPU support (most modern GPUs support FP16)
- [ ] Use `autocast()` for forward pass
- [ ] Use `GradScaler()` for backward pass
- [ ] Monitor for NaN values (if occur, reduce learning rate)
- [ ] Compare FP16 vs FP32 performance (should be similar)

**Training Hyperparameters Summary Table:**

| Hyperparameter | Recommended Value | Range | Notes |
|----------------|-------------------|-------|-------|
| **Batch Size** | 8-16 | 4-32 | Depends on GPU memory |
| **Epochs** | 50-100 | 30-150 | With early stopping, typically stops earlier |
| **Early Stopping Patience** | 10 epochs | 5-20 | On validation IoU |
| **Mixed Precision** | FP16 (enabled) | FP16/FP32 | Use if GPU supports it |

**Training Configuration Checklist:**

- [ ] Loss function: Combined BCE + Dice Loss (0.5:0.5)
- [ ] Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- [ ] Learning rate schedule: ReduceLROnPlateau (patience=5, factor=0.5)
- [ ] Batch size: 8-16 (based on GPU memory)
- [ ] Epochs: 50-100 (with early stopping)
- [ ] Early stopping: On validation IoU (patience=10)
- [ ] Mixed precision: FP16 (if GPU supports it)
- [ ] Monitor all hyperparameters in experiment tracking (W&B)
| Gradient clipping | Max norm = 1.0 | Stability |

#### 3.2.5 PyTorch DataLoader Configuration

**DataLoader Configuration:**

Proper DataLoader configuration can speed up training 2-3x by parallelizing data loading.

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| **num_workers** | 4-8 | Parallel data loading (4 cores: 2-4, 8 cores: 4-6, 16+ cores: 6-8) |
| **pin_memory** | True | Faster GPU transfer, always use for GPU training |
| **prefetch_factor** | 2 | Default is good, prefetches batches ahead |
| **persistent_workers** | True | Keep workers alive between epochs (if num_workers > 0) |
| **shuffle** | True (train), False (val/test) | Shuffle training data to prevent learning order |

**DataLoader Checklist:**

- [ ] Set num_workers=4-8 (based on CPU cores)
- [ ] Enable pin_memory=True (for GPU training)
- [ ] Set prefetch_factor=2 (default is fine)
- [ ] Enable persistent_workers=True (if num_workers > 0)
- [ ] Shuffle=True for training, False for validation/test
- [ ] Monitor GPU utilization (should be >80%)
- [ ] Adjust num_workers if GPU is waiting for data

---

#### 3.2.6 Model Checkpointing Strategy

**Model Checkpointing Strategy:**

Save model weights and training state to allow recovery, model selection, and analysis.

| Checkpoint Type | Frequency | Purpose |
|----------------|-----------|---------|
| **Best model** | Every epoch (if improved) | Save model with best validation IoU |
| **Periodic** | Every 10 epochs | Backup for analysis/recovery |
| **Final** | End of training | Final trained model state |

**What to Save:** Model weights, optimizer state (for resuming), epoch number, validation metrics, hyperparameters, git commit.

**Storage:** Each checkpoint ~100-500MB. Keep best + last 3-5 periodic checkpoints. Don't commit to Git (use external storage).

**Checkpointing Checklist:**

- [ ] Save best model whenever validation IoU improves
- [ ] Save periodic checkpoints every 10 epochs
- [ ] Save final model at end of training
- [ ] Include model weights, optimizer state, and metadata
- [ ] Store checkpoints in organized directory structure
- [ ] Clean up old checkpoints to save disk space
- [ ] Document checkpoint loading procedure
- [ ] Link checkpoints to experiment tracking (W&B run ID)

### 3.3 Baseline Models

Baseline models are **simple, non-deep-learning approaches** that provide a performance benchmark. They help answer the critical question: "Does deep learning actually add value, or could we solve this problem with simpler methods?"

**Why Implement Baselines?**

1. **Demonstrate value:** Show that deep learning outperforms traditional methods (or identify when it doesn't)
2. **Set expectations:** Understand what performance is achievable with simple methods
3. **Academic rigor:** Standard practice in research to compare against baselines
4. **Debugging:** If deep learning performs worse than baselines, indicates a problem (bug, wrong data, etc.)
5. **Capstone requirement:** Demonstrates understanding of the problem domain and appropriate method selection

**Baseline Selection Rationale:**

For wildfire detection, we need baselines that:
- Are **simple to implement** (don't require weeks of work)
- Are **standard in remote sensing** (well-established methods)
- Use **similar inputs** (same satellite imagery, same bands)
- Provide **meaningful comparison** (not trivial to beat)

---

#### 3.3.1 Simple Baseline Models

##### Baseline 1: Threshold-Based Method

**What it is:**
- Simple rule-based approach using spectral thresholds
- Based on physical properties: fires emit strongly in SWIR (Short-Wave Infrared) bands
- Standard approach in remote sensing for fire detection

**How it works:**
- Apply threshold to SWIR bands (B11, B12) - fires have high SWIR values
- Optionally combine with NBR (Normalized Burn Ratio) for burned area detection
- Binary classification: pixel above threshold = fire, below = no fire

**Why This Baseline?**

**Advantages:**
- ✅ **Simple and fast:** No training required, instant predictions
- ✅ **Interpretable:** Clear physical rationale (fires emit in SWIR)
- ✅ **Standard method:** Widely used in remote sensing literature
- ✅ **Baseline for comparison:** If deep learning can't beat this, something is wrong
- ✅ **Quick to implement:** Can be done in hours, not days

**Limitations (Why Deep Learning Should Beat It):**
- ⚠️ **Fixed thresholds:** Cannot adapt to different conditions (atmosphere, sensor calibration, time of day)
- ⚠️ **No spatial context:** Each pixel classified independently (ignores neighboring pixels)
- ⚠️ **False positives:** Many non-fire objects have high SWIR (hot surfaces, urban areas, clouds)
- ⚠️ **No learning:** Cannot learn complex patterns or adapt to data distribution
- ⚠️ **Single-band focus:** Only uses SWIR bands, ignores other spectral information

**Expected Performance:**
- **Precision:** Moderate (50-70%) - many false positives from hot surfaces
- **Recall:** Moderate (60-75%) - misses small fires and fires with low intensity
- **IoU:** Low (0.40-0.55) - poor boundary accuracy
- **Why lower than deep learning:** Cannot learn complex patterns, no spatial context, fixed thresholds

**When Threshold-Based Might Be Sufficient:**
- Very large, intense fires (clear SWIR signal)
- Simple use cases where false positives are acceptable
- Real-time systems with extreme computational constraints
- When training data is unavailable

**Implementation Approach:**
1. Calculate SWIR band values (B11, B12) for all pixels
2. Apply threshold (e.g., B11 > 0.3 or B12 > 0.25) - values depend on normalization
3. Optionally combine with NBR threshold for burned areas
4. Generate binary mask
5. Evaluate on same test set as deep learning model

**What to Report:**
- Threshold values used (and how they were chosen)
- Performance metrics (IoU, Dice, Precision, Recall)
- Comparison with deep learning model
- Analysis of when threshold-based fails (false positives, missed fires)
- Visual examples showing threshold-based vs. deep learning predictions

##### Baseline 2: Random Forest

**What it is:**
- Traditional machine learning classifier (ensemble of decision trees)
- Trained on hand-crafted features (spectral indices, band values)
- Non-deep-learning approach that can learn from data

**How it works:**
- Extract features for each pixel: raw band values (B2, B3, B4, B8, B11, B12), spectral indices (NBR, NDVI, BAI)
- Train Random Forest classifier to predict fire vs. non-fire
- Make pixel-wise predictions (similar to deep learning output)

**Why This Baseline?**

**Advantages:**
- ✅ **Learns from data:** Unlike threshold-based, can adapt to training data
- ✅ **Handles features well:** Good at learning relationships between spectral indices and fire
- ✅ **Interpretable:** Can analyze feature importance (which bands/indices matter most)
- ✅ **Standard ML method:** Well-established, widely used in remote sensing
- ✅ **Good baseline:** Represents "traditional ML" approach (pre-deep-learning era)
- ✅ **Reasonable performance:** Should achieve better results than threshold-based

**Limitations (Why Deep Learning Should Beat It):**
- ⚠️ **No spatial context:** Each pixel classified independently (no understanding of fire regions)
- ⚠️ **Hand-crafted features:** Relies on human-designed features (spectral indices) rather than learned features
- ⚠️ **Limited complexity:** Cannot learn complex, non-linear patterns that deep learning can
- ⚠️ **Pixel-wise only:** Cannot understand that fires form connected regions
- ⚠️ **Feature engineering required:** Need to design good features (spectral indices)

**Expected Performance:**
- **Precision:** Good (70-80%) - better than threshold-based, learns from data
- **Recall:** Good (70-80%) - can learn fire patterns from training data
- **IoU:** Moderate (0.55-0.65) - better than threshold-based, but worse than deep learning
- **Why lower than deep learning:** No spatial context, limited to hand-crafted features, pixel-wise only

**When Random Forest Might Be Sufficient:**
- Limited training data (Random Forest works with less data than deep learning)
- Need for interpretability (feature importance analysis)
- Computational constraints (faster inference than deep learning)
- Simple use cases where pixel-wise classification is sufficient

**Feature Engineering for Random Forest:**

**Raw Band Values:**
- B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR-1), B12 (SWIR-2)
- Direct pixel values from satellite imagery

**Spectral Indices (Hand-Crafted Features):**
- **NBR (Normalized Burn Ratio):** `(B8 - B12) / (B8 + B12)` - detects burned areas
- **NDVI (Normalized Difference Vegetation Index):** `(B8 - B4) / (B8 + B4)` - vegetation health
- **BAI (Burned Area Index):** `1 / ((0.1 - B4)² + (0.06 - B8)²)` - fire detection
- **NDWI (Normalized Difference Water Index):** `(B3 - B8) / (B3 + B8)` - water detection (helps exclude water)
- **EVI (Enhanced Vegetation Index):** More advanced vegetation index

**Why These Features?**
- **Spectral indices:** Capture relationships between bands that are meaningful for fire detection
- **Domain knowledge:** Based on remote sensing research and physical properties
- **Complementary:** Different indices capture different aspects (vegetation, burn, water)

**Random Forest Configuration:**
- **Number of trees:** 100-500 (more trees = better performance, but slower)
- **Max depth:** 10-20 (deeper trees = more complex patterns, but risk overfitting)
- **Min samples per leaf:** 5-10 (prevents overfitting)
- **Feature sampling:** Use all features (or sample subset for each tree)

**What to Report:**
- Features used (raw bands + spectral indices)
- Random Forest hyperparameters (number of trees, max depth, etc.)
- Feature importance analysis (which features matter most for fire detection)
- Performance metrics (IoU, Dice, Precision, Recall)
- Comparison with deep learning model
- Analysis of when Random Forest fails (spatial context, complex patterns)
- Visual examples showing Random Forest vs. deep learning predictions

**Baseline Comparison Strategy:**

**Evaluation Protocol:**
1. **Same test set:** Evaluate all baselines and deep learning model on identical test set
2. **Same metrics:** Use same evaluation metrics (IoU, Dice, Precision, Recall) for fair comparison
3. **Same preprocessing:** Apply same data preprocessing to all methods
4. **Multiple runs:** Run each baseline multiple times (if stochastic) and report average

**Expected Results:**
- **Threshold-based:** Lowest performance (IoU: 0.40-0.55)
- **Random Forest:** Moderate performance (IoU: 0.55-0.65)
- **Deep Learning (U-Net):** Highest performance (IoU: 0.70-0.75) - target from Section 1.4

**Why Deep Learning Should Outperform:**
1. **Spatial context:** U-Net understands that fires form connected regions (not just individual pixels)
2. **Learned features:** Deep learning learns optimal features automatically (better than hand-crafted)
3. **Complex patterns:** Can learn non-linear, complex relationships between bands
4. **End-to-end learning:** Optimizes entire pipeline for the task (not just classification step)
5. **Transfer learning:** Benefits from pretrained features (ImageNet → satellite imagery)

**When to Investigate (If Deep Learning Doesn't Outperform):**
- **Bug in implementation:** Check data loading, model architecture, training loop
- **Insufficient training:** Model may need more epochs or better hyperparameters
- **Data issues:** Check if training data is correct, sufficient, or properly preprocessed
- **Overfitting:** Deep learning may be overfitting to training set (check train/val gap)
- **Baseline too strong:** If Random Forest achieves IoU > 0.70, problem may be too simple for deep learning

**Baseline Implementation Requirements:**

**For Capstone Project:**
- [ ] Implement threshold-based baseline (SWIR thresholds)
- [ ] Implement Random Forest baseline (with spectral indices)
- [ ] Evaluate both baselines on test set
- [ ] Compare baselines with deep learning model
- [ ] Document baseline performance in final report
- [ ] Explain why deep learning outperforms (or investigate if it doesn't)
- [ ] Include baseline results in evaluation section
- [ ] Visual comparison: show example predictions from all three methods

**Baseline Documentation:**
- **Method description:** How each baseline works
- **Hyperparameters:** Threshold values, Random Forest parameters
- **Performance metrics:** IoU, Dice, Precision, Recall for each baseline
- **Comparison table:** Side-by-side comparison of all methods
- **Failure analysis:** When and why baselines fail
- **Conclusion:** Why deep learning is the right choice for this problem

### 3.4 Model Optimization

Model optimization is the process of **finding the best hyperparameters** (settings that control training but aren't learned) to maximize model performance. This includes tuning learning rates, architecture choices, loss function weights, and other training configurations.

**Why Optimize Hyperparameters?**

- **Performance improvement:** Good hyperparameters can improve IoU by 5-10% compared to default values
- **Efficient training:** Optimal hyperparameters lead to faster convergence and better final performance
- **Academic rigor:** Demonstrates thorough experimentation and understanding of the model
- **Capstone requirement:** Shows systematic approach to model development

**Optimization Strategy:**

For a capstone project, we need a **practical approach** that:
- Doesn't require weeks of computation
- Uses accessible tools (Optuna is free and user-friendly)
- Provides meaningful improvements
- Is well-documented for the report

---

#### 3.4.1 Hyperparameter Search Space

The search space defines **which hyperparameters to tune** and **what values to try**. A good search space balances:
- **Coverage:** Explores enough options to find good hyperparameters
- **Efficiency:** Doesn't waste time on obviously bad values
- **Practicality:** Fits within time/compute constraints

**Hyperparameter Selection Rationale:**

We focus on hyperparameters that have the **biggest impact on performance** and are **practical to tune** within a capstone timeline.

##### Learning Rate

**Search Range:** 1e-5 to 1e-3 (logarithmic scale)

**Why This Range?**

- **Lower bound (1e-5):** Very conservative learning rate - safe but slow. Below this, training becomes impractically slow.
- **Upper bound (1e-3):** Aggressive learning rate - fast but risky. Above this, training becomes unstable (loss oscillates or increases).
- **Logarithmic scale:** Learning rate has exponential effect, so we sample logarithmically (1e-5, 3e-5, 1e-4, 3e-4, 1e-3) rather than linearly.

**Why Learning Rate Matters:**
- **Most important hyperparameter:** Has the biggest impact on training stability and final performance
- **Sensitive:** Small changes (e.g., 1e-4 vs 3e-4) can significantly affect convergence
- **Problem-specific:** Optimal LR depends on dataset, model architecture, and optimizer

**Expected Optimal Value:**
- Typically between 1e-4 and 5e-4 for AdamW optimizer
- May be lower (1e-5 to 3e-5) for pretrained encoder (to preserve pretrained features)
- May be higher (3e-4 to 1e-3) for randomly initialized decoder

##### Encoder

**Search Options:** resnet34, resnet50, efficientnet-b0, efficientnet-b1

**Why These Encoders?**

- **ResNet-34:** Recommended default (see Section 3.1.1) - balanced performance
- **ResNet-50:** Larger, more capacity - may improve accuracy but slower
- **EfficientNet-B0:** Efficient alternative - faster, less memory
- **EfficientNet-B1:** Slightly larger than B0 - middle ground between B0 and ResNet-34

**Why Not Include More Encoders?**
- **Time constraint:** Each encoder requires full training (hours per trial)
- **Diminishing returns:** More encoders = more trials = longer optimization
- **Coverage:** These four cover the main options (small/medium/large, efficient/standard)

**Encoder Selection Rationale:**
- **ResNet-34 vs ResNet-50:** Trade-off between accuracy and speed/memory
- **ResNet vs EfficientNet:** Trade-off between proven performance (ResNet) and efficiency (EfficientNet)
- **Expected outcome:** ResNet-34 or EfficientNet-B0 likely optimal (good balance)

**Why Encoder Matters:**
- **Feature extraction:** Different encoders extract different features
- **Model capacity:** Larger encoders can learn more complex patterns (but may overfit)
- **Training speed:** EfficientNet trains faster, ResNet-50 slower
- **Transfer learning:** All use ImageNet pretrained weights, but quality varies

##### Batch Size

**Search Options:** 8, 16, 32

**Why These Values?**

- **8:** Minimum practical size - works on most GPUs, good for small datasets
- **16:** Medium size - good balance, works on 8GB+ GPUs
- **32:** Large size - faster training, requires 16GB+ GPUs

**Why Not Include Smaller (4) or Larger (64)?**
- **4:** Too small - unstable gradients, slow training
- **64:** Too large - requires very high-end GPUs, may hurt generalization

**Why Batch Size Matters:**
- **Gradient stability:** Larger batches = more stable gradients (but may generalize worse)
- **Training speed:** Larger batches = fewer iterations per epoch = faster training
- **Memory constraint:** Limited by GPU memory (can't go too large)
- **Generalization:** Smaller batches sometimes generalize better (stochastic gradient noise helps)

**Expected Optimal Value:**
- Typically 8 or 16 for this task (good balance)
- 32 may be faster but may not improve accuracy

##### Loss Weights (BCE:Dice)

**Search Options:** 0.3:0.7, 0.5:0.5, 0.7:0.3

**Why These Ratios?**

- **0.5:0.5 (default):** Balanced contribution from both losses
- **0.3:0.7:** Emphasize Dice Loss (better for class imbalance, region overlap)
- **0.7:0.3:** Emphasize BCE Loss (better for pixel-level accuracy)

**Why Not Include More Options?**
- **Three options sufficient:** Covers main trade-offs (BCE vs Dice emphasis)
- **More options:** Diminishing returns, adds trials without much benefit
- **Default (0.5:0.5) is usually good:** Rarely need extreme ratios

**Why Loss Weights Matter:**
- **Class imbalance:** Higher Dice weight helps with imbalanced data (fire pixels are rare)
- **Pixel vs region:** Higher BCE weight emphasizes pixel accuracy, higher Dice weight emphasizes region overlap
- **Task-specific:** Optimal ratio depends on whether pixel accuracy or region overlap is more important

**Expected Optimal Value:**
- Likely 0.5:0.5 (default) or 0.3:0.7 (if class imbalance is severe)
- 0.7:0.3 less common (pixel accuracy usually less important than region overlap)

##### Dropout

**Search Options:** 0.0, 0.1, 0.2

Dropout is a regularization technique that randomly sets some neurons to zero during training to prevent overfitting by preventing co-adaptation of neurons.

**Why These Values?**

- **0.0 (no dropout):** Baseline - no regularization from dropout
- **0.1 (light dropout):** Light regularization - may help if slight overfitting
- **0.2 (moderate dropout):** Moderate regularization - helps if overfitting is an issue

**Why Not Include Higher (0.3, 0.5)?**
- **Too aggressive:** High dropout can hurt performance (model becomes too regularized)
- **Rarely needed:** For segmentation with pretrained encoders, dropout is often not needed

**Why Dropout Matters:**
- **Overfitting prevention:** Helps if model overfits to training data
- **May hurt performance:** If model doesn't overfit, dropout can reduce capacity unnecessarily
- **Architecture-dependent:** Some architectures benefit more from dropout than others

**Expected Optimal Value:**
- Likely 0.0 or 0.1 (pretrained encoders provide regularization, may not need dropout)
- 0.2 only if significant overfitting observed

**Hyperparameter Search Strategy:**

**Two-Phase Approach:**

**Phase 1: Coarse Search (20-30 trials)**
- Explore entire search space broadly
- Identify promising regions
- Goal: Find approximate optimal values

**Phase 2: Fine-Tuning (10-20 trials)**
- Focus search around best results from Phase 1
- Narrower ranges: ±50% around best values
- Goal: Refine to exact optimal values

**Why Two Phases?**
- **Efficiency:** Coarse search finds good region quickly
- **Refinement:** Fine-tuning improves within good region
- **Time management:** Better use of limited trials

**Optuna Configuration:**

**Why Optuna?**
- **Free and accessible:** No cost, easy to set up
- **User-friendly:** Simple API, good documentation
- **Efficient:** Uses Tree-structured Parzen Estimator (TPE) for smart search
- **Academic-friendly:** Good for capstone (not overly complex)

**Trial Configuration:**
- **Number of trials:** 20-50 total (30-40 recommended)
- **Pruning:** Enable pruning to stop bad trials early (saves time)
- **Objective:** Maximize validation IoU (not loss - see Section 3.2.4)
- **Timeout:** Set per-trial timeout (e.g., 2 hours) to prevent stuck trials

**Why 20-50 Trials?**
- **Too few (<20):** May not explore search space adequately
- **Too many (>50):** Diminishing returns, time-consuming for capstone
- **Sweet spot:** 30-40 trials provides good exploration without excessive time

**Hyperparameter Importance Analysis:**

**What it is:**
- Optuna automatically analyzes which hyperparameters matter most
- Helps understand what drives performance

**Why This Matters:**
- **Understanding:** Learn which hyperparameters are critical vs. less important
- **Future optimization:** Focus on important hyperparameters in future work
- **Documentation:** Shows systematic approach to optimization

**Expected Importance Order:**
1. **Learning rate** (most important) - biggest impact on performance
2. **Encoder** (very important) - determines model capacity
3. **Loss weights** (moderately important) - affects class imbalance handling
4. **Batch size** (less important) - mainly affects training speed
5. **Dropout** (least important) - often not needed with pretrained encoders

**Documentation Requirements:**

**For Capstone Report:**
- **Search space:** Document all hyperparameters and ranges explored
- **Best hyperparameters:** Report optimal values found
- **Performance improvement:** Compare optimized vs. default hyperparameters
- **Convergence plots:** Show how validation IoU improved over trials
- **Hyperparameter importance:** Include Optuna's importance analysis
- **Two-phase results:** Document Phase 1 (coarse) and Phase 2 (fine-tuning) separately
- **Time investment:** Report total optimization time (trials × time per trial)

**Optimization Checklist:**
- [ ] Define search space (learning rate, encoder, batch size, loss weights, dropout)
- [ ] Set up Optuna with 20-50 trials
- [ ] Run Phase 1: Coarse search (20-30 trials)
- [ ] Analyze results, identify promising regions
- [ ] Run Phase 2: Fine-tuning (10-20 trials) around best results
- [ ] Document best hyperparameters found
- [ ] Report performance improvement vs. default hyperparameters
- [ ] Include hyperparameter importance analysis
- [ ] Create convergence plots (validation IoU over trials)

---

#### 3.4.2 Model Export Formats

Model export converts the trained PyTorch model into a format suitable for deployment. The exported model can be loaded and used for inference without the full training code.

**Why Export Models?**

- **Deployment:** Need model in format compatible with deployment environment
- **Inference speed:** Some formats are optimized for faster inference
- **Portability:** Some formats work across different platforms/languages
- **Simplicity:** Exported models are self-contained (no training dependencies)

**Export Format Selection Rationale:**

For a capstone project, we need a format that is:
- **Simple to use:** Easy to export and load
- **Compatible:** Works with deployment platform (HuggingFace Spaces, Streamlit)
- **Reliable:** Produces same results as training model
- **Well-documented:** Easy to understand and explain

##### PyTorch (.pt) Format

**What it is:**
- Native PyTorch format - saves model state dictionary
- Standard format for PyTorch models

**Why Recommended for Capstone:**

**Advantages:**
- ✅ **Simplest:** Native PyTorch format, no conversion needed
- ✅ **Reliable:** Guaranteed to work with PyTorch (same framework as training)
- ✅ **Complete:** Saves all model weights exactly as trained
- ✅ **Easy to load:** Simple loading code, well-documented
- ✅ **Compatible:** Works with HuggingFace Spaces, Streamlit, FastAPI
- ✅ **No conversion errors:** No risk of conversion bugs or precision loss

**Limitations:**
- ⚠️ **PyTorch-only:** Requires PyTorch to load (not a problem for Python deployment)
- ⚠️ **Not optimized:** May be slightly slower than optimized formats (negligible for capstone)
- ⚠️ **Larger file size:** May be larger than optimized formats (not a concern for capstone)

**When to Use:**
- **⭐ Recommended for capstone** - simplest and most reliable
- Python-based deployment (HuggingFace Spaces, Streamlit, FastAPI)
- When simplicity and reliability are more important than optimization

**Export Process:**
1. Save model state dictionary after training
2. Load model architecture (same as training)
3. Load state dictionary into model
4. Set model to evaluation mode
5. Use for inference

**Verification:**
- Test that exported model produces same predictions as training model
- Compare predictions on sample inputs (should be identical)
- Document model size, inference time, memory requirements

##### TorchScript (.pt) Format

**What it is:**
- Optimized PyTorch format - compiles model to TorchScript
- Can be used without Python (C++ deployment)

**Why Not Recommended for Capstone:**

**Advantages:**
- ✅ **Faster inference:** Optimized execution (marginal improvement)
- ✅ **C++ compatible:** Can be used in C++ applications
- ✅ **Production-ready:** Used in production PyTorch serving

**Disadvantages:**
- ⚠️ **More complex:** Requires tracing or scripting (more steps)
- ⚠️ **Compatibility issues:** Some PyTorch operations not supported
- ⚠️ **Debugging harder:** Less intuitive if issues occur
- ⚠️ **Overkill for capstone:** Unnecessary complexity for demo

**When to Use:**
- Production deployment with PyTorch serving
- C++ applications
- When inference speed is critical (not a concern for capstone)

**Why Skip for Capstone:**
- Adds complexity without significant benefit for a demo
- PyTorch .pt format is sufficient and simpler
- Time better spent on other aspects of the project

##### ONNX (.onnx) Format

**What it is:**
- Cross-platform format - works with multiple frameworks (PyTorch, TensorFlow, etc.)
- Industry standard for model interoperability

**Why Not Recommended for Capstone:**

**Advantages:**
- ✅ **Cross-platform:** Works with multiple frameworks
- ✅ **Optimized:** Can be optimized for different hardware
- ✅ **Industry standard:** Widely used in production

**Disadvantages:**
- ⚠️ **Conversion complexity:** Requires conversion from PyTorch (potential issues)
- ⚠️ **Compatibility:** Not all PyTorch operations convert cleanly
- ⚠️ **Debugging:** Harder to debug conversion issues
- ⚠️ **Unnecessary:** Not needed if staying in PyTorch ecosystem

**When to Use:**
- Cross-platform deployment (different frameworks)
- Hardware optimization (specific accelerators)
- Production systems requiring framework flexibility

**Why Skip for Capstone:**
- Unnecessary complexity for a Python-based demo
- PyTorch .pt format is sufficient
- Conversion may introduce bugs or precision issues
- Time better spent on other project aspects

**Export Format Decision Tree:**

**Model Export Requirements:**

**For Capstone Demo:**
- [ ] Export model in PyTorch .pt format
- [ ] Verify exported model produces same results as training model
- [ ] Document model size (file size in MB)
- [ ] Document inference time (time per prediction)
- [ ] Document memory requirements (RAM/VRAM needed for inference)
- [ ] Test loading and inference in deployment environment
- [ ] Include export/loading code in repository

**Export Documentation:**
- **Model file:** Location and name of exported model
- **Model size:** File size (e.g., "150 MB")
- **Inference time:** Time per 256×256 patch (e.g., "0.5 seconds on CPU, 0.1 seconds on GPU")
- **Memory requirements:** RAM needed for loading (e.g., "2 GB RAM, 1 GB VRAM")
- **Loading procedure:** How to load model for inference
- **Verification:** Confirmation that exported model matches training model

**Model Export Best Practices:**

1. **Save after best validation:** Export model with best validation IoU (not final epoch)
2. **Include metadata:** Save hyperparameters, data version, training metrics with model
3. **Version control:** Don't commit large model files to Git (use Git LFS or external storage)
4. **Documentation:** Clearly document model version, training date, performance metrics
5. **Testing:** Thoroughly test exported model before deployment
6. **Backup:** Keep backup of exported model (don't rely on single copy)

### 3.5 Experiment Tracking & MLOps

Experiment tracking systematically records all training runs, hyperparameters, metrics, and configurations. This enables reproducibility, comparison of experiments, and debugging of training issues.

**Why This Matters for Capstone:**

- **Reproducibility:** Essential for academic work - must be able to reproduce results
- **Comparison:** Compare different hyperparameters, architectures, and data configurations
- **Debugging:** Identify what changed when performance degrades
- **Documentation:** Automatic logging reduces manual documentation burden
- **Academic rigor:** Demonstrates systematic approach to model development

---

#### 3.5.1 Weights & Biases (W&B) Setup

**Why Weights & Biases?**

**Tool Selection Rationale:**

| Tool | Pros | Cons | Verdict |
|------|------|------|---------|
| **Weights & Biases** | ✅ Free for academic use<br>✅ Intuitive web interface<br>✅ Excellent visualization<br>✅ Easy integration<br>✅ Automatic logging | ⚠️ Requires internet connection | **⭐ Recommended** |
| MLflow | ✅ Open source<br>✅ Can run locally | ⚠️ More complex setup<br>⚠️ Less intuitive UI<br>⚠️ Requires infrastructure | Good alternative if prefer open source |
| TensorBoard | ✅ Built into PyTorch<br>✅ No account needed | ⚠️ Limited features<br>⚠️ Less user-friendly<br>⚠️ No cloud sync | Use for local debugging only |

**Why W&B is Recommended:**

- **Academic-friendly:** Free tier sufficient for capstone, no credit card required
- **User-friendly:** Web interface easier than TensorBoard for non-developers
- **Automatic logging:** Minimal code changes needed, tracks most things automatically
- **Visualization:** Excellent plots for training curves, hyperparameter sweeps, confusion matrices
- **Collaboration:** Team members can view experiments without local setup
- **Reports:** Built-in report generation for capstone documentation

**What to Track and Why:**

**Hyperparameters:**
- **Learning rate, batch size, encoder, loss weights:** Core training configuration - essential for reproducibility
- **Optimizer settings:** Weight decay, betas - affect training dynamics
- **Why track:** Need to know exact configuration that produced best results

**Training Metrics (Per Epoch):**
- **Loss components:** Total loss, BCE loss, Dice loss - understand training dynamics
- **Training IoU, Dice, Precision, Recall:** Monitor training performance (though validation is more important)
- **Why track:** Identify overfitting (train metrics much higher than validation), training stability

**Validation Metrics (Per Epoch):**
- **Validation IoU, Dice, Precision, Recall:** Primary performance indicators
- **Why track:** These determine best model selection, early stopping, and final performance

**Data Version:**
- **Dataset checksum/version:** Which data version was used for training
- **Preprocessing parameters:** Normalization method, augmentation settings, patch size
- **Why track:** Critical for reproducibility - same model + same data = same results

**Model Architecture:**
- **Encoder type, input/output shapes:** Model configuration
- **Why track:** Compare different architectures, document final model

**Training Time:**
- **Total time, time per epoch:** Performance benchmarking, resource planning
- **Why track:** Understand computational requirements, compare training efficiency

**Environment:**
- **PyTorch version, CUDA version, requirements.txt:** Reproducibility
- **Why track:** Different versions can produce different results (even with same random seed)

**Random Seeds:**
- **Training seed, data loading seed:** Reproducibility
- **Why track:** Must be able to reproduce exact results

**Best Practices:**

**Log Every Run:**
- Don't skip failed experiments - they provide valuable information about what doesn't work
- Failed runs help identify bugs, unstable configurations, or data issues

**Use Tags:**
- Organize experiments: "baseline", "augmentation-test", "hyperparameter-search", "final-model"
- Makes it easy to filter and compare related experiments

**Save Model Checkpoints:**
- Use W&B artifact storage for model checkpoints (if within free tier limits)
- Links models to specific training runs automatically

**Create Reports:**
- Use W&B reports to summarize experiments for capstone documentation
- Include training curves, best hyperparameters, performance comparisons

**Link Model Versions to Data Versions:**
- Document which data version produced which model
- Critical for understanding model performance changes

**Alternative Tools:**

**MLflow:** Good alternative if you prefer open-source tools or need local-only tracking. More complex setup but offers more control. For capstone, W&B's simplicity is usually preferable.

**TensorBoard:** Use for local debugging and quick visualizations, but not sufficient for comprehensive experiment tracking. No cloud sync, limited collaboration features.

---

#### 3.5.2 Model Versioning

Model versioning assigns unique identifiers to trained models and stores metadata about each version. This enables tracking which model was used when, comparing model performance, and rolling back to previous versions if needed.

**Why Model Versioning Matters:**

- **Reproducibility:** Know exactly which model produced which results
- **Comparison:** Compare performance across model versions
- **Deployment:** Track which model is deployed in production/demo
- **Academic documentation:** Required for capstone report - must document model versions

**Versioning Strategy:**

**Semantic Versioning:**
- Format: `v1.0.0` (major.minor.patch)
- **Major version (v1.x.x):** Architecture changes (e.g., different encoder)
- **Minor version (v.x.1.x):** Significant improvements (e.g., better hyperparameters, new training data)
- **Patch version (v.x.x.1):** Bug fixes, minor improvements

**Example:**
- `v1.0.0`: Initial baseline model (ResNet-34, default hyperparameters)
- `v1.1.0`: Improved model (optimized hyperparameters from Optuna)
- `v1.2.0`: Model trained with additional data augmentation
- `v2.0.0`: Model with EfficientNet-B0 encoder (architecture change)

**Model Metadata to Store:**

| Metadata | Purpose | Example |
|----------|---------|---------|
| **Version number** | Unique identifier | v1.1.0 |
| **Training date** | Temporal reference | 2026-02-15 |
| **Performance metrics** | Model quality | IoU: 0.72, Dice: 0.82 |
| **Hyperparameters** | Reproducibility | LR: 3e-4, batch: 16, encoder: resnet34 |
| **Data version** | Reproducibility | CEMS-Wildfire v1.0, checksum: abc123 |
| **W&B run ID** | Link to experiment tracking | run-abc123xyz |
| **Git commit** | Code version | commit: def456 |
| **Training time** | Resource info | 12 hours, 45 epochs |

**Linking to W&B Runs:**

- Each model version should link to specific W&B run
- Enables automatic retrieval of all training details
- Provides complete audit trail from code → data → model → metrics

**Model Card Documentation:**

For each model version, create a model card documenting:
- **Performance:** Metrics on test set and Catalonia validation set
- **Limitations:** Known failure modes, edge cases, geographic biases
- **Intended use:** What the model is designed for (and what it's not)
- **Training data:** Summary of datasets used
- **Ethical considerations:** Potential biases, misuse scenarios

**Model Versioning Checklist:**
- [ ] Use semantic versioning (v1.0.0 format)
- [ ] Store metadata with each model (date, metrics, hyperparameters, data version)
- [ ] Link model versions to W&B runs
- [ ] Create model card for final model version
- [ ] Document version history in repository README
- [ ] Tag model versions in Git (if storing model metadata in Git)

### 3.6 Evaluation Metrics

Model evaluation measures how well the trained model performs on unseen data. For segmentation tasks, this requires multiple metrics to capture different aspects of performance (overlap, accuracy, false alarms, missed detections).

**Why Comprehensive Evaluation Matters:**

- **Academic rigor:** Multiple metrics provide complete picture of model performance
- **Real-world relevance:** Different metrics matter for different use cases (precision for alerts, recall for safety)
- **Model comparison:** Enables fair comparison with baselines and other methods
- **Failure analysis:** Identifies where model needs improvement

---

#### 3.6.1 Primary Metrics

**Why These Four Metrics?**

These metrics capture the essential aspects of segmentation performance: overlap quality (IoU, Dice), detection accuracy (Precision), and detection completeness (Recall).

**IoU (Intersection over Union) - Primary Metric**

**What it measures:** Overlap between predicted and ground truth fire areas.

**Formula:** `IoU = TP / (TP + FP + FN)`

**Why it matters:**
- **Primary segmentation metric:** Standard in computer vision literature
- **Directly measures overlap:** What we care about for fire area detection
- **Balanced:** Considers both false positives and false negatives
- **Interpretable:** 0.70 means 70% of combined predicted and actual fire area overlaps correctly

**Why target ≥ 0.70:**
- **Realistic for capstone:** Achievable with good data and training
- **Meaningful performance:** 0.70 IoU indicates good segmentation quality
- **Literature standard:** Common target in remote sensing segmentation tasks
- **Below 0.60:** Poor quality, model likely has fundamental issues
- **Above 0.75:** Excellent performance, may be difficult to achieve

**Dice Score (F1-Score for Segmentation)**

**What it measures:** Harmonic mean of precision and recall, emphasizing overlap.

**Formula:** `Dice = 2TP / (2TP + FP + FN)`

**Why it matters:**
- **More forgiving than IoU:** Less penalized by small differences (better for small fires)
- **Directly related to IoU:** Dice ≈ 2×IoU / (1 + IoU) for binary segmentation
- **Common in medical imaging:** Widely used in segmentation literature
- **Class imbalance handling:** Works well with imbalanced data

**Why target ≥ 0.80:**
- **Higher than IoU target:** Dice is typically higher than IoU (e.g., IoU 0.70 ≈ Dice 0.82)
- **Strong performance indicator:** 0.80 Dice indicates good overlap
- **Achievable:** With proper class imbalance handling, this is realistic

**Precision (Positive Predictive Value)**

**What it measures:** Fraction of predicted fires that are actually real fires.

**Formula:** `Precision = TP / (TP + FP)`

**Why it matters:**
- **False alarm control:** Critical for alert systems - high precision means fewer false alarms
- **Trust:** Users trust system more if detections are reliable
- **Resource efficiency:** Reduces wasted resources investigating false positives
- **Real-world impact:** False alarms can desensitize users or waste emergency resources

**Why target ≥ 0.85:**
- **High reliability:** 85% of detections are correct - acceptable for alert systems
- **Below 0.75:** Too many false alarms, system becomes unreliable
- **Above 0.90:** Excellent, but may come at cost of lower recall

**Recall (Sensitivity, True Positive Rate)**

**What it measures:** Fraction of real fires that are successfully detected.

**Formula:** `Recall = TP / (TP + FN)`

**Why it matters:**
- **Safety critical:** Missing fires has severe consequences (safety, property, environment)
- **Completeness:** Measures how well model finds all fires
- **Class imbalance challenge:** Most difficult metric to optimize (fire pixels are rare)

**Why target ≥ 0.80:**
- **Good coverage:** Detecting 80% of fires is strong performance
- **Below 0.70:** Missing too many fires, model needs improvement
- **Above 0.85:** Excellent, but may come at cost of lower precision (more false alarms)

**Metric Trade-offs:**

| Scenario | Priority | Rationale |
|----------|----------|-----------|
| **Alert system** | Precision > Recall | False alarms waste resources, reduce trust |
| **Safety monitoring** | Recall > Precision | Missing fires is worse than false alarms |
| **Area measurement** | IoU > Precision/Recall | Accurate area measurement is primary goal |
| **Capstone balance** | All metrics important | Demonstrate balanced performance |

---

#### 3.6.2 Secondary Metrics

**Why Secondary Metrics?**

Primary metrics focus on segmentation quality, but secondary metrics provide additional insights into model behavior and practical usability.

**Pixel Accuracy:**

**What it measures:** Overall percentage of correctly classified pixels.

**Why it's secondary:**
- **Misleading with class imbalance:** Can achieve 99% accuracy by predicting "no fire" for everything
- **Less informative:** Doesn't capture fire detection quality (fire pixels are <1% of image)
- **Still useful:** Provides overall context, but not primary metric

**When to report:** Include for completeness, but don't optimize for it.

**False Positive Rate (FPR):**

**What it measures:** Fraction of non-fire pixels incorrectly classified as fire.

**Formula:** `FPR = FP / (FP + TN)`

**Why it matters:**
- **Alert reliability:** Low FPR means fewer false alarms
- **Complement to precision:** Precision focuses on detections, FPR focuses on non-fire areas
- **Real-world impact:** High FPR can overwhelm users with false alerts

**When to use:** Critical for alert systems, less important for area measurement tasks.

**Detection Latency:**

**What it measures:** Time from image input to prediction output.

**Why it matters:**
- **Real-time applications:** Must be fast enough for near-real-time detection
- **User experience:** Slow inference frustrates users
- **Resource planning:** Determines hardware requirements

**Target:** < 5 seconds per tile for capstone demo (not production requirement).

---

#### 3.6.3 Evaluation Protocol

**Why a Structured Protocol?**

A systematic evaluation protocol ensures fair, comprehensive, and reproducible assessment of model performance. This is essential for academic work and model comparison.

**1. Per-Patch Evaluation**

**What it is:** Calculate metrics on individual 256×256 patches.

**Why this matters:**
- **Standard unit:** Patches are the training unit, so evaluation should match
- **Statistical validity:** Many patches provide robust statistics
- **Identifies problematic patches:** Can find specific failure cases

**How to do it:**
- Evaluate each patch independently
- Aggregate metrics across all test patches (mean, std, median)
- Report distribution (not just mean) - some patches may be much harder than others

**2. Per-Event Evaluation**

**What it is:** Aggregate predictions for entire fire events (multiple patches per fire).

**Why this matters:**
- **Real-world relevance:** Fires span multiple patches, evaluation should reflect this
- **Handles patch boundaries:** Some fires may be split across patches
- **More meaningful:** Users care about detecting entire fires, not just patches

**How to do it:**
- Group patches by fire event (using ground truth fire IDs or geographic proximity)
- Aggregate predictions within each event
- Calculate metrics per event, then aggregate across events

**3. Threshold Analysis**

**What it is:** Plot precision-recall curve and ROC curve, find optimal threshold.

**Why this matters:**
- **Threshold selection:** Default 0.5 may not be optimal for this task
- **Trade-off visualization:** See precision/recall trade-off across thresholds
- **Task-specific optimization:** Can optimize for precision (alerts) or recall (safety)

**How to do it:**
- Vary threshold from 0.1 to 0.9 (or 0.0 to 1.0)
- Calculate precision and recall at each threshold
- Plot Precision-Recall curve and ROC curve
- Find optimal threshold (e.g., maximize F1, or balance precision/recall based on use case)

**4. Geographic Generalization**

**What it is:** Test on held-out regions (e.g., train on other regions, test on Catalonia).

**Why this matters:**
- **Transfer learning validation:** Tests if model generalizes to new regions
- **Real-world applicability:** Model must work on Catalonia (target region)
- **Identifies geographic bias:** May perform well on training regions but poorly on Catalonia

**How to do it:**
- Train on non-Catalonia data (e.g., other European fires)
- Test on Catalonia validation set (created from Generalitat data)
- Compare Catalonia performance vs general test set performance
- Analyze transfer learning effectiveness

---

#### 3.6.4 Comprehensive Evaluation Requirements

**Why Comprehensive Evaluation?**

A single metric doesn't tell the full story. Comprehensive evaluation provides complete understanding of model strengths, weaknesses, and failure modes.

**Confusion Matrices:**

**What they show:** Breakdown of predictions into True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).

**Why multiple levels:**
- **Per-patch:** Understand patch-level errors
- **Per-event:** Understand event-level errors (more relevant for users)
- **Aggregate:** Overall performance summary

**What to analyze:**
- **TP:** Correct fire detections - verify these are actually fires
- **FP:** False alarms - identify common false positive sources (urban areas, hot surfaces)
- **FN:** Missed fires - identify why fires were missed (size, smoke, edge cases)
- **TN:** Correct non-fire - less interesting but confirms model works on background

**Error Analysis:**

**Why it matters:**
- **Identifies improvement areas:** Know where to focus future work
- **Documents limitations:** Essential for model card and academic honesty
- **Real-world preparation:** Understand failure modes before deployment

**Common Failure Modes to Document:**

**False Positives:**
- **Urban areas:** Hot surfaces (rooftops, roads) can trigger fire detection
- **Agricultural burns:** Controlled burns may be detected (could be feature or bug)
- **Hot surfaces:** Industrial areas, solar panels, reflective surfaces
- **Cloud edges:** Bright cloud edges can mimic fire signatures

**False Negatives:**
- **Small fires:** Fires smaller than a few pixels may be missed
- **Smoke-obscured fires:** Heavy smoke can hide fire signatures
- **Edge cases:** Fires at patch boundaries, partially visible fires
- **Low-intensity fires:** Smoldering fires with weak SWIR signal

**Visual Examples:**
- Include sample predictions showing TP, FP, FN cases
- Helps readers understand model behavior
- Essential for capstone report visualization

**Geographic Analysis:**
- Performance breakdown by region/terrain type if possible
- Identifies geographic biases
- Helps understand transfer learning effectiveness

**Threshold Optimization:**

**Why optimize threshold:**
- **Default 0.5 may not be optimal:** Task-specific requirements may favor precision or recall
- **Use case dependent:** Alert systems need high precision, safety monitoring needs high recall
- **Performance improvement:** Optimal threshold can improve F1 by 2-5%

**ROC Curve (Receiver Operating Characteristic):**
- **X-axis:** False Positive Rate (FPR)
- **Y-axis:** True Positive Rate (TPR = Recall)
- **What it shows:** Trade-off between detecting fires (recall) and false alarms (FPR)
- **AUC (Area Under Curve):** Single number summarizing ROC performance (higher is better)

**Precision-Recall Curve:**
- **X-axis:** Recall
- **Y-axis:** Precision
- **What it shows:** Trade-off between detecting fires (recall) and detection reliability (precision)
- **More informative for imbalanced data:** ROC can be misleading with class imbalance, PR curve is better

**Optimal Threshold Selection:**
- **Maximize F1:** Balance precision and recall
- **Maximize precision:** If false alarms are costly (alert systems)
- **Maximize recall:** If missing fires is worse (safety monitoring)
- **Document rationale:** Explain why chosen threshold is appropriate for use case

**Catalonia-Specific Evaluation:**

**Why mandatory:**
- **Target region:** Model must work on Catalonia (project focus)
- **Transfer learning test:** Tests if model generalizes from training regions to Catalonia
- **Academic requirement:** Demonstrates real-world applicability

**What to report:**
- **Separate metrics:** IoU, Dice, Precision, Recall on Catalonia validation set
- **Comparison:** Catalonia performance vs general test set performance
- **Analysis:** Why performance differs (if it does) - geographic bias, data distribution differences
- **Transfer learning effectiveness:** How well model transfers from training regions to Catalonia

**Evaluation Checklist:**
- [ ] Calculate all primary metrics (IoU, Dice, Precision, Recall) on test set
- [ ] Calculate secondary metrics (FPR, pixel accuracy, latency)
- [ ] Create confusion matrices (per-patch, per-event, aggregate)
- [ ] Perform threshold optimization (ROC and PR curves)
- [ ] Evaluate on Catalonia validation set separately
- [ ] Document error analysis (common failure modes, visual examples)
- [ ] Compare with baseline models (threshold-based, Random Forest)
- [ ] Include all results in evaluation report

---

## 4. Application Features

Application features define what the system can do from a user perspective. For a capstone project, features should demonstrate model capabilities while remaining implementable within time constraints.

**Feature Selection Rationale:**

- **Core features:** Essential for demonstrating model inference and basic functionality
- **Advanced features:** Show system potential but may be simplified for capstone
- **Focus on ML:** Prioritize features that showcase model performance over complex UI/UX

---

### 4.1 Core Features

**Why These Core Features?**

Core features are the minimum necessary to demonstrate a working wildfire detection system. They showcase model inference, spatial analysis, and basic usability.

#### 4.1.1 Fire Detection

**What it does:** Identifies fire pixels in satellite imagery using the trained segmentation model.

**Why binary segmentation:**
- **Model output:** Segmentation model produces pixel-level classifications (fire/no-fire)
- **Standard approach:** Binary segmentation is standard for fire detection tasks
- **Interpretable:** Clear output - each pixel is either fire or not fire

**Confidence map:**
- **What it is:** Probability values (0-1) for each pixel, not just binary classification
- **Why it matters:** Users can see detection confidence, helps identify uncertain detections
- **Use cases:** Threshold adjustment, uncertainty visualization, quality assessment

**Threshold adjustment:**
- **What it is:** User can change detection threshold (default 0.5) to balance precision/recall
- **Why it matters:** Different use cases need different thresholds (alerts need high precision, safety needs high recall)
- **Implementation:** Simple parameter in API/UI, no model retraining needed

#### 4.1.2 Fire Area Extraction

**What it does:** Converts pixel-level predictions into geographic polygons for visualization and analysis.

**Polygon extraction:**
- **Why needed:** Binary masks are raster data (pixels), but users need vector data (polygons) for GIS tools
- **How it works:** Convert connected fire pixels into polygons using contour detection
- **Use cases:** Overlay on maps, area calculation, export to GIS software

**Minimum area filter:**
- **What it is:** Discard detections smaller than threshold (e.g., 0.1 hectares)
- **Why it matters:** Filters noise and very small detections that may be false positives
- **Typical threshold:** 0.1-0.5 hectares (depends on use case and spatial resolution)
- **Rationale:** Very small fires (<0.1 ha) are often false positives or not actionable

**GeoJSON output:**
- **What it is:** Standard format for geographic data (JSON with coordinates)
- **Why GeoJSON:** Widely supported by web maps (Leaflet, Mapbox, Google Maps), GIS software (QGIS, ArcGIS), and APIs
- **Alternative formats:** Shapefile (more complex), KML (Google Earth), but GeoJSON is most universal

**Coordinate system:**
- **WGS84 (EPSG:4326):** Standard web coordinate system (latitude/longitude) - used for map display
- **UTM Zone 31N (EPSG:32631):** Projected coordinate system for Catalonia - used for accurate area calculations
- **Why both:** Web maps use WGS84, but area calculations need projected coordinates (UTM) for accuracy
- **Conversion:** System converts between coordinate systems as needed

#### 4.1.3 Fire Area Measurement

**What it does:** Calculates quantitative metrics about detected fire areas.

**Why these metrics:**
- **Total area:** Primary metric - how much area is affected by fire
- **Fire percentage:** Contextual metric - what fraction of analyzed region is on fire
- **Perimeter:** Useful for firefighting planning - boundary length indicates containment difficulty
- **Centroid:** Geographic center - useful for location identification and mapping
- **Bounding box:** Extent coordinates - defines region of interest, useful for zooming/display

**Total area calculation:**
- **Method:** Sum of all fire pixels × pixel area (accounts for spatial resolution)
- **Units:** Hectares (standard for fire area reporting)
- **Accuracy:** Depends on spatial resolution (10-20m for Sentinel-2) and coordinate system (UTM for accuracy)

**Why hectares:**
- **Standard unit:** Fire area is typically reported in hectares (or acres in US)
- **Meaningful scale:** Hectares are appropriate for fire sizes (small fires: <1 ha, large fires: >100 ha)
- **User familiarity:** Emergency services and forestry departments use hectares

---

### 4.2 Advanced Features

**Why Advanced Features?**

Advanced features demonstrate system potential beyond basic detection. For capstone, these may be simplified or partially implemented to show capability without excessive complexity.

#### 4.2.1 Fire Spread Detection (Multi-temporal Analysis)

**What it does:** Compares fire detections across multiple time periods to identify spreading fires.

**Why this matters:**
- **Real-world relevance:** Fire spread is critical information for emergency response
- **Demonstrates capability:** Shows system can do more than single-image detection
- **Academic value:** Multi-temporal analysis is advanced remote sensing technique

**Temporal comparison:**
- **What it is:** Compare detections from two dates (T1 and T2) to identify changes
- **Why two dates:** Simplest multi-temporal analysis - more dates add complexity
- **Time interval:** Typically 1-7 days (depends on satellite revisit frequency and fire spread rate)

**Spread area:**
- **What it is:** New fire areas present in T2 but not in T1
- **Why it matters:** Identifies actively spreading fires (critical for emergency response)
- **Calculation:** Set difference (T2 fire areas - T1 fire areas)

**Contained area:**
- **What it is:** Fire areas present in T1 but not in T2 (fire extinguished or contained)
- **Why it matters:** Shows fire suppression effectiveness, identifies contained fires
- **Calculation:** Set difference (T1 fire areas - T2 fire areas)

**Spread direction and rate:**
- **What it is:** Vector indicating predominant spread direction, hectares per day spread rate
- **Why it matters:** Helps predict future spread, plan firefighting resources
- **Complexity:** Requires centroid calculation and temporal interpolation - may be simplified for capstone

**Spread alert:**
- **What it is:** Boolean flag indicating if fire is actively spreading
- **Why it matters:** Prioritizes alerts - spreading fires need immediate attention
- **Criteria:** Spread area > threshold (e.g., >1 hectare) indicates active spread

**Capstone simplification:**
- **Focus on spread area:** Most important metric, simplest to implement
- **Skip complex features:** Spread direction/rate may be omitted if time-constrained
- **Demonstrate concept:** Show that multi-temporal analysis is possible, even if simplified

#### 4.2.2 Alert System

**What it does:** Automatically generates alerts when fires are detected, with severity levels based on fire characteristics.

**Why alerts matter:**
- **Real-world application:** Automated alerts are key use case for fire detection systems
- **Demonstrates integration:** Shows system can integrate with notification systems
- **User value:** Alerts provide actionable information to users

**Alert level rationale:**

**LOW (🟡 Yellow):**
- **Criteria:** < 1 hectare, not spreading
- **Why:** Small fires may be false positives or controlled burns - monitor but low priority
- **Action:** Log detection, no immediate alert needed

**MEDIUM (🟠 Orange):**
- **Criteria:** 1-10 hectares, or small but spreading
- **Why:** Moderate-sized fires or spreading small fires need attention
- **Action:** Send alert to monitoring systems, prepare for potential escalation

**HIGH (🔴 Red):**
- **Criteria:** 10-100 hectares, or medium and spreading
- **Why:** Large fires or actively spreading fires require immediate response
- **Action:** Send high-priority alerts, notify emergency services

**CRITICAL (⚫ Black):**
- **Criteria:** > 100 hectares, or large and rapidly spreading
- **Why:** Very large fires or rapidly spreading fires are emergencies
- **Action:** Immediate high-priority alerts, emergency response activation

**Alert content:**
- **Timestamp (UTC):** When fire was detected - critical for temporal context
- **Location:** Coordinates and nearest municipality - essential for response planning
- **Fire area:** Size in hectares - determines alert level and resource allocation
- **Alert level:** Severity indicator - helps prioritize response
- **Spread status:** Whether fire is spreading - critical for resource planning
- **Link to dashboard:** Allows users to view detailed information

**Capstone simplification:**
- **Basic alert levels:** Implement 2-3 levels (LOW, HIGH) instead of all 4
- **Simple notifications:** Email or webhook only, skip SMS/push
- **Focus on detection:** Alert generation is more important than complex notification routing

#### 4.2.3 Notification Channels

**What it does:** Delivers alerts through various communication channels.

**Why multiple channels:**
- **Reliability:** Multiple channels ensure alerts are received even if one fails
- **User preference:** Different users prefer different notification methods
- **Urgency:** Different channels for different alert levels

**Channel selection rationale:**

**Email (SMTP):**
- **Why:** Universal, reliable, easy to implement
- **Use case:** Standard alerts, non-urgent notifications
- **Implementation:** Simple SMTP integration, HTML formatting for readability

**Webhook (POST to URL):**
- **Why:** Flexible, integrates with Slack, Discord, custom systems
- **Use case:** Team notifications, integration with existing systems
- **Implementation:** Simple HTTP POST with JSON payload

**SMS (Twilio or similar):**
- **Why:** High-priority alerts, reaches users even when not online
- **Use case:** Critical alerts only (due to cost)
- **Implementation:** Requires SMS service (Twilio, AWS SNS) - may be skipped for capstone

**Push notification (Web Push API):**
- **Why:** Real-time notifications for dashboard users
- **Use case:** Users actively monitoring dashboard
- **Implementation:** Browser-based push notifications - more complex, may be skipped

**Capstone recommendation:**
- **Start with email:** Simplest to implement, sufficient for demo
- **Add webhook if time allows:** Shows integration capability
- **Skip SMS/push:** More complex, not essential for capstone demo

#### 4.2.4 Historical Analysis

**What it does:** Analyzes past fire detections to identify patterns and trends.

**Why this matters:**
- **Academic value:** Demonstrates data analysis and visualization capabilities
- **Real-world relevance:** Historical analysis is important for fire management
- **System completeness:** Shows system can do more than just detection

**Detection history:**
- **What it is:** Database of all past detections with timestamps, locations, areas
- **Why it matters:** Enables temporal analysis, pattern identification
- **Storage:** Simple database (SQLite for capstone) or CSV files

**Time series:**
- **What it is:** Fire area over time for a specific region
- **Why it matters:** Shows fire evolution, identifies trends
- **Visualization:** Line chart showing area vs time

**Seasonal patterns:**
- **What it is:** Aggregated statistics by month/season (e.g., summer has more fires)
- **Why it matters:** Identifies fire season, helps with resource planning
- **Analysis:** Group detections by month/season, calculate statistics

**Comparison:**
- **What it is:** Current year vs historical average
- **Why it matters:** Identifies unusual fire activity, contextualizes current situation
- **Calculation:** Compare current year statistics to multi-year averages

**Capstone simplification:**
- **Basic history:** Store detections with timestamps, locations, areas
- **Simple visualization:** Time series chart, basic statistics
- **Skip complex analysis:** Seasonal patterns and comparisons may be omitted if time-constrained

### 4.2 Advanced Features

#### 4.2.1 Fire Spread Detection (Multi-temporal Analysis)

| Feature | Description |
|---------|-------------|
| Temporal comparison | Compare detections from two dates |
| Spread area | New fire areas (in T2 but not T1) |
| Contained area | Fire areas that are no longer active (in T1 but not T2) |
| Spread direction | Vector indicating predominant spread direction |
| Spread rate | Hectares per day (if timestamps available) |
| Spread alert | Boolean flag if fire is actively spreading |

#### 4.2.2 Alert System

**Alert Levels:**

| Level | Criteria | Color |
|-------|----------|-------|
| LOW | < 1 hectare, not spreading | 🟡 Yellow |
| MEDIUM | 1-10 hectares, or small but spreading | 🟠 Orange |
| HIGH | 10-100 hectares, or medium and spreading | 🔴 Red |
| CRITICAL | > 100 hectares, or large and rapidly spreading | ⚫ Black |

**Alert Content:**
- Timestamp (UTC)
- Location (coordinates, nearest municipality)
- Fire area (hectares)
- Alert level
- Spread status
- Link to dashboard view

#### 4.2.3 Notification Channels

| Channel | Implementation |
|---------|----------------|
| Email | SMTP integration, HTML formatted alerts |
| Webhook | POST to user-configured URL (Slack, Discord, custom) |
| SMS | Twilio or similar (optional, higher priority alerts only) |
| Push notification | Web push API for dashboard users |

#### 4.2.4 Historical Analysis

| Feature | Description |
|---------|-------------|
| Detection history | Database of all past detections |
| Time series | Fire area over time for a region |
| Seasonal patterns | Aggregated statistics by month/season |
| Comparison | Current year vs historical average |

---

## 5. API Design (Simplified for Capstone)

API design defines how external systems (web UI, mobile apps, other services) interact with the wildfire detection system. For a capstone project, the API should be simple but demonstrate core functionality.

**Design Philosophy:**

- **Simplified for capstone:** Focus on demonstrating model inference, not production-ready features
- **RESTful principles:** Use standard HTTP methods and status codes
- **Clear contracts:** Well-defined request/response schemas
- **Documentation:** Automatic API documentation for easy testing

---

### 5.1 Technology Choice

**Framework: FastAPI**

**Why FastAPI?**

| Framework | Pros | Cons | Verdict |
|-----------|------|------|---------|
| **FastAPI** | ✅ Async support<br>✅ Auto OpenAPI docs<br>✅ Type validation<br>✅ High performance<br>✅ Easy to learn | ⚠️ Python-only | **⭐ Recommended** |
| Flask | ✅ Simple<br>✅ Widely used | ⚠️ No async (slower for I/O)<br>⚠️ Manual validation<br>⚠️ Manual docs | Good for simple APIs |
| Django REST | ✅ Full framework<br>✅ Admin panel | ⚠️ Overkill for simple API<br>⚠️ More complex | Too heavy for capstone |

**Rationale for FastAPI:**

**Async support:**
- **Why it matters:** Fetching satellite imagery is I/O-bound (waiting for external APIs)
- **Performance:** Async allows handling multiple requests efficiently
- **Real-world relevance:** Demonstrates modern API design patterns

**Automatic OpenAPI documentation:**
- **What it is:** Interactive API documentation (Swagger UI) generated automatically
- **Why it matters:** Easy testing, clear API contract, professional appearance
- **Capstone value:** Demonstrates API design best practices

**Type hints and Pydantic validation:**
- **What it is:** Request/response validation using Python type hints
- **Why it matters:** Catches errors early, clear API contracts, better developer experience
- **Academic value:** Shows understanding of modern Python practices

**High performance:**
- **Why it matters:** Fast response times improve user experience
- **Benchmark:** FastAPI is one of the fastest Python frameworks
- **Capstone note:** Performance is less critical for demo, but good practice

**Simple to learn:**
- **Why it matters:** Reduces learning curve, more time for model work
- **Documentation:** Excellent documentation, many examples
- **Community:** Large community, easy to find help

---

### 5.2 Simplified Endpoint Summary

**Why Simplified?**

For capstone, the API should demonstrate core functionality without production complexity. Focus is on model inference, not enterprise features.

**Simplified Features Rationale:**

**Single synchronous endpoint:**
- **Why:** Simplest to implement and understand
- **Alternative:** Async jobs for long-running tasks (more complex, not needed for capstone)
- **Trade-off:** Synchronous means users wait for response (acceptable for demo)

**Basic health check:**
- **Why:** Essential for deployment monitoring, simple to implement
- **What it checks:** API is running, model is loaded, dependencies available
- **Use case:** Deployment platforms (HuggingFace Spaces) use health checks

**Optional authentication:**
- **Why optional:** Simplifies demo, no user management needed
- **If implemented:** Simple API key in header (no database, no user accounts)
- **Production note:** Real systems need proper authentication, but not for capstone

**Removed complex features:**
- **Alert subscriptions:** Requires database, user management - too complex for capstone
- **Spread analysis:** Can be done client-side or omitted
- **History tracking:** Requires database - can be simplified or omitted

#### 5.2.1 Core Detection Endpoints

**POST `/api/v1/detect`**

**What it does:** Detects fires in satellite imagery for a given bounding box and date.

**Why POST:**
- **Request body:** Bounding box and date are complex parameters (not simple query params)
- **Semantic correctness:** POST is appropriate for operations that process data
- **Future extensibility:** Easy to add more parameters (threshold, bands, etc.)

**Why `/api/v1/`:**
- **Versioning:** Allows future API versions without breaking changes
- **Best practice:** Standard REST API versioning pattern
- **Capstone note:** v1 is sufficient, but shows understanding of API design

**GET `/api/v1/health`**

**What it does:** Returns API health status.

**Why GET:**
- **Idempotent:** Health check doesn't change state
- **Cacheable:** Health status can be cached (though typically not)
- **Standard:** Health checks are typically GET endpoints

**Response fields:**
- **Status:** "healthy" or "unhealthy"
- **Model loaded:** Whether PyTorch model is loaded
- **Dependencies:** Status of external services (Sentinel API, etc.)
- **Timestamp:** When health check was performed

---

### 5.3 Request/Response Schemas

**Why Defined Schemas?**

Clear request/response schemas ensure API contracts are well-defined, enable automatic validation, and make API easier to use and test.

#### Detection Request

**Bounding Box (bbox):**

**Why bounding box:**
- **Flexible region selection:** Users can analyze any geographic region
- **Standard format:** Bounding boxes are standard in GIS and remote sensing
- **Simple to specify:** Four coordinates (min_lon, min_lat, max_lon, max_lat)

**Coordinate system:** WGS84 (EPSG:4326) - standard web coordinate system

**Validation:**
- **Range checks:** Longitude [-180, 180], Latitude [-90, 90]
- **Order checks:** min_lon < max_lon, min_lat < max_lat
- **Size limits:** Maximum bounding box size (e.g., 1° × 1°) to prevent excessive processing

**Date:**

**Why ISO format (YYYY-MM-DD):**
- **Standard:** ISO 8601 is international standard for dates
- **Unambiguous:** No confusion about date format (MM/DD vs DD/MM)
- **Sortable:** ISO dates sort correctly as strings
- **Validation:** Easy to validate format and range

**Date constraints:**
- **Past dates only:** Can't detect fires in future (satellite imagery is historical)
- **Minimum date:** Sentinel-2 data available from 2015
- **Maximum date:** Typically up to 2-3 days ago (processing delay)

**Threshold:**

**Why optional:**
- **Default value:** 0.5 is reasonable default (balanced precision/recall)
- **User control:** Advanced users may want to adjust threshold
- **Flexibility:** Allows experimentation without code changes

**Range:** 0.0 to 1.0 (probability range)

#### Detection Response

**Fire detected (boolean):**

**Why boolean:**
- **Quick check:** Users can quickly determine if any fire was found
- **Simple logic:** Enables conditional behavior (alert if true, skip if false)
- **Clear semantics:** Binary outcome is easy to understand

**Total area (hectares):**

**Why hectares:**
- **Standard unit:** Fire area is typically reported in hectares
- **Meaningful scale:** Appropriate for fire sizes
- **User familiarity:** Emergency services use hectares

**Fire areas (GeoJSON array):**

**Why GeoJSON:**
- **Standard format:** Widely supported by web maps and GIS software
- **Self-contained:** Includes coordinates and metadata in single format
- **Web-friendly:** JSON format works well with web applications

**Why array:**
- **Multiple fires:** Single image may contain multiple fire events
- **Individual polygons:** Each fire gets its own polygon for detailed analysis
- **Flexibility:** Can handle any number of fires

**Processing time (seconds):**

**Why include:**
- **Performance monitoring:** Helps identify slow requests
- **User feedback:** Users know how long operation took
- **Debugging:** Helps identify performance bottlenecks
- **Academic value:** Demonstrates understanding of performance metrics

---

### 5.4 Authentication (Optional)

**Why Optional for Capstone?**

Authentication adds complexity (user management, tokens, security) that isn't essential for demonstrating model inference. For capstone demo, authentication can be skipped.

**If Implemented: Simple API Key**

**Why API key (not OAuth/JWT):**
- **Simplicity:** API key is simplest authentication method
- **Sufficient for demo:** Provides basic access control
- **No user management:** No need for user accounts, passwords, registration
- **Easy to implement:** Single header check, no complex token validation

**Implementation:**
- **Header:** `X-API-Key: your-api-key-here`
- **Storage:** Environment variable or simple config file (not database)
- **Validation:** Check API key matches configured key(s)

**Production note:** Real systems need proper authentication (OAuth, JWT, user management), but API key is sufficient for capstone demo.

---

### 5.5 Error Codes

**Why Standard HTTP Codes?**

Using standard HTTP status codes ensures API follows REST conventions, is easy to understand, and works well with HTTP clients and tools.

**200 Success:**
- **When:** Request processed successfully, fires detected or not detected
- **Why:** Standard success code for GET/POST requests
- **Response:** Always includes detection results (even if no fires found)

**400 Bad Request:**
- **When:** Invalid parameters (invalid bbox, invalid date format, out of range values)
- **Why:** Client error - user provided invalid input
- **Response:** Error message explaining what was wrong

**404 Not Found:**
- **When:** No satellite imagery available for requested date/region
- **Why:** Resource not found - imagery doesn't exist
- **Alternative:** Could use 400 (bad request), but 404 is more semantically correct

**500 Internal Server Error:**
- **When:** Server-side error (model loading failed, processing error, bug)
- **Why:** Server error - not user's fault
- **Response:** Generic error message (don't expose internal details)

**503 Service Unavailable:**
- **When:** External service down (Sentinel API unavailable, model not loaded)
- **Why:** Service temporarily unavailable - different from 500 (permanent error)
- **Response:** Error message indicating service is temporarily unavailable

**Error Response Format:**
- **Consistent structure:** All errors use same JSON format
- **Error code:** HTTP status code
- **Error message:** Human-readable error description
- **Error details:** Optional additional information (for 400 errors, list invalid fields)

---

## 6. User Interface (Simplified for Capstone)

The user interface (UI) is how end users interact with the wildfire detection system. For a capstone project, the UI should be functional and demonstrate model capabilities without excessive complexity.

**UI Design Philosophy:**

- **Simplified for capstone:** Focus on demonstrating model inference, not production UI/UX
- **Functional over beautiful:** Working demo is more important than polished design
- **Time-efficient:** Choose technology that minimizes development time
- **Model-focused:** UI serves to showcase model, not be the main feature

---

### 6.1 Technology Stack - Simplified Options

**Why Simplified Options?**

For capstone, UI technology should minimize learning curve and development time, allowing more focus on model development and evaluation.

#### Option 1: Streamlit (Recommended)

**Why Streamlit is Recommended:**

**Python-based:**
- **No JavaScript needed:** Team can use Python (same language as model)
- **Reduced learning curve:** No need to learn JavaScript, HTML, CSS
- **Consistency:** Same language for model, API, and UI

**Easy PyTorch integration:**
- **Direct model loading:** Can load PyTorch model directly in Streamlit app
- **No API needed:** Can call model directly (though API is still recommended for separation)
- **Simple inference:** Model inference code works directly in Streamlit

**Built-in map support:**
- **Folium/streamlit-folium:** Python libraries for interactive maps
- **No external services:** Don't need Mapbox/Google Maps API keys
- **GeoJSON support:** Easy to display fire polygons on maps

**Fastest to implement:**
- **Rapid prototyping:** Can create working UI in hours, not days
- **Less code:** Streamlit handles much of the UI complexity
- **Time savings:** More time for model work, less time on UI

**Limitations:**
- **Less flexible:** Limited customization compared to custom HTML/JS
- **Performance:** May be slower for complex interactions
- **Styling:** Limited control over appearance (acceptable for capstone)

**When to use:** **⭐ Recommended for capstone** - best balance of simplicity and functionality.

#### Option 2: Simple HTML/JavaScript

**Why consider this option:**

**Minimal code:**
- **Simple structure:** Basic HTML, minimal JavaScript
- **No frameworks:** Vanilla JavaScript is sufficient for simple demo
- **Lightweight:** Fast loading, simple deployment

**Leaflet.js for maps:**
- **Open source:** Free, no API keys needed
- **Well-documented:** Extensive documentation and examples
- **GeoJSON support:** Easy to display fire polygons

**Direct API calls:**
- **Simple integration:** Fetch API for HTTP requests
- **No backend complexity:** UI just calls API endpoint
- **Clear separation:** UI and model/API are separate

**Limitations:**
- **JavaScript required:** Team needs JavaScript knowledge
- **More code:** More code than Streamlit for same functionality
- **Manual styling:** Need to write CSS for appearance

**When to use:** If team has JavaScript experience and prefers separation of concerns (UI vs API).

#### Option 3: React/Next.js

**Why not recommended for capstone:**

**More complex:**
- **Learning curve:** Requires React knowledge, component architecture
- **More code:** More boilerplate and setup code
- **Build process:** Requires build tools, deployment configuration

**Time-consuming:**
- **Development time:** Takes longer to implement than Streamlit
- **Less time for model:** Time spent on UI is time not spent on model
- **Overkill:** More features than needed for capstone demo

**When to use:** Only if team has prior React/Next.js experience and wants to demonstrate full-stack skills.

**Recommendation:** Use Streamlit for capstone - focus time on model, not UI complexity.

---

### 6.2 Simplified UI Structure (Streamlit Example)

**Why This Structure?**

The UI structure should enable users to run fire detection and view results with minimal complexity. Each component serves a specific purpose in the detection workflow.

**Main Page Components:**

**Map display:**
- **What it is:** Interactive map showing region of interest and fire detections
- **Why it matters:** Visual representation is essential for understanding fire locations
- **Technology:** Folium or streamlit-folium (Python map libraries)
- **Features:** Zoom, pan, click to get coordinates, display fire polygons

**Date picker:**
- **What it is:** User selects date for satellite imagery analysis
- **Why it matters:** Different dates show different fire situations
- **Constraints:** Past dates only (satellite imagery is historical)
- **Default:** Recent date (e.g., 7 days ago) or user-specified

**Bounding box input:**
- **What it is:** User specifies geographic region to analyze
- **Why it matters:** Users may want to analyze specific regions (e.g., Catalonia)
- **Options:**
  - **Text input:** Enter coordinates (min_lon, min_lat, max_lon, max_lat)
  - **Draw on map:** Click and drag to draw bounding box (more user-friendly)
  - **Preset regions:** Dropdown with predefined regions (Catalonia, specific fire areas)

**"Run Detection" button:**
- **What it does:** Triggers fire detection for selected region and date
- **Why it matters:** Clear action trigger, user controls when detection runs
- **Feedback:** Show loading indicator while processing (detection takes 10-30 seconds)

**Results display:**
- **What it shows:** Fire polygons on map, statistics (total area, fire count)
- **Why it matters:** Users need to see detection results and understand fire characteristics
- **Components:**
  - **Map overlay:** Fire polygons displayed on map
  - **Statistics panel:** Total area (hectares), number of fires, processing time
  - **Confidence visualization:** Optional - color-code polygons by confidence

**Export button:**
- **What it does:** Downloads fire polygons as GeoJSON file
- **Why it matters:** Users may want to use results in GIS software or other tools
- **Format:** GeoJSON (standard format, works with QGIS, ArcGIS, web maps)

**Minimal Features Rationale:**

**Basic map with fire overlays:**
- **Why minimal:** Complex map features (layers, styles, controls) add complexity
- **Sufficient:** Basic map with fire polygons is enough to demonstrate functionality
- **Focus:** Map serves to visualize fires, not be a full GIS application

**Simple controls:**
- **Why simple:** Complex controls (multiple date ranges, advanced filters) add development time
- **Sufficient:** Date, region, threshold are core controls needed for detection
- **User-friendly:** Simple controls are easier to understand and use

**Results display:**
- **Why essential:** Users must see detection results to understand system functionality
- **Format:** Map visualization + statistics table is clear and informative
- **Sufficient:** Detailed analysis can be done in exported GeoJSON

**Export functionality:**
- **Why important:** Enables users to use results in other tools (GIS, analysis)
- **Simple implementation:** File download is straightforward in Streamlit
- **Standard format:** GeoJSON is universal format for geographic data

**Skipped Complex Features:**

**Spread analysis UI:**
- **Why skip:** Requires multi-date selection, temporal comparison logic
- **Complexity:** Adds significant development time
- **Alternative:** Can be done in exported data or omitted for capstone

**Alert management:**
- **Why skip:** Requires user accounts, alert configuration, notification setup
- **Complexity:** Adds database, authentication, notification infrastructure
- **Alternative:** Show alert generation capability without full management UI

**Historical analysis:**
- **Why skip:** Requires database, time series visualization, statistical analysis
- **Complexity:** Adds significant backend and frontend complexity
- **Alternative:** Show single detection capability, mention historical analysis as future work

**Complex navigation:**
- **Why skip:** Single-page application is sufficient for capstone demo
- **Complexity:** Multi-page navigation adds routing, state management
- **Alternative:** All features on single page (scroll or tabs if needed)

**Focus:** Working demo that shows model inference, not production UI.

**UI Development Checklist:**
- [ ] Set up Streamlit project structure
- [ ] Implement map component with folium/streamlit-folium
- [ ] Add date picker (past dates only)
- [ ] Add bounding box input (text or draw on map)
- [ ] Add "Run Detection" button with loading indicator
- [ ] Implement results display (map overlay + statistics)
- [ ] Add export functionality (GeoJSON download)
- [ ] Test end-to-end workflow (select region → detect → view results → export)
- [ ] Basic styling and polish (make it presentable, not perfect)

---

## 7. Deployment (Simplified for Capstone)

Deployment makes the wildfire detection system accessible to users via the internet or local network. For a capstone project, deployment should be simple, free, and demonstrate the working system.

**Deployment Philosophy:**

- **Simplified for capstone:** Focus on getting system online, not production infrastructure
- **Free options:** Use free tiers to avoid costs
- **Time-efficient:** Choose deployment method that minimizes setup time
- **Demonstration-focused:** Goal is to show working system, not enterprise deployment

---

### 7.1 Simplified Deployment Options

**Why Simplified Options?**

For capstone, deployment should be straightforward and require minimal infrastructure knowledge. Complex deployment (Kubernetes, cloud orchestration) is unnecessary.

#### Option 1: HuggingFace Spaces (Recommended)

**Why HuggingFace Spaces is Recommended:**

**Free hosting:**
- **No cost:** Free tier sufficient for capstone demo
- **Academic-friendly:** Designed for ML projects and demos
- **No credit card:** No payment required

**Easy deployment:**
- **Git-based:** Push code to Git repository, automatic deployment
- **No infrastructure:** No servers, containers, or configuration needed
- **Fast setup:** Can deploy in minutes, not hours

**Built-in model hosting:**
- **Model storage:** Can store model files in Space or HuggingFace model hub
- **No external storage:** Don't need S3, Google Drive, or other storage services
- **Version control:** Model versions can be tracked in HuggingFace

**Framework support:**
- **FastAPI:** Can deploy FastAPI applications
- **Streamlit:** Can deploy Streamlit applications
- **Gradio:** Alternative UI framework (if preferred)

**Zero infrastructure management:**
- **No servers:** HuggingFace manages all infrastructure
- **No scaling:** Automatic scaling handled by platform
- **No monitoring:** Basic monitoring provided (can add custom if needed)

**Limitations:**
- **Resource limits:** Free tier has CPU/memory limits (sufficient for demo)
- **Customization:** Less control over environment compared to self-hosted
- **Dependencies:** Must use supported Python packages

**When to use:** **⭐ Recommended for capstone** - best balance of simplicity and functionality.

#### Option 2: Streamlit Cloud

**Why consider Streamlit Cloud:**

**Streamlit-specific:**
- **Optimized for Streamlit:** Designed specifically for Streamlit apps
- **Simple deployment:** Automatic deployment from GitHub
- **Fast:** Very quick to deploy Streamlit apps

**Free hosting:**
- **No cost:** Free tier for public repositories
- **Academic-friendly:** Good for academic projects

**Limitations:**
- **Streamlit only:** Can't deploy FastAPI or other frameworks
- **Less flexible:** More limited than HuggingFace Spaces
- **GitHub required:** Must use GitHub (not GitLab, Bitbucket)

**When to use:** If using Streamlit UI and want simplest possible deployment.

#### Option 3: Local Docker Container

**Why consider local deployment:**

**No cloud required:**
- **Offline demo:** Can demo without internet connection
- **No external dependencies:** Everything runs locally
- **Full control:** Complete control over environment

**Simple setup:**
- **Dockerfile:** Single Dockerfile defines entire environment
- **Reproducible:** Same environment on any machine with Docker
- **Portable:** Can run on any OS (Windows, Mac, Linux)

**Limitations:**
- **Not accessible online:** Can't share via URL (unless using port forwarding)
- **Requires Docker:** Users need Docker installed
- **Local only:** Not suitable for remote demos or grading

**When to use:** For in-person demos, presentations, or when internet is unreliable.

#### Option 4: Google Colab

**Why consider Google Colab:**

**Jupyter environment:**
- **Familiar interface:** Jupyter notebooks are familiar to students
- **Free GPU:** Can use free GPU for inference (if available)
- **Easy sharing:** Share via link, no deployment needed

**Limitations:**
- **Not a web app:** Not suitable for interactive web interface
- **Session-based:** Sessions timeout, not persistent
- **Limited UI:** Can't create full web interface like Streamlit

**When to use:** For quick inference demos or model testing, not full system deployment.

---

### 7.2 Recommended Approach: HuggingFace Spaces

**Why HuggingFace Spaces?**

HuggingFace Spaces provides the best balance of simplicity, functionality, and academic-friendliness for capstone projects.

**Advantages:**
- **Simplest deployment:** Git push → automatic deployment
- **Free:** No cost for academic projects
- **Academic-focused:** Designed for ML demos and research projects
- **Time-efficient:** Minimal setup time, more time for model work
- **Professional appearance:** Clean URLs, good for presentations

**Setup Process:**

**1. Create HuggingFace account:**
- **Free signup:** No credit card required
- **Academic email:** Use university email if available (may get additional benefits)
- **Verification:** Email verification required

**2. Create new Space:**
- **Space type:** Choose FastAPI or Streamlit (depending on UI choice)
- **Visibility:** Public (required for free tier) or Private (if available)
- **SDK:** Choose Python SDK

**3. Upload model and code:**
- **Git integration:** Connect GitHub repository or upload files directly
- **Model files:** Upload PyTorch model (.pt file) to Space or model hub
- **Code:** Upload API/UI code, requirements.txt, README

**4. Automatic deployment:**
- **Build process:** HuggingFace automatically builds and deploys
- **URL:** Get public URL for your deployed system
- **Updates:** Push to Git → automatic redeployment

**Storage Strategy:**

**Model files:**
- **Option 1:** HuggingFace model hub (recommended) - versioned, shareable
- **Option 2:** Space storage - simpler, but less organized
- **Size limits:** Free tier has storage limits (typically sufficient for single model)

**No database needed:**
- **In-memory:** Use in-memory storage for demo (detections, alerts)
- **Skip persistence:** For capstone demo, persistence is optional
- **Alternative:** SQLite file if persistence needed (simple, no external service)

**No Redis/cache needed:**
- **Simplified:** Skip caching for capstone demo
- **Time savings:** Caching adds complexity without essential benefit
- **Future work:** Can add caching if time allows

**Deployment Checklist:**
- [ ] Create HuggingFace account
- [ ] Create new Space (FastAPI or Streamlit)
- [ ] Upload model to model hub or Space storage
- [ ] Upload code (API/UI, requirements.txt)
- [ ] Test deployed system (health check, detection endpoint)
- [ ] Document deployment URL and access instructions
- [ ] Create demo video/screenshots of deployed system

---

### 7.3 Alternative: Simple Cloud Deployment (If Needed)

**When to Consider Alternatives:**

If HuggingFace Spaces doesn't meet requirements (e.g., specific framework needs, resource limits, or team preference), consider these alternatives.

**Component-by-Component Approach:**

**Frontend:**
- **Streamlit Cloud:** Simplest if using Streamlit (free, automatic deployment)
- **Vercel (free tier):** If using HTML/JavaScript frontend (good for static sites)
- **Why these:** Free, simple, sufficient for capstone demo

**API/Model:**
- **HuggingFace Spaces:** Still recommended for API deployment
- **Railway (free tier):** Alternative if HuggingFace doesn't work (simple deployment, Docker-based)
- **Why these:** Free tiers, simple deployment, no infrastructure management

**Database:**
- **SQLite (file-based):** Simplest database option (no external service needed)
- **Skip database:** For capstone demo, database may not be needed
- **Why SQLite:** No setup required, works for simple persistence needs

**Storage:**
- **Local files:** Store files in deployment environment (simplest)
- **HuggingFace storage:** Use HuggingFace model hub or Space storage
- **Why not S3/Cloud Storage:** Adds complexity and potential costs, not needed for demo

**Cost Analysis:**

**Total cost: $0**
- All recommended services have free tiers
- Free tiers are sufficient for capstone demo (low traffic, single user)
- No hidden costs or credit card required

**When to use alternatives:**
- **Specific requirements:** If HuggingFace Spaces doesn't support needed framework/features
- **Team preference:** If team has experience with specific platform
- **Resource needs:** If free tier limits are insufficient (unlikely for capstone)

---

## 8. Testing & Validation

Testing and validation ensure the system works correctly, performs well, and meets requirements. For a capstone project, comprehensive testing demonstrates system reliability and academic rigor.

**Testing Philosophy:**

- **Comprehensive but practical:** Test all components without excessive complexity
- **Model-focused:** Prioritize model validation (most important for capstone)
- **End-to-end:** Ensure entire pipeline works together
- **Documentation:** Test results are essential for capstone report

---

### 8.1 Testing Levels

**Why Multiple Testing Levels?**

Different testing levels catch different types of issues. Unit tests find code bugs, integration tests find system issues, and model validation ensures ML performance.

#### 8.1.1 Unit Tests

Unit tests verify individual components work correctly in isolation. Each function or class is tested independently.

**Why Unit Tests Matter:**

- **Early bug detection:** Catch bugs before integration
- **Confidence:** Know each component works before combining
- **Refactoring safety:** Can change code knowing tests will catch regressions
- **Documentation:** Tests serve as examples of how components work

**Component Testing Rationale:**

**Preprocessing:**
- **Normalization:** Verify values are in correct range (0-1), no NaN/Inf
- **Resampling:** Verify resolution changes are correct, no data loss
- **Patching:** Verify patches are correct size, no boundary issues
- **Why test:** Preprocessing bugs cause model failures (wrong input format)

**Model:**
- **Forward pass:** Verify model runs without errors, produces output
- **Output shapes:** Verify output shape matches expected (batch, channels, height, width)
- **Output range:** Verify output values are in correct range (0-1 for probabilities)
- **Why test:** Model bugs cause inference failures or incorrect predictions

**Postprocessing:**
- **Polygon extraction:** Verify polygons are valid GeoJSON, correct coordinates
- **Area calculation:** Verify area calculations are correct (test with known areas)
- **Coordinate conversion:** Verify WGS84 ↔ UTM conversions are accurate
- **Why test:** Postprocessing bugs cause incorrect results even if model is correct

**API:**
- **Endpoint validation:** Verify endpoints accept valid requests, reject invalid ones
- **Error handling:** Verify correct error codes and messages for various error cases
- **Response format:** Verify responses match schema, correct data types
- **Why test:** API bugs prevent users from using the system

**Unit Testing Scope for Capstone:**
- **Focus on critical paths:** Test main functionality, not every edge case
- **Model and preprocessing:** Most important - these affect ML performance
- **API basics:** Test main endpoints, basic error handling
- **Skip exhaustive testing:** Don't need 100% coverage for capstone

#### 8.1.2 Integration Tests

Integration tests verify that multiple components work together correctly. They test the interaction between components, not individual components.

**Why Integration Tests Matter:**

- **System behavior:** Individual components may work but fail when combined
- **Data flow:** Verify data flows correctly through entire pipeline
- **Error propagation:** Verify errors are handled correctly across components
- **Real-world simulation:** Tests simulate actual usage scenarios

**Integration Test Rationale:**

**End-to-end detection:**
- **What it tests:** Full pipeline from API request → imagery fetch → model inference → postprocessing → response
- **Why it matters:** Most realistic test - if this works, system works
- **What to verify:** Correct response format, reasonable processing time, no crashes

**Sentinel API integration:**
- **What it tests:** Fetching imagery from Sentinel API, handling API errors, processing API responses
- **Why it matters:** External API is critical dependency - must handle failures gracefully
- **What to verify:** Successful fetch, error handling (rate limits, no data available), timeout handling

**Database (if used):**
- **What it tests:** Storing and retrieving detections, alerts, user data
- **Why it matters:** Database bugs cause data loss or incorrect retrieval
- **What to verify:** CRUD operations work, data integrity, query correctness

**Alerts (if implemented):**
- **What it tests:** Detection triggers alert, alert is formatted correctly, notification is sent
- **Why it matters:** Alert system must work reliably (critical for real-world use)
- **What to verify:** Alert generation, notification delivery, alert content correctness

**Integration Testing Scope for Capstone:**
- **Focus on main workflows:** Test primary use cases (detection, results display)
- **Skip complex scenarios:** Don't need to test every possible combination
- **Mock external services:** Use mocks for Sentinel API to avoid rate limits and costs

#### 8.1.3 Model Validation

Model validation tests the trained model's performance on unseen data. This is the most important testing for a capstone project focused on ML.

**Why Model Validation Matters:**

- **Performance verification:** Ensures model meets success criteria (Section 1.4)
- **Generalization:** Tests if model works on new data (not just training data)
- **Failure analysis:** Identifies where model fails (edge cases, geographic biases)
- **Academic requirement:** Essential for capstone report - must demonstrate model quality

**Model Validation Test Rationale:**

**Test set metrics:**
- **What it tests:** Model performance on held-out test set (never seen during training)
- **Why it matters:** Most important validation - tests true generalization
- **Metrics:** IoU, Dice, Precision, Recall (see Section 3.6)
- **Targets:** Must meet success criteria (IoU ≥ 0.70, etc.)

**Geographic generalization:**
- **What it tests:** Model performance on Catalonia (target region) vs training regions
- **Why it matters:** Tests transfer learning effectiveness - model must work on Catalonia
- **Comparison:** Compare Catalonia metrics vs general test set metrics
- **Analysis:** If performance drops on Catalonia, investigate geographic bias

**Temporal generalization:**
- **What it tests:** Model performance on recent fires (not in training data)
- **Why it matters:** Tests if model works on new fires (temporal generalization)
- **Data:** Use fires from 2024-2025 (after training data cutoff)
- **Analysis:** Identifies if model is overfitted to training time period

**Edge cases:**
- **What it tests:** Model performance on challenging scenarios (clouds, smoke, urban areas, water)
- **Why it matters:** Real-world data has edge cases - model must handle them
- **Scenarios:**
  - **Clouds:** Heavy cloud cover may obscure fires
  - **Smoke:** Smoke plumes may be confused with fires or obscure fires
  - **Urban areas:** Hot surfaces (rooftops, roads) may trigger false positives
  - **Water:** Water bodies should never be detected as fires
- **Analysis:** Document failure modes, identify improvement areas

**Model Validation Scope for Capstone:**
- **Comprehensive:** Model validation is most important - allocate significant time
- **Multiple test sets:** Test set, Catalonia validation set, edge case examples
- **Detailed analysis:** Error analysis, confusion matrices, visual examples
- **Documentation:** All validation results must be in capstone report

---

### 8.2 Catalonia Validation Set (Mandatory)

**Why Catalonia Validation Set is Mandatory?**

The project focuses on Catalonia, so the model must be validated specifically on Catalonia data. This tests transfer learning and ensures the model works on the target region.

**Why This Matters:**

- **Project focus:** System is designed for Catalonia - must validate on Catalonia
- **Transfer learning test:** Tests if model generalizes from training regions to Catalonia
- **Academic requirement:** Demonstrates real-world applicability
- **Geographic bias detection:** Identifies if model is biased toward training regions

**Creation Process Rationale:**

**1. Obtain fire perimeters:**
- **Source:** Generalitat de Catalunya (official government data)
- **Time period:** 2023-2024 fires (recent, not in training data)
- **Format:** CSV with fire perimeters (coordinates, dates, areas)
- **Why this source:** Authoritative, official data ensures ground truth quality

**2. Download Sentinel-2 imagery:**
- **Matching dates:** Download imagery for dates when fires occurred
- **Region:** Catalonia bounding box (covers all fire locations)
- **Product:** Sentinel-2 L2A (atmospherically corrected)
- **Why match dates:** Need imagery from fire dates to create accurate ground truth

**3. Create ground truth masks:**
- **Method:** Convert fire perimeters (polygons) to raster masks
- **Resolution:** Match model input resolution (10-20m)
- **Alignment:** Ensure masks align with Sentinel-2 imagery (same coordinate system)
- **Why raster masks:** Model expects raster masks for training/evaluation

**4. Evaluate model:**
- **Metrics:** Calculate IoU, Dice, Precision, Recall on Catalonia validation set
- **Comparison:** Compare with general test set metrics
- **Analysis:** Identify performance differences, explain transfer learning effectiveness

**Requirements Rationale:**

**Mandatory (not optional):**
- **Project requirement:** System is for Catalonia - must validate on Catalonia
- **Academic rigor:** Demonstrates thorough validation approach
- **Real-world relevance:** Shows system works on target region

**Document creation process:**
- **Reproducibility:** Others must be able to recreate Catalonia validation set
- **Transparency:** Shows how ground truth was created from government data
- **Academic documentation:** Required for capstone report

**Report separate metrics:**
- **Comparison:** Catalonia vs general test set performance
- **Analysis:** Why performance differs (if it does) - geographic bias, data distribution
- **Transfer learning:** Demonstrates if model transfers well to Catalonia

**Include in final report:**
- **Essential component:** Catalonia validation is key project deliverable
- **Demonstrates applicability:** Shows system works on target region
- **Academic value:** Demonstrates understanding of transfer learning and validation

---

### 8.3 Performance Benchmarks

Performance benchmarks measure system speed and capacity. They ensure the system is fast enough for practical use and can handle expected load.

**Why Performance Benchmarks Matter:**

- **Usability:** Slow systems frustrate users
- **Real-world feasibility:** System must be fast enough for practical use
- **Resource planning:** Helps determine hardware requirements
- **Academic value:** Demonstrates understanding of system performance

**Benchmark Targets Rationale:**

**Single tile inference (< 5 seconds):**
- **What it measures:** Time to process one 256×256 patch through model
- **Why this target:** Fast enough for interactive use, reasonable for GPU inference
- **Factors:** Model size, GPU vs CPU, batch size
- **Why important:** Users expect responsive system

**API response uncached (< 30 seconds):**
- **What it measures:** Time from API request to response (including imagery fetch, inference, postprocessing)
- **Why this target:** Acceptable for capstone demo (not production requirement)
- **Breakdown:** Imagery fetch (10-20s) + inference (5s) + postprocessing (2-5s)
- **Why important:** Total user wait time must be reasonable

**API response cached (< 2 seconds):**
- **What it measures:** Response time when imagery is cached (no fetch needed)
- **Why this target:** Fast response for repeated requests
- **Caching:** Cache recently fetched imagery to avoid repeated API calls
- **Why important:** Improves user experience for repeated analyses

**Concurrent requests (10 simultaneous):**
- **What it measures:** System can handle multiple requests at once
- **Why this target:** Reasonable for capstone demo (not production scale)
- **Testing:** Send 10 simultaneous requests, verify all complete successfully
- **Why important:** Tests system stability and resource handling

**Performance Benchmarking Scope:**
- **Focus on main operations:** Test detection endpoint, not every operation
- **Realistic scenarios:** Test with typical bounding box sizes, not edge cases
- **Document results:** Include performance metrics in capstone report

---

### 8.4 User Acceptance Testing

User acceptance testing (UAT) verifies that the system is usable and meets user needs. It tests the system from an end-user perspective, not a technical perspective.

**Why User Acceptance Testing Matters:**

- **Usability verification:** Ensures system is actually usable, not just functional
- **User perspective:** Tests what users care about (not technical details)
- **Demo preparation:** Identifies issues before capstone presentation
- **Academic value:** Demonstrates system is complete and usable

**UAT Test Rationale:**

**Detection workflow:**
- **What it tests:** User can select region, select date, run detection, view results
- **Why it matters:** Core functionality - if this doesn't work, system is unusable
- **Success criteria:** User can complete workflow without confusion or errors
- **Testing:** Have someone unfamiliar with system try to use it

**Results display:**
- **What it tests:** Fire polygons visible on map, statistics displayed clearly
- **Why it matters:** Users must be able to see and understand results
- **Success criteria:** Results are clear, map is readable, statistics are understandable
- **Testing:** Verify polygons render correctly, map is interactive, statistics are accurate

**Alert subscription (if implemented):**
- **What it tests:** User can subscribe to alerts and receive test alert
- **Why it matters:** Alert system must work for users who want notifications
- **Success criteria:** Subscription works, alert is received, alert content is correct
- **Testing:** Subscribe, trigger test detection, verify alert delivery

**Export functionality:**
- **What it tests:** GeoJSON downloads correctly and opens in QGIS (or other GIS software)
- **Why it matters:** Users may want to use results in other tools
- **Success criteria:** File downloads, opens in QGIS, data is correct
- **Testing:** Download GeoJSON, open in QGIS, verify polygons and attributes

**Mobile usability:**
- **What it tests:** Dashboard is usable on mobile phone (responsive design)
- **Why it matters:** Users may access system from mobile devices
- **Success criteria:** Interface is readable, controls are usable, map is interactive
- **Testing:** Open dashboard on phone, test main workflows

**UAT Scope for Capstone:**
- **Focus on core workflows:** Test main use cases, not every feature
- **Real users:** Have someone unfamiliar with system test it (catches usability issues)
- **Document issues:** Note any usability problems, even if not fixed
- **Demo preparation:** UAT helps prepare for capstone presentation

---

## 9. Project Timeline

### 9.1 Phase Overview

| Phase | Duration | Focus | Key Deliverables |
|-------|----------|-------|-----------------|
| Phase 1 | 3 weeks | Data preparation & exploration | Processed datasets, data quality report, Catalonia validation set |
| Phase 2 | 4 weeks | Model development & optimization | Trained model, baseline comparisons, hyperparameter tuning results |
| Phase 3 | 2 weeks | API development | Working FastAPI with detection endpoint |
| Phase 4 | 2 weeks | UI development | Streamlit dashboard with map and controls |
| Phase 5 | 1 week | Deployment | Deployed system on HuggingFace Spaces |
| Phase 6 | 2 weeks | Testing, validation & documentation | Comprehensive evaluation report, model card, documentation |

**Total: 14 weeks (includes buffer for unexpected delays)**

---

### 9.2 Detailed Implementation Plan

#### Phase 1: Data Preparation & Exploration (Weeks 1-3)

**Week 1: Dataset Acquisition & Setup**

**1.1 Environment & Tool Setup**
- [ ] Set up Python environment (conda/venv) with required packages
- [ ] Install core libraries: `torch`, `rasterio`, `geopandas`, `sentinelsat`, `pystac-client`, `segmentation_models_pytorch`, `wandb`, `optuna`
- [ ] Set up Git repository with proper `.gitignore`
- [ ] Create project directory structure (data/, models/, notebooks/, src/, tests/)
- [ ] Set up Weights & Biases (W&B) account and create project
- [ ] Configure W&B project settings (entity, project name, tags)

**1.2 Primary Dataset Download**
- [ ] Download CEMS-Wildfire dataset from HuggingFace (primary dataset)
  - [ ] Extract and organize dataset files
  - [ ] Verify dataset structure and annotations
  - [ ] Document dataset version and source
- [ ] Download EO4WildFires dataset (secondary, large-scale dataset)
  - [ ] Extract relevant subsets if full dataset is too large
  - [ ] Document which events/subset are used
- [ ] Download Sen2Fire or Land8Fire dataset (additional diversity)
  - [ ] Choose based on availability and download speed
  - [ ] Document selection rationale

**1.3 Data Versioning Setup**
- [ ] Set up external storage for large datasets (Google Drive, university storage, or cloud bucket)
- [ ] Create data versioning strategy document
- [ ] Set up Git LFS for small files (CSV metadata, config files, small GeoJSON)
  - [ ] Configure `.gitattributes` for Git LFS
  - [ ] Initialize Git LFS in repository
- [ ] Create manual versioning system for large files (checksums, version logs)
- [ ] Document data storage locations and access methods

**Week 2: Data Preprocessing & Quality Exploration**

**2.1 Preprocessing Pipeline Implementation**
- [ ] Implement Sentinel-2 band selection (B2, B3, B4, B8, B8A, B11, B12)
- [ ] Implement resolution harmonization (resample all bands to 10m or 20m)
- [ ] Implement normalization (per-band min-max or z-score normalization)
- [ ] Implement patch extraction (256×256 or 512×512 patches)
- [ ] Implement coordinate system handling (WGS84 ↔ UTM conversions)
- [ ] Test preprocessing pipeline on sample images
- [ ] Document preprocessing parameters and choices

**2.2 Data Quality & Exploration (Comprehensive)**
- [ ] Create data exploration notebook template
- [ ] **Class distribution analysis:**
  - [ ] Calculate fire vs non-fire pixel ratios per dataset
  - [ ] Visualize class distribution histograms
  - [ ] Document class imbalance severity
- [ ] **Geographic distribution analysis:**
  - [ ] Map fire event locations on world map
  - [ ] Identify geographic coverage gaps
  - [ ] Document regional distribution
- [ ] **Temporal distribution analysis:**
  - [ ] Plot fire events over time (monthly/yearly)
  - [ ] Identify temporal coverage and gaps
- [ ] **Cloud cover statistics:**
  - [ ] Calculate cloud cover percentage per image
  - [ ] Identify heavily clouded images
  - [ ] Document cloud filtering strategy
- [ ] **Band value ranges:**
  - [ ] Calculate min/max/mean/std for each band
  - [ ] Identify outliers or anomalies
  - [ ] Document normalization ranges
- [ ] **Missing data handling:**
  - [ ] Check for NaN/Inf values
  - [ ] Document missing data strategy
- [ ] **Ground truth alignment:**
  - [ ] Verify mask alignment with imagery
  - [ ] Check coordinate system consistency
- [ ] **Sample review:**
  - [ ] Visually inspect random samples for annotation errors
  - [ ] Document any quality issues found
- [ ] **Coordinate system validation:**
  - [ ] Verify all data uses consistent CRS
  - [ ] Document coordinate system choices
- [ ] **Duplicate detection:**
  - [ ] Check for duplicate patches
  - [ ] Remove duplicates if found
- [ ] Create data quality report with findings and recommendations

**2.3 Catalonia Validation Data Preparation (Mandatory)**
- [ ] Obtain 2023-2024 fire perimeters from Generalitat de Catalunya
  - [ ] Download official fire perimeter data (CSV/Shapefile)
  - [ ] Document data source and access method
- [ ] Download corresponding Sentinel-2 imagery for fire dates
  - [ ] Use `sentinelsat` or Copernicus Data Space API
  - [ ] Download L2A products for Catalonia region
  - [ ] Match imagery dates with fire occurrence dates
- [ ] Create ground truth masks from fire perimeters
  - [ ] Convert polygon perimeters to raster masks
  - [ ] Align masks with Sentinel-2 imagery (same CRS, resolution)
  - [ ] Verify mask quality and alignment
- [ ] Document Catalonia validation set creation process
- [ ] Store Catalonia validation set separately (not in training data)

**Week 3: Dataset Splitting & Augmentation Setup**

**3.1 Dataset Split Strategy**
- [ ] Implement geographic validation split (no spatial overlap between train/val/test)
- [ ] Create train/validation/test splits (70/15/15 or 80/10/10)
  - [ ] Ensure no temporal leakage (train on older data, test on newer)
  - [ ] Ensure geographic separation (different regions in each split)
- [ ] Verify split statistics (class distribution, geographic coverage)
- [ ] Document split methodology and rationale
- [ ] Save split indices/metadata for reproducibility

**3.2 Class Imbalance Handling Setup**
- [ ] Analyze class distribution in training set
- [ ] Implement loss weighting strategy
  - [ ] Calculate class weights based on inverse frequency
  - [ ] Configure weighted BCE loss
- [ ] Implement weighted sampling (if using)
  - [ ] Create weighted sampler for DataLoader
- [ ] Implement patch filtering (remove patches with <5% fire pixels if needed)
- [ ] Document class imbalance strategy and chosen approach

**3.3 Data Augmentation Implementation**
- [ ] Implement geometric augmentations (rotation, flipping, scaling)
- [ ] Implement spectral augmentations (brightness, contrast, noise)
- [ ] Implement spatial augmentations (translation, elastic deformation)
- [ ] Test augmentation pipeline on sample patches
- [ ] Configure augmentation parameters (probability, ranges)
- [ ] Document augmentation strategy and parameters

**3.4 Data Pipeline Finalization**
- [ ] Create PyTorch Dataset class for training data
- [ ] Create PyTorch Dataset class for validation/test data
- [ ] Test data loading pipeline (verify shapes, types, augmentations)
- [ ] Measure data loading performance
- [ ] Document final data pipeline architecture

**Phase 1 Deliverables:**
- ✅ Processed and organized datasets
- ✅ Data quality exploration notebook with visualizations
- ✅ Data versioning system set up
- ✅ Catalonia validation set created and documented
- ✅ Train/val/test splits with geographic validation
- ✅ Data preprocessing pipeline implemented and tested
- ✅ Class imbalance handling strategy implemented
- ✅ Data augmentation pipeline implemented
- ✅ Data quality report document

---

#### Phase 2: Model Development & Optimization (Weeks 4-7)

**Week 4: Baseline Models & Initial Architecture**

**4.1 Baseline Model Implementation**
- [ ] **Threshold-based baseline:**
  - [ ] Implement SWIR-based threshold detection (B11, B12)
  - [ ] Implement NBR (Normalized Burn Ratio) threshold
  - [ ] Tune threshold values on validation set
  - [ ] Evaluate on test set (IoU, Dice, Precision, Recall)
  - [ ] Document threshold values and performance
- [ ] **Random Forest baseline:**
  - [ ] Implement feature extraction (spectral bands, indices: NBR, NDVI, BAI)
  - [ ] Train Random Forest classifier
  - [ ] Tune hyperparameters (n_estimators, max_depth, min_samples_split)
  - [ ] Evaluate on test set (comprehensive metrics)
  - [ ] Document feature importance and performance
- [ ] Create baseline comparison report
- [ ] Visualize baseline predictions vs ground truth (example images)

**4.2 U-Net Architecture Setup**
- [ ] Set up `segmentation_models_pytorch` library
- [ ] Implement U-Net with pretrained encoder (ResNet-34 or EfficientNet-B0)
- [ ] Configure model input/output (7-channel input, 1-channel binary output)
- [ ] Test model forward pass (verify output shapes)
- [ ] Calculate model parameters and size
- [ ] Document architecture choice and rationale

**4.3 Training Infrastructure Setup**
- [ ] Configure PyTorch DataLoader:
  - [ ] Set `num_workers` (4-8 for CPU, 2-4 per GPU)
  - [ ] Set `pin_memory=True` if using GPU
  - [ ] Set `prefetch_factor=2-4`
  - [ ] Set `persistent_workers=True`
  - [ ] Configure `shuffle=True` for training, `False` for validation
- [ ] Set up model checkpointing strategy:
  - [ ] Implement best model checkpointing (based on validation IoU)
  - [ ] Implement periodic checkpointing (every N epochs)
  - [ ] Configure checkpoint directory structure
- [ ] Set up W&B logging:
  - [ ] Log hyperparameters, training metrics, validation metrics
  - [ ] Log sample predictions (images) periodically
  - [ ] Configure W&B project settings

**Week 5: Initial Training & Loss Function Selection**

**5.1 Loss Function Implementation**
- [ ] Implement combined BCE + Dice Loss
- [ ] Implement Focal Loss (as alternative)
- [ ] Configure loss weights (BCE weight: 0.5, Dice weight: 0.5)
- [ ] Test loss functions on sample predictions
- [ ] Document loss function choice and rationale

**5.2 Optimizer & Learning Rate Setup**
- [ ] Configure AdamW optimizer:
  - [ ] Set learning rate (initial: 1e-4 or 3e-4)
  - [ ] Set weight decay (1e-4)
  - [ ] Set betas (0.9, 0.999)
- [ ] Configure learning rate schedule:
  - [ ] Implement ReduceLROnPlateau (recommended)
  - [ ] Set patience (5-10 epochs), factor (0.5), min_lr (1e-6)
- [ ] Document optimizer and schedule choices

**5.3 Initial Training Run**
- [ ] Train baseline U-Net model for 20-30 epochs
- [ ] Monitor training/validation loss and metrics
- [ ] Log to W&B (loss curves, metrics, sample predictions)
- [ ] Evaluate on validation set after training
- [ ] Analyze training curves (overfitting, convergence)
- [ ] Save best model checkpoint

**Week 6: Hyperparameter Tuning**

**6.1 Optuna Setup**
- [ ] Set up Optuna study for hyperparameter optimization
- [ ] Define search space:
  - [ ] Learning rate: [1e-5, 1e-3] (log scale)
  - [ ] Encoder: ['resnet34', 'efficientnet-b0']
  - [ ] Batch size: [8, 16, 32] (based on GPU memory)
  - [ ] Loss weights: BCE [0.3, 0.7], Dice [0.3, 0.7]
  - [ ] Dropout: [0.0, 0.3] (if using)
- [ ] Configure Optuna study (direction: maximize validation IoU)
- [ ] Set up pruning (MedianPruner or HyperbandPruner)

**6.2 Hyperparameter Tuning Execution**
- [ ] Run Optuna optimization (20-50 trials)
- [ ] Monitor trials in Optuna dashboard
- [ ] Let trials run (may take 1-2 days depending on resources)
- [ ] Analyze hyperparameter importance (Optuna feature importance)
- [ ] Select best hyperparameters based on validation IoU
- [ ] Document tuning results and best hyperparameters

**6.3 Final Model Training**
- [ ] Train final model with best hyperparameters
- [ ] Train for full epoch budget (50-100 epochs) with early stopping
- [ ] Monitor training closely (W&B dashboard)
- [ ] Save best model checkpoint (based on validation IoU)
- [ ] Export final model for deployment (PyTorch .pt format)

**Week 7: Model Evaluation & Analysis**

**7.1 Comprehensive Model Evaluation**
- [ ] Evaluate final model on test set:
  - [ ] Calculate primary metrics (IoU, Dice, Precision, Recall)
  - [ ] Calculate secondary metrics (Pixel Accuracy, FPR, Detection Latency)
  - [ ] Verify metrics meet success criteria (Section 1.4)
- [ ] **Evaluate on Catalonia validation set (Mandatory):**
  - [ ] Calculate same metrics on Catalonia set
  - [ ] Compare Catalonia performance vs general test set
  - [ ] Analyze transfer learning effectiveness
  - [ ] Document performance differences
- [ ] Create confusion matrices (per-patch, per-event, aggregate)
- [ ] Perform threshold optimization (ROC and PR curves)
- [ ] Document all evaluation results

**7.2 Error Analysis**
- [ ] Identify common failure modes:
  - [ ] False positives (clouds, urban areas, water)
  - [ ] False negatives (small fires, smoke-obscured fires)
- [ ] Create visual examples of failures
- [ ] Analyze geographic biases (performance by region)
- [ ] Analyze temporal biases (performance by season)
- [ ] Document error analysis findings

**7.3 Baseline Comparison**
- [ ] Compare final model with baseline models:
  - [ ] Threshold-based baseline
  - [ ] Random Forest baseline
- [ ] Create comparison table (all metrics)
- [ ] Visual comparison (same examples, all methods)
- [ ] Document why deep learning outperforms (or investigate if it doesn't)

**7.4 Model Documentation**
- [ ] Create model card:
  - [ ] Model architecture and hyperparameters
  - [ ] Training data description
  - [ ] Performance metrics (test set, Catalonia validation)
  - [ ] Limitations and failure modes
  - [ ] Intended use and recommendations
- [ ] Create training report:
  - [ ] Hyperparameter tuning results
  - [ ] Training curves (loss, metrics over epochs)
  - [ ] Validation results
  - [ ] Best hyperparameters and rationale
- [ ] Export model in deployment format (PyTorch .pt)

**Phase 2 Deliverables:**
- ✅ Baseline models implemented and evaluated
- ✅ U-Net model trained and optimized
- ✅ Hyperparameter tuning completed (Optuna results)
- ✅ Final model evaluated on test set and Catalonia validation set
- ✅ Comprehensive evaluation report with metrics and error analysis
- ✅ Model card and training report
- ✅ Model exported for deployment

---

#### Phase 3: API Development (Weeks 8-9)

**Week 8: FastAPI Setup & Core Endpoints**

**8.1 FastAPI Project Setup**
- [ ] Create FastAPI project structure
- [ ] Set up virtual environment and install dependencies
- [ ] Create `requirements.txt` with all dependencies
- [ ] Set up project configuration (config files, environment variables)
- [ ] Create basic FastAPI app structure

**8.2 Model Loading & Inference**
- [ ] Implement model loading function (load PyTorch .pt model)
- [ ] Implement inference function:
  - [ ] Preprocess input imagery (normalization, patching)
  - [ ] Run model forward pass
  - [ ] Postprocess output (threshold, polygon extraction)
- [ ] Test inference on sample imagery
- [ ] Measure inference performance (latency)

**8.3 Sentinel-2 Integration**
- [ ] Set up Copernicus Data Space API access (or Sentinel Hub API)
- [ ] Implement imagery fetching function:
  - [ ] Query Sentinel-2 products by bounding box and date
  - [ ] Download or stream imagery
  - [ ] Handle API errors (rate limits, no data available)
- [ ] Test imagery fetching with sample requests
- [ ] Implement basic caching (optional, for demo)

**8.4 Core Endpoints Implementation**
- [ ] Implement `POST /api/v1/detect` endpoint:
  - [ ] Request validation (bbox, date, optional threshold)
  - [ ] Fetch Sentinel-2 imagery
  - [ ] Run model inference
  - [ ] Postprocess results (polygon extraction, area calculation)
  - [ ] Return response (fire detected, total area, fire areas GeoJSON, processing time)
- [ ] Implement `GET /api/v1/health` endpoint:
  - [ ] Check API status
  - [ ] Check model loaded status
  - [ ] Check external dependencies (Sentinel API)
  - [ ] Return health status
- [ ] Test endpoints with sample requests
- [ ] Implement error handling (400, 404, 500 errors)

**Week 9: API Testing & Documentation**

**9.1 API Testing**
- [ ] Write unit tests for inference function
- [ ] Write unit tests for preprocessing/postprocessing
- [ ] Write integration tests for endpoints
- [ ] Test error handling (invalid requests, API failures)
- [ ] Test end-to-end pipeline (request → response)
- [ ] Measure API performance (response times)

**9.2 API Documentation**
- [ ] Set up automatic OpenAPI documentation (FastAPI generates this)
- [ ] Test Swagger UI (interactive API docs)
- [ ] Write basic API usage guide
- [ ] Document request/response schemas
- [ ] Document error codes and messages

**9.3 Optional Features**
- [ ] Implement simple authentication (API key) if needed
- [ ] Add request logging
- [ ] Add basic rate limiting (if needed for demo)

**Phase 3 Deliverables:**
- ✅ FastAPI application with detection and health endpoints
- ✅ Sentinel-2 imagery integration
- ✅ Model inference pipeline integrated
- ✅ API tested and documented
- ✅ OpenAPI documentation available

---

#### Phase 4: User Interface Development (Weeks 10-11)

**Week 10: Streamlit Setup & Core Components**

**10.1 Streamlit Project Setup**
- [ ] Create Streamlit project structure
- [ ] Install Streamlit and dependencies (`folium`, `geopandas`, etc.)
- [ ] Set up Streamlit app entry point (`app.py` or `main.py`)
- [ ] Configure Streamlit settings (page title, layout)

**10.2 Map Component**
- [ ] Implement interactive map using `folium`:
  - [ ] Set default view (Catalonia region)
  - [ ] Enable map interaction (pan, zoom)
  - [ ] Add bounding box selection tool (draw rectangle on map)
- [ ] Test map component functionality

**10.3 Detection Controls**
- [ ] Implement date picker (select date for imagery)
- [ ] Implement bounding box input (from map selection or manual input)
- [ ] Implement threshold slider (adjust detection threshold)
- [ ] Implement "Run Detection" button
- [ ] Add loading indicator during detection
- [ ] Test controls and user interaction

**Week 11: Results Display & Export**

**11.1 Results Display**
- [ ] Implement fire polygon overlay on map:
  - [ ] Display detected fire polygons on map
  - [ ] Color-code polygons (e.g., red for fires)
  - [ ] Add polygon tooltips (area, confidence)
- [ ] Implement statistics display:
  - [ ] Total fire area (hectares)
  - [ ] Number of fire events
  - [ ] Detection confidence
  - [ ] Processing time
- [ ] Implement confidence map visualization (optional)
- [ ] Test results display with sample detections

**11.2 Export Functionality**
- [ ] Implement GeoJSON download:
  - [ ] Convert fire polygons to GeoJSON format
  - [ ] Add download button
  - [ ] Test GeoJSON opens correctly in QGIS
- [ ] Test export functionality

**11.3 UI Polish & Testing**
- [ ] Basic styling (colors, fonts, layout)
- [ ] Improve UI/UX (clear labels, helpful messages)
- [ ] Test complete user workflow:
  - [ ] Select region and date
  - [ ] Run detection
  - [ ] View results on map
  - [ ] Export GeoJSON
- [ ] Test on different screen sizes (responsive design)
- [ ] Create UI screenshots for documentation

**Phase 4 Deliverables:**
- ✅ Streamlit dashboard with interactive map
- ✅ Detection controls (date picker, bbox selection, threshold)
- ✅ Results display (fire polygons on map, statistics)
- ✅ GeoJSON export functionality
- ✅ Polished UI ready for demo

---

#### Phase 5: Deployment (Week 12)

**12.1 HuggingFace Spaces Setup**
- [ ] Create HuggingFace account (if not already created)
- [ ] Create new Space (choose FastAPI or Streamlit SDK)
- [ ] Configure Space settings (visibility, SDK, hardware)
- [ ] Set up Git repository connection (or prepare files for upload)

**12.2 Model & Code Deployment**
- [ ] Upload model to HuggingFace model hub (or Space storage):
  - [ ] Upload PyTorch model (.pt file)
  - [ ] Document model version and metadata
- [ ] Upload code to Space:
  - [ ] API code (if FastAPI) or Streamlit app (if Streamlit)
  - [ ] `requirements.txt` with all dependencies
  - [ ] `README.md` with setup instructions
  - [ ] Configuration files
- [ ] Test deployment build process
- [ ] Verify deployment succeeds

**12.3 Deployment Testing**
- [ ] Test deployed system:
  - [ ] Health check endpoint works
  - [ ] Detection endpoint works with sample request
  - [ ] UI loads and functions correctly (if Streamlit)
- [ ] Test with real Catalonia region and recent date
- [ ] Verify performance (response times acceptable)
- [ ] Document deployment URL and access instructions

**12.4 Demo Preparation**
- [ ] Create demo video:
  - [ ] Record screen showing complete workflow
  - [ ] Show detection on Catalonia region
  - [ ] Show results and export
  - [ ] Keep video concise (3-5 minutes)
- [ ] Create demo screenshots (key features, results)
- [ ] Document deployment process (for reproducibility)

**Phase 5 Deliverables:**
- ✅ System deployed on HuggingFace Spaces
- ✅ Deployed system tested and working
- ✅ Demo video and screenshots created
- ✅ Deployment documentation

---

#### Phase 6: Testing, Validation & Documentation (Weeks 13-14)

**Week 13: Comprehensive Testing**

**13.1 Unit Testing**
- [ ] Write unit tests for preprocessing functions:
  - [ ] Normalization, resampling, patching
  - [ ] Coordinate conversions
- [ ] Write unit tests for model:
  - [ ] Forward pass, output shapes, output ranges
- [ ] Write unit tests for postprocessing:
  - [ ] Polygon extraction, area calculation
- [ ] Write unit tests for API:
  - [ ] Endpoint validation, error handling, response format
- [ ] Run unit test suite, fix any failures

**13.2 Integration Testing**
- [ ] Test end-to-end detection pipeline:
  - [ ] API request → imagery fetch → inference → postprocessing → response
- [ ] Test Sentinel API integration:
  - [ ] Successful fetch, error handling, timeout handling
- [ ] Test database operations (if implemented)
- [ ] Test alert system (if implemented)
- [ ] Document integration test results

**13.3 Model Validation (Final)**
- [ ] Final evaluation on test set:
  - [ ] Calculate all metrics (IoU, Dice, Precision, Recall, FPR, Pixel Accuracy)
  - [ ] Create confusion matrices
  - [ ] Perform threshold optimization
- [ ] Final evaluation on Catalonia validation set:
  - [ ] Calculate same metrics
  - [ ] Compare with test set performance
  - [ ] Document transfer learning effectiveness
- [ ] Edge case testing:
  - [ ] Test on clouds, smoke, urban areas, water
  - [ ] Document edge case performance
- [ ] Performance benchmarks:
  - [ ] Measure single tile inference time (< 5 seconds target)
  - [ ] Measure API response time (< 30 seconds uncached, < 2 seconds cached)
  - [ ] Test concurrent requests (10 simultaneous)
  - [ ] Document benchmark results

**13.4 User Acceptance Testing**
- [ ] Test detection workflow:
  - [ ] User can select region, date, run detection
  - [ ] Workflow is intuitive and error-free
- [ ] Test results display:
  - [ ] Fire polygons visible on map
  - [ ] Statistics displayed clearly
- [ ] Test export functionality:
  - [ ] GeoJSON downloads correctly
  - [ ] GeoJSON opens in QGIS
- [ ] Test mobile usability:
  - [ ] Dashboard usable on phone
- [ ] Document UAT results and any issues found

**Week 14: Documentation & Finalization**

**14.1 Comprehensive Evaluation Report**
- [ ] Create evaluation report with:
  - [ ] Test set metrics (all metrics, confusion matrices)
  - [ ] Catalonia validation set metrics (separate section)
  - [ ] Baseline comparison (threshold-based, Random Forest)
  - [ ] Error analysis (failure modes, visual examples)
  - [ ] Geographic and temporal analysis
  - [ ] Performance benchmarks
  - [ ] Threshold optimization results
- [ ] Include visualizations (curves, confusion matrices, example predictions)
- [ ] Document all findings and recommendations

**14.2 Model Documentation**
- [ ] Finalize model card:
  - [ ] Architecture, hyperparameters, training data
  - [ ] Performance metrics, limitations, intended use
- [ ] Finalize training report:
  - [ ] Hyperparameter tuning process and results
  - [ ] Training curves, validation results
  - [ ] Best hyperparameters and rationale

**14.3 Data Documentation**
- [ ] Document all datasets used:
  - [ ] Dataset names, sources, sizes, coverage
  - [ ] Download URLs and access methods
  - [ ] Dataset versions used
- [ ] Document preprocessing pipeline:
  - [ ] All preprocessing steps and parameters
  - [ ] Normalization ranges, patch sizes
- [ ] Document dataset splits:
  - [ ] Split methodology, statistics
  - [ ] Geographic and temporal distribution
- [ ] Document Catalonia validation set:
  - [ ] Creation process, data source
  - [ ] Ground truth creation methodology

**14.4 Reproducibility Guide**
- [ ] Create reproducibility guide:
  - [ ] Environment setup instructions
  - [ ] Data download and preparation steps
  - [ ] Model training instructions
  - [ ] How to reproduce results
  - [ ] Required software versions
- [ ] Document all configuration files and parameters
- [ ] Ensure code is well-commented and organized

**14.5 Final Bug Fixes & Polish**
- [ ] Fix any bugs found during testing
- [ ] Improve code quality (comments, documentation)
- [ ] Ensure all code is properly version controlled
- [ ] Create final project README with overview and setup

**14.6 Presentation Preparation**
- [ ] Prepare capstone presentation:
  - [ ] Project overview and objectives
  - [ ] Methodology (data, model, training)
  - [ ] Results (metrics, visualizations, comparisons)
  - [ ] Demo (live or video)
  - [ ] Conclusions and future work
- [ ] Practice presentation
- [ ] Prepare demo (live system or video)

**Phase 6 Deliverables:**
- ✅ Comprehensive evaluation report (test set, Catalonia validation, baselines, error analysis)
- ✅ Model card and training report
- ✅ Data documentation (datasets, preprocessing, splits)
- ✅ Reproducibility guide
- ✅ All testing completed (unit, integration, model validation, UAT)
- ✅ Performance benchmarks documented
- ✅ Final bug fixes and code polish
- ✅ Capstone presentation prepared

---

### 9.3 Task Dependencies & Critical Path

**Critical Path (Must Complete in Order):**
1. Data preparation → Model training → Model evaluation
2. Model training → API development → UI development
3. API + UI development → Deployment
4. All development → Testing & Documentation

**Parallel Work Opportunities:**
- Data exploration can happen while downloading additional datasets
- Baseline models can be implemented while setting up U-Net
- API documentation can be written while developing UI
- Some documentation can be written incrementally during development

**Key Milestones:**
- **End of Week 3:** Data pipeline complete, ready for training
- **End of Week 7:** Model trained and evaluated, ready for deployment
- **End of Week 11:** UI complete, ready for deployment
- **End of Week 12:** System deployed, ready for final testing
- **End of Week 14:** All deliverables complete, ready for presentation

---

### 9.4 Resource Requirements

**Computing Resources:**
- **GPU:** Recommended for model training (CUDA-capable GPU with 8GB+ VRAM)
  - Alternatives: Google Colab (free GPU), cloud GPU instances
- **Storage:** 100-200GB for datasets (external storage recommended)
- **Memory:** 16GB+ RAM recommended for data processing

**Software & Services:**
- **Free tier sufficient for capstone:**
  - HuggingFace Spaces (deployment)
  - Weights & Biases (experiment tracking)
  - Copernicus Data Space (Sentinel-2 imagery)
  - GitHub (code version control)

**Time Allocation:**
- **Data preparation:** ~25% of total time (critical foundation)
- **Model development:** ~35% of total time (core ML work)
- **API/UI development:** ~20% of total time (application layer)
- **Testing & documentation:** ~20% of total time (validation and reporting)

---

### 9.5 Risk Mitigation in Timeline

**Built-in Buffers:**
- Extra week in Phase 1 (data preparation often takes longer than expected)
- Extra week in Phase 2 (model training and tuning can be unpredictable)
- 2-week buffer in Phase 6 (testing and documentation always takes longer)

**Early Warning Signs:**
- **Week 3:** If data preparation is behind, prioritize core datasets only
- **Week 6:** If hyperparameter tuning is slow, reduce trial count (20 instead of 50)
- **Week 10:** If UI development is slow, simplify UI features
- **Week 13:** If testing reveals major issues, prioritize critical fixes only

**Contingency Plans:**
- **Data issues:** Use smaller subset of datasets, focus on CEMS-Wildfire
- **Model performance:** Extend training time, try different architectures
- **Deployment issues:** Use simpler deployment (local Docker) as backup
- **Time constraints:** Prioritize model quality over UI polish

---

## 10. Risks & Mitigations

Risk management identifies potential problems that could derail the project and develops strategies to prevent or mitigate them. For a capstone project, proactive risk management ensures timely completion and quality deliverables.

**Risk Management Philosophy:**

- **Proactive identification:** Identify risks early, before they become problems
- **Prioritization:** Focus on high-impact, high-probability risks
- **Practical mitigation:** Choose mitigation strategies that are feasible for capstone scope
- **Continuous monitoring:** Review risks regularly and adjust strategies as needed

---

### 10.1 Technical Risks

Technical risks are problems related to technology, tools, or implementation that could prevent the system from working correctly or meeting performance targets.

#### 10.1.1 Model Performance Risks

**Risk: Model accuracy below target (IoU < 0.70)**

**Impact:** High - Model performance is the primary success criterion for capstone

**Probability:** Medium - Deep learning performance can be unpredictable

**Why this risk exists:**
- Insufficient training data or poor data quality
- Model architecture not suitable for task
- Hyperparameters not optimized
- Class imbalance not handled properly
- Overfitting to training data

**Mitigation Strategies:**

**Prevention:**
- **Use proven architecture:** U-Net with pretrained encoder (Section 3.1) - well-established for segmentation
- **Comprehensive data exploration:** Identify and fix data quality issues early (Section 2.8)
- **Proper class imbalance handling:** Implement loss weighting, weighted sampling, or focal loss (Section 2.6)
- **Geographic validation split:** Ensure model generalizes, not just memorizes training regions (Section 2.5)
- **Catalonia validation set:** Test transfer learning early (Section 8.2) - catch geographic bias before final evaluation

**Response (if risk occurs):**
- **Iterate on data augmentation:** Increase augmentation diversity, adjust parameters
- **Hyperparameter tuning:** Run more Optuna trials, expand search space
- **Try different encoder:** Switch from ResNet-34 to EfficientNet-B0 or vice versa
- **Ensemble models:** Combine multiple models if single model insufficient
- **Fine-tune on Catalonia data:** If transfer learning fails, fine-tune on Catalonia validation set
- **Adjust success criteria:** If IoU consistently 0.65-0.70, document why and focus on other metrics

**Early warning signs:**
- Validation IoU plateaus below 0.65 after 20+ epochs
- Large gap between training and validation metrics (overfitting)
- Poor performance on Catalonia validation set (geographic bias)

---

**Risk: Model training instability (loss doesn't converge, NaN values)**

**Impact:** High - Training failure prevents model development

**Probability:** Low - Modern frameworks handle this well, but can occur

**Why this risk exists:**
- Learning rate too high
- Gradient explosion
- Numerical instability in loss function
- Data preprocessing errors (NaN/Inf values)

**Mitigation Strategies:**

**Prevention:**
- **Learning rate schedule:** Use ReduceLROnPlateau to automatically reduce LR (Section 3.2)
- **Gradient clipping:** Implement gradient clipping (max_norm=1.0) to prevent explosion
- **Data validation:** Check for NaN/Inf in preprocessing pipeline (Section 2.8)
- **Loss function stability:** Use combined BCE+Dice loss (more stable than Dice alone)
- **Mixed precision training:** Use automatic mixed precision (AMP) for numerical stability

**Response (if risk occurs):**
- **Reduce learning rate:** Lower initial LR (1e-5 instead of 1e-4)
- **Check data:** Verify no NaN/Inf in input data
- **Simplify model:** Temporarily use smaller model to test stability
- **Check loss function:** Verify loss calculation is correct

---

#### 10.1.2 Infrastructure & Integration Risks

**Risk: Sentinel API rate limits or access issues**

**Impact:** Medium - Prevents live data fetching, but can work with cached data

**Probability:** Medium - Free tier APIs often have rate limits

**Why this risk exists:**
- Copernicus Data Space API has rate limits for free accounts
- Sentinel Hub API requires account setup and may have limits
- Network issues or API downtime

**Mitigation Strategies:**

**Prevention:**
- **Implement caching:** Cache fetched imagery to avoid repeated API calls
- **Request queuing:** Queue requests and add delays between calls
- **Multiple API options:** Support both Copernicus Data Space and Sentinel Hub (fallback)
- **Test API access early:** Verify API access in Phase 1, not Phase 3
- **Use batch requests:** Request multiple dates/regions in single call when possible

**Response (if risk occurs):**
- **Use cached data:** For demo, use pre-downloaded imagery instead of live fetch
- **Reduce API calls:** Cache more aggressively, reuse imagery for multiple tests
- **Alternative data source:** Use downloaded Sentinel-2 imagery from training datasets
- **Simplify demo:** Show detection on pre-loaded imagery instead of live fetch

---

**Risk: Integration complexity (APIs, libraries don't work together)**

**Impact:** Medium - Delays development, but workarounds exist

**Probability:** Medium - Multiple libraries and APIs increase integration risk

**Why this risk exists:**
- Multiple Python libraries with version conflicts
- API authentication and setup complexity
- Coordinate system conversions between libraries
- Different data formats (raster, vector, GeoJSON)

**Mitigation Strategies:**

**Prevention:**
- **Use well-documented APIs:** Copernicus Data Space and Sentinel Hub have good documentation
- **Build incrementally:** Test each component separately before integrating
- **Version pinning:** Pin library versions in requirements.txt to avoid conflicts
- **Isolated testing:** Test API integration in separate script before adding to main code
- **Standard formats:** Use standard formats (GeoJSON, WGS84) to minimize conversions

**Response (if risk occurs):**
- **Simplify integration:** Use simpler library or manual implementation
- **Mock external services:** Use mock data for development, integrate later
- **Seek help:** Use library documentation, Stack Overflow, or community forums
- **Alternative libraries:** Switch to alternative library if primary doesn't work

---

**Risk: Deployment platform issues (HuggingFace Spaces)**

**Impact:** Medium - Can use alternative deployment, but adds time

**Probability:** Low - HuggingFace Spaces is stable, but can have issues

**Why this risk exists:**
- Platform downtime or maintenance
- Resource limits (CPU, memory, storage)
- Build failures due to dependencies or configuration

**Mitigation Strategies:**

**Prevention:**
- **Test deployment early:** Deploy minimal version early to test platform
- **Document dependencies:** Clear requirements.txt, test locally first
- **Resource monitoring:** Check resource usage, optimize if needed
- **Alternative ready:** Have backup deployment option (Streamlit Cloud, local Docker)

**Response (if risk occurs):**
- **Use alternative platform:** Switch to Streamlit Cloud or local Docker
- **Simplify deployment:** Reduce dependencies, use smaller model
- **Local demo:** Deploy locally for demo if cloud deployment fails

---

#### 10.1.3 Resource & Cost Risks

**Risk: Cloud costs exceed budget (if using paid services)**

**Impact:** Medium - May need to switch to free alternatives

**Probability:** Low - All recommended services have free tiers sufficient for capstone

**Why this risk exists:**
- Accidental use of paid services
- Exceeding free tier limits
- Misconfiguration leading to charges

**Mitigation Strategies:**

**Prevention:**
- **Use free tiers only:** HuggingFace Spaces, W&B, Copernicus Data Space all have free tiers
- **Set billing alerts:** If using any paid service, set up billing alerts
- **Monitor usage:** Check resource usage regularly
- **Avoid paid services:** Don't use AWS, GCP, Azure unless absolutely necessary

**Response (if risk occurs):**
- **Switch to free alternatives:** Use HuggingFace Spaces instead of paid cloud
- **Reduce usage:** Use smaller datasets, fewer API calls
- **Local resources:** Use local GPU or CPU instead of cloud

---

**Risk: GPU availability for training**

**Impact:** High - Training without GPU is very slow

**Probability:** Low - Multiple free GPU options available

**Why this risk exists:**
- Google Colab GPU quotas may be limited
- University GPU resources may be unavailable
- Local GPU may not be available

**Mitigation Strategies:**

**Prevention:**
- **Multiple GPU options:** Identify Google Colab, university resources, local GPU
- **Start training early:** Begin training as soon as data is ready
- **Optimize training:** Use smaller batch sizes, fewer epochs if GPU limited
- **CPU fallback:** Can train on CPU (slow but possible)

**Response (if risk occurs):**
- **Use Google Colab:** Free GPU available (may have quotas)
- **Reduce training:** Smaller model, fewer hyperparameter trials
- **CPU training:** Accept slower training, start earlier
- **University resources:** Request access to university GPU cluster

---

### 10.2 Data Risks

Data risks are problems related to datasets, data quality, or data availability that could prevent successful model training or evaluation.

#### 10.2.1 Data Quality & Generalization Risks

**Risk: Training data doesn't generalize to Catalonia**

**Impact:** High - Model must work on Catalonia (project requirement)

**Probability:** Medium - Geographic transfer learning can fail

**Why this risk exists:**
- Training data from different regions (not Catalonia)
- Different vegetation, climate, fire characteristics
- Geographic bias in training data

**Mitigation Strategies:**

**Prevention:**
- **Catalonia validation set (mandatory):** Create and test early (Section 8.2)
- **Geographic validation split:** Ensure test set includes diverse regions (Section 2.5)
- **Include Spanish data:** Use CEMS-Wildfire which includes Spanish fires
- **Data augmentation:** Augment with geographic diversity (rotation, scaling)
- **Monitor geographic performance:** Track performance by region during training

**Response (if risk occurs):**
- **Fine-tune on Catalonia data:** Fine-tune model on Catalonia validation set
- **Increase Catalonia data:** Add more Catalonia-specific training data
- **Transfer learning adjustments:** Adjust learning rate, freeze/unfreeze layers
- **Document limitations:** If performance lower on Catalonia, document why and analyze

---

**Risk: Dataset quality issues (annotation errors, misalignment)**

**Impact:** Medium - Poor data quality reduces model performance

**Probability:** Low - Datasets are well-curated, but errors can exist

**Why this risk exists:**
- Human annotation errors
- Misalignment between imagery and masks
- Incorrect coordinate systems
- Missing or corrupted data

**Mitigation Strategies:**

**Prevention:**
- **Thorough data exploration:** Comprehensive quality checks (Section 2.8)
  - Visual inspection of random samples
  - Check mask alignment with imagery
  - Verify coordinate systems
  - Check for missing data
- **Use authoritative sources:** CEMS-Wildfire from Copernicus (official source)
- **Validate preprocessing:** Verify preprocessing doesn't introduce errors
- **Document data issues:** Record any quality issues found

**Response (if risk occurs):**
- **Clean problematic data:** Remove or fix erroneous samples
- **Use alternative dataset:** Switch to different dataset if primary has issues
- **Manual correction:** Manually correct critical errors if feasible
- **Document limitations:** Note data quality issues in final report

---

#### 10.2.2 Data Availability Risks

**Risk: Cloud cover limits available imagery**

**Impact:** Medium - Reduces available training/evaluation data

**Probability:** Medium - Cloud cover is common in satellite imagery

**Why this risk exists:**
- Sentinel-2 imagery often has cloud cover
- Catalonia validation set may have limited clear imagery
- Live detection may encounter cloudy dates

**Mitigation Strategies:**

**Prevention:**
- **Cloud filtering:** Filter out heavily clouded images (Section 2.8)
- **Multiple dates:** Use multiple dates per fire event (not just single date)
- **Cloud masks:** Use provided cloud masks to exclude clouded pixels
- **Temporal flexibility:** Allow date range selection, not just single date

**Response (if risk occurs):**
- **Use alternative dates:** Select different dates with less cloud cover
- **Accept partial cloud:** Train/evaluate on partially clouded images
- **Cloud-aware evaluation:** Evaluate only on clear pixels (using cloud masks)
- **Document cloud impact:** Analyze and document how cloud cover affects performance

---

**Risk: Dataset download/access issues**

**Impact:** Medium - Delays data preparation phase

**Probability:** Low - Datasets are publicly available, but access can be slow

**Why this risk exists:**
- Large dataset sizes (slow downloads)
- HuggingFace dataset viewer limitations
- Network issues or server downtime
- Access restrictions or authentication issues

**Mitigation Strategies:**

**Prevention:**
- **Start downloads early:** Begin downloading datasets in Week 1
- **Multiple sources:** Have backup datasets ready (CEMS-Wildfire, EO4WildFires)
- **Prioritize core dataset:** Focus on CEMS-Wildfire first (primary dataset)
- **Use university network:** May have faster download speeds
- **Download in background:** Download while working on other tasks

**Response (if risk occurs):**
- **Use smaller subset:** Work with available data, download more later
- **Alternative dataset:** Switch to alternative dataset if primary unavailable
- **Manual download:** Use browser or wget if HuggingFace API fails
- **Extend Phase 1:** Add time buffer if downloads are slow

---

### 10.3 Schedule Risks

Schedule risks are problems that could cause project delays, preventing timely completion of deliverables.

#### 10.3.1 Development Timeline Risks

**Risk: Model training takes longer than expected**

**Impact:** Medium - Delays model development, but buffers included

**Probability:** Medium - Training time is hard to predict

**Why this risk exists:**
- Hyperparameter tuning requires many trials
- Model may need more epochs to converge
- GPU availability may be limited
- Iterative improvements require multiple training runs

**Mitigation Strategies:**

**Prevention:**
- **Start training early:** Begin as soon as data is ready (Week 4)
- **Use GPU:** Significantly faster than CPU training
- **Limit search space:** Focused hyperparameter ranges (Section 3.4)
- **Early stopping:** Stop training if no improvement (prevents wasted time)
- **Buffer time:** Extra week included in Phase 2 (4 weeks instead of 3)

**Response (if risk occurs):**
- **Reduce hyperparameter trials:** 20 trials instead of 50 if time limited
- **Use smaller model:** Faster training, may still meet targets
- **Parallel training:** Train multiple models simultaneously if resources allow
- **Accept current performance:** If close to target, document and proceed

---

**Risk: Data preparation takes longer than expected**

**Impact:** Medium - Delays entire project timeline

**Probability:** Medium - Data preparation is complex and time-consuming

**Why this risk exists:**
- Multiple datasets to download and process
- Data exploration is comprehensive (10 checklist items)
- Catalonia validation set creation is detailed
- Preprocessing pipeline is complex

**Mitigation Strategies:**

**Prevention:**
- **Extended Phase 1:** 3 weeks allocated (instead of 2)
- **Start immediately:** Begin data exploration while downloading
- **Prioritize core tasks:** Focus on essential datasets and tasks first
- **Parallel work:** Data exploration can happen during downloads
- **Incremental approach:** Process one dataset at a time, don't wait for all

**Response (if risk occurs):**
- **Simplify data exploration:** Focus on critical checks, skip less important ones
- **Use smaller datasets:** Work with subset if full dataset too large
- **Defer non-critical tasks:** Postpone advanced preprocessing if time limited
- **Extend Phase 1:** Use buffer from later phases if needed

---

**Risk: Learning curve for new tools (W&B, Optuna, FastAPI, Streamlit)**

**Impact:** Medium - Delays development, but tools are relatively simple

**Probability:** High - Team may not be familiar with all tools

**Why this risk exists:**
- Multiple new tools to learn (W&B, Optuna, FastAPI, Streamlit, geospatial libraries)
- Limited time to learn each tool
- Tool-specific issues or bugs

**Mitigation Strategies:**

**Prevention:**
- **Choose simple tools:** W&B, Streamlit are user-friendly (Section 3.5, Section 6.1)
- **Start early:** Set up W&B in Phase 1, learn tools incrementally
- **Good documentation:** All recommended tools have excellent documentation
- **Tutorials:** Use official tutorials and examples
- **Allocate extra time:** Phase 1 has extra week for learning

**Response (if risk occurs):**
- **Simplify tool usage:** Use basic features only, skip advanced features
- **Alternative tools:** Switch to simpler alternatives if needed
- **Seek help:** Use documentation, tutorials, Stack Overflow
- **Pair programming:** Team members can help each other learn

---

#### 10.3.2 Scope & Feature Risks

**Risk: Scope creep (adding too many features)**

**Impact:** Medium - Delays completion, reduces focus on ML work

**Probability:** Medium - Temptation to add features is common

**Why this risk exists:**
- Desire to make system more impressive
- Adding features seems easy but takes time
- Unclear priorities between ML and application features

**Mitigation Strategies:**

**Prevention:**
- **Clear priorities:** ML work is primary focus (Section 1.1)
- **Simplified features:** API and UI are simplified for capstone (Sections 5, 6)
- **Feature list:** Stick to planned features, defer others
- **Regular reviews:** Review progress weekly, catch scope creep early

**Response (if risk occurs):**
- **Defer non-essential features:** Postpone advanced features to "future work"
- **Simplify existing features:** Reduce complexity of planned features
- **Focus on ML:** Prioritize model quality over UI polish
- **Document deferred features:** Note what was planned but not implemented

---

**Risk: Testing and documentation takes longer**

**Impact:** Medium - Delays final deliverables, but buffers included

**Probability:** High - Testing and documentation always take longer than expected

**Why this risk exists:**
- Comprehensive testing is time-consuming
- Documentation requires careful writing
- Bug fixes discovered during testing
- Evaluation report is extensive

**Mitigation Strategies:**

**Prevention:**
- **Extended Phase 6:** 2 weeks allocated for testing and documentation
- **Incremental documentation:** Write documentation during development, not just at end
- **Early testing:** Start testing components as they're completed
- **Template preparation:** Prepare report templates early

**Response (if risk occurs):**
- **Prioritize critical documentation:** Focus on evaluation report, model card
- **Simplify testing:** Focus on essential tests, skip exhaustive edge cases
- **Accept minor bugs:** Fix critical bugs only, document known issues
- **Use templates:** Use provided templates to speed up documentation

---

### 10.4 Academic & Quality Risks

Academic risks are problems related to meeting capstone requirements, academic standards, or evaluation criteria.

#### 10.4.1 Evaluation & Success Criteria Risks

**Risk: Model doesn't meet success criteria (IoU < 0.70)**

**Impact:** High - Primary success criterion not met

**Probability:** Medium - Target is achievable but not guaranteed

**Why this risk exists:**
- Success criteria are ambitious but realistic
- Model performance can be unpredictable
- Data or training issues may prevent reaching target

**Mitigation Strategies:**

**Prevention:**
- **Proven architecture:** U-Net is well-established for segmentation
- **Comprehensive training:** Proper hyperparameter tuning, data augmentation
- **Early validation:** Test on validation set regularly during training
- **Catalonia validation:** Test transfer learning early (catch issues before final evaluation)

**Response (if risk occurs):**
- **Document thoroughly:** Explain why target wasn't met, what was tried
- **Focus on other metrics:** Highlight strong performance on other metrics (Dice, Precision, Recall)
- **Error analysis:** Comprehensive error analysis shows understanding despite lower IoU
- **Baseline comparison:** Show improvement over baselines (demonstrates value)
- **Academic discussion:** Discuss limitations, future improvements (shows critical thinking)

---

**Risk: Insufficient documentation for reproducibility**

**Impact:** Medium - Reduces academic rigor, but can be fixed

**Probability:** Low - Documentation is planned, but may be incomplete

**Why this risk exists:**
- Documentation is time-consuming
- May forget to document some steps
- Reproducibility requires detailed documentation

**Mitigation Strategies:**

**Prevention:**
- **Reproducibility guide:** Planned in Phase 6 (Section 9.2)
- **Incremental documentation:** Document as you go, not just at end
- **Version control:** Use Git to track code changes
- **Data versioning:** Document data versions and preprocessing (Section 2.7)
- **Configuration files:** Save all hyperparameters and configs

**Response (if risk occurs):**
- **Add missing documentation:** Fill gaps in reproducibility guide
- **Code comments:** Add comments to code explaining key steps
- **README:** Comprehensive README with setup instructions
- **Document known issues:** Note any reproducibility limitations

---

### 10.5 Risk Monitoring & Response

**How to Monitor Risks:**

**Weekly risk review:**
- Review project progress against timeline
- Identify new risks as they emerge
- Check early warning signs (Section 9.5)
- Adjust mitigation strategies if needed

**Key indicators to watch:**
- **Week 3:** Data preparation progress (should be 80% complete)
- **Week 6:** Model training progress (should have initial trained model)
- **Week 10:** UI development progress (should have working prototype)
- **Week 13:** Testing progress (should have most tests complete)

**When to escalate:**
- **High-impact risks:** If high-impact risk occurs, prioritize mitigation immediately
- **Cascading risks:** If one risk causes others, address root cause
- **Timeline delays:** If more than 1 week behind, adjust scope or timeline

**Response procedures:**

**Immediate response:**
1. Identify the risk that occurred
2. Assess impact and urgency
3. Implement mitigation strategy (from risk table)
4. Monitor effectiveness
5. Adjust if needed

**Documentation:**
- Document any risks that occurred
- Note how they were mitigated
- Include in final report if relevant
- Learn from risks for future projects

---

### 10.6 Risk Summary Table

| Risk Category | Highest Priority Risks | Mitigation Status |
|---------------|----------------------|-------------------|
| **Technical** | Model accuracy below target | ✅ Multiple prevention strategies, early validation |
| **Technical** | API rate limits | ✅ Caching, multiple API options |
| **Data** | Training data doesn't generalize | ✅ Catalonia validation set (mandatory), early testing |
| **Data** | Cloud cover limits imagery | ✅ Cloud filtering, multiple dates |
| **Schedule** | Training takes longer | ✅ Buffer time, GPU usage, early start |
| **Schedule** | Learning curve for tools | ✅ Simple tools chosen, extra time allocated |
| **Academic** | Success criteria not met | ✅ Proven architecture, comprehensive training |

**Overall Risk Assessment:**

**Project risk level: Medium**
- Most risks have effective mitigation strategies
- Buffers included in timeline
- Multiple alternatives for critical components
- Focus on ML work (reduces application complexity risks)

**Confidence in success: High**
- Well-planned timeline with buffers
- Proven technologies and architectures
- Comprehensive risk mitigation strategies
- Clear priorities and scope

---

## Appendix

### A. Resource Links

#### Datasets

| Resource | URL |
|----------|-----|
| CEMS-Wildfire Dataset (links-ads) | https://huggingface.co/datasets/links-ads/wildfires-cems |
| CEMS-Wildfire Dataset (9334hq) | https://huggingface.co/datasets/9334hq/wildfires-cems |
| CEMS-Wildfire GitHub | https://github.com/MatteoM95/CEMS-Wildfire-Dataset |
| CEMS-HLS (Alternative) | https://huggingface.co/datasets/morenoj11/CEMS-HLS |
| EO4WildFires Dataset | https://huggingface.co/datasets/AUA-Informatics-Lab/eo4wildfires |
| Sen2Fire Dataset | https://github.com/Orion-AI-Lab/Sen2Fire (check arXiv paper) |
| Sen2Fire Paper | https://arxiv.org/abs/2403.17884 |
| Land8Fire Dataset | https://www.mdpi.com/2072-4292/17/16/2776 |
| TS-SatFire Dataset | https://arxiv.org/abs/2412.11555 |
| S2-WCD Dataset | https://ieee-dataport.org/documents/sentinel-2-wildfire-change-detection-s2-wcd |
| FireSR Dataset | https://zenodo.org/records/13384289 |
| FLOGA Dataset | https://arxiv.org/abs/2311.03339 |
| Active Fire (Landsat-8) | https://arxiv.org/abs/2101.03409 |

#### Data Sources & APIs

| Resource | URL |
|----------|-----|
| Copernicus Data Space Browser | https://browser.dataspace.copernicus.eu/ |
| Copernicus Data Space API | https://documentation.dataspace.copernicus.eu/ |
| Sentinel Hub API | https://docs.sentinel-hub.com/ |
| Sentinel Hub Python | https://sentinelhub-py.readthedocs.io/ |
| sentinelsat Python Library | https://sentinelsat.readthedocs.io/ |
| pystac-client | https://pystac-client.readthedocs.io/ |
| NASA FIRMS | https://firms.modaps.eosdis.nasa.gov/ |
| NASA FIRMS API | https://firms.modaps.eosdis.nasa.gov/api/ |
| EFFIS Portal | https://effis.jrc.ec.europa.eu/ |
| EFFIS Active Fires | https://effis.jrc.ec.europa.eu/apps/effis.current-situation/active-fires |
| Eye on the Fire API | https://eyeonthefire.com/data-sources |
| Ambee Fire API | https://www.getambee.com/api/fire |
| Generalitat Fire Data | https://agricultura.gencat.cat/ca/ambits/medi-natural/incendis-forestals/ |
| Generalitat Fire Perimeters (CSV) | https://analisi.transparenciacatalunya.cat/api/views/bks7-dkfd/rows.csv?accessType=DOWNLOAD |
| Generalitat Civil Protection Risk Map | https://datos.gob.es/en/catalogo/a09002970-mapa-de-proteccion-civil-de-cataluna-riesgo-de-incendios-forestales |
| ICGC CatLC Dataset | https://www.icgc.cat/en/Geoinformation-and-Maps/Maps/Dataset-Land-cover-map-CatLC |
| ICGC CatLC FTP Download | https://ftp.icgc.cat/descarregues/CatLCNet |
| PREVINCAT Server | https://previncat.ctfc.cat/en/index.html |
| PREVINCAT MDPI Paper | https://www.mdpi.com/2072-4292/12/24/4124 (Note: May require direct browser access) |
| Copernicus EMS Catalonia Fire (2022) | https://data.jrc.ec.europa.eu/dataset/7d5a5041-efac-4762-b9d1-c0b290ab2ce7 |
| WUI Map Catalonia | https://pdxscholar.library.pdx.edu/esm_fac/215/ |
| Fire Sondes Data (Catalonia) | https://zenodo.org/records/6424854 |

#### Python Libraries & Tools

**Core ML Libraries:**
| Library | Purpose | PyPI | GitHub | Documentation |
|---------|---------|------|--------|---------------|
| PyTorch | Deep learning framework | [pypi.org/project/torch](https://pypi.org/project/torch/) | [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) | [pytorch.org/docs](https://pytorch.org/docs/) |
| segmentation_models_pytorch | U-Net segmentation with pretrained encoders | [pypi.org/project/segmentation-models-pytorch](https://pypi.org/project/segmentation-models-pytorch/) | [github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) | [smp.readthedocs.io](https://smp.readthedocs.io/) |
| Weights & Biases (wandb) | Experiment tracking and MLOps | [pypi.org/project/wandb](https://pypi.org/project/wandb/) | [github.com/wandb/wandb](https://github.com/wandb/wandb) | [docs.wandb.ai](https://docs.wandb.ai/) |
| Optuna | Hyperparameter optimization | [pypi.org/project/optuna](https://pypi.org/project/optuna/) | [github.com/optuna/optuna](https://github.com/optuna/optuna) | [optuna.readthedocs.io](https://optuna.readthedocs.io/) |

**Geospatial Libraries:**
| Library | Purpose | PyPI | GitHub | Documentation |
|---------|---------|------|--------|---------------|
| rasterio | Geospatial raster I/O | [pypi.org/project/rasterio](https://pypi.org/project/rasterio/) | [github.com/rasterio/rasterio](https://github.com/rasterio/rasterio) | [rasterio.readthedocs.io](https://rasterio.readthedocs.io/) |
| geopandas | Geospatial vector data operations | [pypi.org/project/geopandas](https://pypi.org/project/geopandas/) | [github.com/geopandas/geopandas](https://github.com/geopandas/geopandas) | [geopandas.org](https://geopandas.org/) |
| sentinelsat | Search and download Sentinel-2 products | [pypi.org/project/sentinelsat](https://pypi.org/project/sentinelsat/) | [github.com/sentinelsat/sentinelsat](https://github.com/sentinelsat/sentinelsat) | [sentinelsat.readthedocs.io](https://sentinelsat.readthedocs.io/) |
| pystac-client | STAC API client for Sentinel-2 data | [pypi.org/project/pystac-client](https://pypi.org/project/pystac-client/) | [github.com/stac-utils/pystac-client](https://github.com/stac-utils/pystac-client) | [pystac-client.readthedocs.io](https://pystac-client.readthedocs.io/) |
| odc-stac | Open Data Cube integration with STAC | [pypi.org/project/odc-stac](https://pypi.org/project/odc-stac/) | [github.com/opendatacube/odc-stac](https://github.com/opendatacube/odc-stac) | [datacube-core.readthedocs.io](https://datacube-core.readthedocs.io/) |
| sentinelhub | High-level API for on-demand processing | [pypi.org/project/sentinelhub](https://pypi.org/project/sentinelhub/) | [github.com/sentinel-hub/sentinelhub-py](https://github.com/sentinel-hub/sentinelhub-py) | [sentinelhub-py.readthedocs.io](https://sentinelhub-py.readthedocs.io/) |
| folium | Interactive maps for Python | [pypi.org/project/folium](https://pypi.org/project/folium/) | [github.com/python-visualization/folium](https://github.com/python-visualization/folium) | [python-visualization.github.io/folium](https://python-visualization.github.io/folium/) |

**Application Libraries:**
| Library | Purpose | PyPI | GitHub | Documentation |
|---------|---------|------|--------|---------------|
| FastAPI | API framework | [pypi.org/project/fastapi](https://pypi.org/project/fastapi/) | [github.com/tiangolo/fastapi](https://github.com/tiangolo/fastapi) | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) |
| Streamlit | Web app framework | [pypi.org/project/streamlit](https://pypi.org/project/streamlit/) | [github.com/streamlit/streamlit](https://github.com/streamlit/streamlit) | [docs.streamlit.io](https://docs.streamlit.io/) |

**Data & Utility Libraries:**
| Library | Purpose | PyPI | GitHub | Documentation |
|---------|---------|------|--------|---------------|
| DVC | Data Version Control | [pypi.org/project/dvc](https://pypi.org/project/dvc/) | [github.com/iterative/dvc](https://github.com/iterative/dvc) | [dvc.org/doc](https://dvc.org/doc/) |
| numpy | Numerical computing | [pypi.org/project/numpy](https://pypi.org/project/numpy/) | [github.com/numpy/numpy](https://github.com/numpy/numpy) | [numpy.org/doc](https://numpy.org/doc/) |
| matplotlib | Plotting and visualization | [pypi.org/project/matplotlib](https://pypi.org/project/matplotlib/) | [github.com/matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) | [matplotlib.org](https://matplotlib.org/) |
| scipy | Scientific computing (for image processing) | [pypi.org/project/scipy](https://pypi.org/project/scipy/) | [github.com/scipy/scipy](https://github.com/scipy/scipy) | [docs.scipy.org](https://docs.scipy.org/) |

### B. Technology Stack Summary

| Layer | Technology | Notes |
|-------|------------|-------|
| **ML Framework** | PyTorch | Deep learning framework |
| **Segmentation** | segmentation_models_pytorch | U-Net with pretrained encoders (ResNet-34 or EfficientNet-B0) |
| **Experiment Tracking** | Weights & Biases (W&B) | Mandatory for training runs and hyperparameter tracking |
| **Data Versioning** | Hybrid: External storage + Git LFS (small files only) | External storage for large datasets, Git LFS for CSV/configs, manual checksums |
| **Hyperparameter Tuning** | Optuna | 20-50 trials recommended |
| **Geospatial Data** | rasterio, geopandas | Raster and vector data processing |
| **Sentinel-2 Access** | sentinelsat, pystac-client, odc-stac | Copernicus Data Space API access |
| **API** | FastAPI | REST API for detection endpoints |
| **Frontend** | Streamlit (recommended) | Simplified UI for capstone |
| **Maps** | Folium | Interactive maps in Streamlit |
| **Database** | SQLite or skip | Optional for simple demo |
| **Deployment** | HuggingFace Spaces (recommended) | Free hosting for ML demos |
| **Model Format** | PyTorch (.pt) | Standard PyTorch checkpoint format |

### C. Glossary

| Term | Definition |
|------|------------|
| **IoU** | Intersection over Union — primary segmentation overlap metric (target: ≥0.70) |
| **Dice Score** | Harmonic mean of precision and recall (target: ≥0.80) |
| **SWIR** | Short-Wave Infrared — spectral bands (B11, B12) critical for fire detection |
| **NBR** | Normalized Burn Ratio — index for burn scar detection: (B8 - B12) / (B8 + B12) |
| **NDVI** | Normalized Difference Vegetation Index — vegetation health: (B8 - B4) / (B8 + B4) |
| **BAI** | Burned Area Index — active fire enhancement index |
| **L2A** | Level-2A — atmospherically corrected Sentinel-2 product (recommended) |
| **GeoJSON** | Standard JSON format for geographic data (polygons, coordinates) |
| **WGS84** | World Geodetic System 1984 — standard web coordinate system (EPSG:4326) |
| **UTM Zone 31N** | Universal Transverse Mercator — projected coordinate system for Catalonia (EPSG:32631) |
| **DVC** | Data Version Control — tool for ML data versioning (alternative to Git LFS) |
| **Git LFS** | Git Large File Storage — only for small files (CSV, configs) in this project |
| **W&B** | Weights & Biases — experiment tracking platform (mandatory for this project) |
| **Optuna** | Hyperparameter optimization framework |
| **U-Net** | Encoder-decoder segmentation architecture with skip connections |
| **ResNet-34** | Recommended encoder for U-Net (21M parameters, balanced performance) |
| **EfficientNet-B0** | Alternative encoder for U-Net (5M parameters, more efficient) |
| **BCE + Dice Loss** | Combined loss function (0.5:0.5 weight) for class imbalance |
| **AdamW** | Optimizer with weight decay (recommended learning rate: 1e-4 to 3e-4) |
| **ReduceLROnPlateau** | Learning rate schedule that reduces LR when validation plateaus |
| **Catalonia Validation Set** | Mandatory validation set created from Generalitat fire perimeters (2023-2024) |
| **HuggingFace Spaces** | Recommended deployment platform (free hosting for ML demos) |

### D. ML Documentation Deliverables

**Required Documentation for Grading:**

1. **Model Card:** Performance metrics, limitations, intended use, training data summary
2. **Training Report:** Hyperparameters, training curves, validation results, best model selection
3. **Evaluation Report:** Test set results, Catalonia validation results, confusion matrices, error analysis, threshold optimization
4. **Data Documentation:** Dataset sources, preprocessing steps, train/val/test splits, data quality findings
5. **Reproducibility Guide:** Environment setup, data access, code execution, how to reproduce results
6. **Baseline Comparison:** Results of threshold-based and Random Forest baselines vs deep learning model

---

*Document Version: 1.0*
*Last Updated: January 2026*
