![Cover Image](images/image1.jpg)

# Table of Contents

- [Abstract](#abstract)
- [Devastating Blazes](#devastating-blazes)
  - [Wildfire monitoring](#wildfire-monitoring)
- [Using Satellite Images for Fire Detection](#using-satellite-images-for-fire-detection)
- [Our Approach to Wildfire Detection](#our-approach-to-wildfire-detection)
  - [Hypothesis](#hypothesis)
  - [Framework: Cloud Infrastructure and Libraries](#framework-cloud-infrastructure-and-libraries)
    - [Cloud Infrastructure](#cloud-infrastructure)
    - [Libraries and Frameworks](#libraries-and-frameworks)
  - [Datasets](#datasets)
    - [From Where to Gather Data](#from-where-gather-data)
    - [Dataset Sources and Selection](#dataset-sources-and-selection)
    - [CEMS Dataset](#cems-dataset)
    - [Sen2Fire Dataset](#sen2fire-dataset)
  - [Data Pipeline](#data-pipeline)
- [Model Architectures and Training](#model-architectures-and-training)
  - [CNN from Scratch](#cnn-from-scratch)
  - [YOLO](#yolo)
  - [U-Net from Scratch](#u-net-from-scratch)
    - [Architecture](#architecture)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Hypothesis](#hypothesis-1)
    - [Experiment Configuration](#experiment-configuration)
    - [Results](#results)
    - [Conclusions](#conclusions)
  - [SMP Models with Pretrained Encoders](#smp-models-with-pretrained-encoders)
    - [Overview](#overview)
    - [ResNet50 + U-Net++ Architecture (Primary)](#resnet50--u-net-architecture-primary)
    - [Two-Phase Training Pipeline](#two-phase-training-pipeline)
      - [Phase 1 — Binary Fire Detection](#phase-1--binary-fire-detection)
      - [Phase 2 — Severity Fine-Tuning](#phase-2--severity-fine-tuning)
      - [Compute Resources](#compute-resources)  
  - [Results Comparison](#results-comparison)
    - [Summary Table](#summary-table)
    - [Discussion](#discussion)
- [Resource Consumption](#resource-consumption)
- [Model Selection for Application](#model-selection-for-application)
  - [ResNet50 + U-Net++ (Primary)](#resnet50--u-net-primary)
  - [U-Net from Scratch (Secondary)](#u-net-from-scratch-secondary)
  - [Why the Other Models Were Not Selected](#why-the-other-models-were-not-selected)
- [Application](#application)
  - [Overview](#overview-1)
  - [How It Works](#how-it-works)
  - [Features](#features)
  - [Technical Architecture](#technical-architecture)
  - [Application Screenshots](#application-screenshots)
- [Code](#code)
- [Conclusions](#conclusions-1)
- [Bibliography](#bibliography)
- [Resource Links](#resource-links)
  - [Literature Mentioned](#literature-mentioned)     
# Abstract

This project is based on the motivation to improve detection of wildfires through selected datasets from satellite imagery and leveraging a number of deep learning models to find out the best fit in terms of precision and performance. It makes a recommendation on which deep learning approaches fit better to wildfire detection and we show how those deep learning models work on fire detection through an application that scans satellite images across a selected geographical area.

Wildfires have been a major concern because of the change in their behavior, their speed and their destructive capacity. Wildfires have increased fourfold in the number of disastrous events from 1980 to 2023, imposing enormous economic and human costs, particularly in Mediterranean and Temperate Conifer Forest biomes. On the other hand,
over the last years, a new series of data collected through Copernicus Emergency Management System (CEMS), from Sentinel-2 satellites together with the development of a number of deep learning models on classification and segmentation of images is providing a solid foundation for developing deep learning based architectures and applications that can improve fire detection performance.

Our focus has been on fire detection at the Catalonia geographical area, nevertheless we have relied on datasets from CEMS across the world. Our objective has been to build a deep learning system that detects active wildfires and burned areas from Sentinel-2 satellite imagery, with a focus on the Catalonia region. The system should provide near-real-time detection capabilities, fire area measurement, spread analysis, and automated alerts.

Thanks to labeled datasets, we have taken a selection of them as starting point for the training of our deep learning system. We trained and compared multiple deep learning architectures --- pretrained segmentation models via the SMP framework, a U-Net built from scratch, a lightweight CNN classifier, and YOLOv8-Seg --- on two datasets: the Copernicus Emergency Management Service (CEMS) wildfire dataset covering
275 fires across Europe, and the Sen2Fire dataset covering Australian bushfires. Models receive 8-channel multispectral input (7 Sentinel-2 bands plus NDVI) and output pixel-wise fire maps.

Results show that ResNet50 + U-Net++, trained with a two-phase binary-then-severity strategy on the combined CEMS + Sen2Fire dataset, delivers the best performance (Fire IoU 0.779, Fire Dice 0.846). A lightweight U-Net built entirely from scratch achieves surprisingly strong results (Fire IoU 0.745, \~118K parameters) with no pretrained weights. These models were integrated into a Streamlit application that fetches Sentinel-2 imagery from Microsoft Planetary Computer and produces binary fire maps and 5-level severity maps. Based on these experiments, we foresee these models could be used as a detection tool for authorities or entities that need to track how wildfires impact a specific geography.

# Devastating Blazes

Disastrous blazes increased more than fourfold from 1980 to 2023 according to a paper in the journal Science. (Calum X. Cunningham, 2025). Based on that paper, disastrous wildfires occurred globally but were disproportionately concentrated in the Mediterranean and Temperate Conifer Forest biomes, and in populated regions that experience intense fire. Major economic losses caused by wildfire events increased by 4.4 fold from 1980 to 2023, amounting a 28.3 US\$ billion and 0.03% of global GDP. Of the 200 most damaging events, 43% occurred in the last 10 years.

And the impact of the fires on health should not be overlooked. Based on a paper on how "Wildfire smoke exposure" will increase mortality in USA over next decades (Minghao Qiu, 2025 ), that wildfire smoke would kill an estimated 70,000 Americans each year by 2050.

Disasters were heavily concentrated in the Mediterranean Forest/Woodland/Scrub Biome (Europe, southern South America, western USA, South Africa, and southern Australia) and the Temperate Conifer Forest Biome (mostly western North America), where disasters occurred 12.1 and 4.1 times more than expected based on the areas of those
biomes, respectively.

On the other side, increasing expenditure on fire suppression has not prevented the rising occurrence of wildfire disasters. For example, US federal expenditure on fire suppression increased by 3.6 fold from 1985-2022 Therefore, it's of utmost importance the implementation of strategies that reduce fire risk, NOT ONLY including retrofitting existing structures, using stringent fire-sensitive design and materials
in new builds, establishing defendable space, and removing nearby fuel in the home ignition zone, BUT, as well, with a fast, early identification of fire spots or burned areas.

## Wildfire monitoring

In fact, wildfire season, in many regions, is year-round and continuous fire surveillance becomes critical to minimize potential impact in forests and communities. For a long time, watchtowers setup at hilltops have been the traditional means of wildfire monitoring, with an observation radius of up 20 km under ideal conditions (Rego, 2006). Then, automated image observation devices (e.g. video surveillance systems) mounted on watchtowers have largely replaced human observers, demonstrating superior performance during daylight hours (Günay, 2009). Even though, while automated thermal cameras enable basic 24-hour operation, there are still some limitations such visibility constraints from terrain or weather create permanent blind spots unreachable by static towers (Guangqing Zhai, 2025).

Thanks to the development of time series data, advancements in new sensor systems and image processing technology, and the increased availability of free images, Remote Sensing (RS) based on satellite and radar images has been widely applied in wildfire monitoring, especially suitable for vast remote areas (Guangqing Zhai, 2025). It has been recognized that updated information on landscape fire activity is essential to aid fire management, and that information can only be provided by using satellite Earth Observation approaches (Wooster, 2021) This form of Earth observation is based on detecting the footprint or signature of the electromagnetic radiation emitted as biomass burns. Since the early 1980's active fire (AF) remote sensing conducted using Earth orbiting (LEO) satellites has been deployed in certain regions of the world to map the location and timing of landscape fire occurrence, and from the early 2000's global-scale information updated multiple times per day has been easily available to all. Geostationary (GEO) satellites provide even higher frequency AF information, more than 100 times per day in some cases, and both LEO- and GEO-derived AF products now often include estimates of a fires characteristics, such as its fire radiative power output, in addition to the fires detection (Wooster, 2021). AF data provide  information relevant to fire activity ongoing when the EO data were collected, and this can be delivered with very low latency times.

Satellite EO can be used to probe many fire characteristics, including burned area (Giglio, 2018) and the concentration and composition of smoke plumes . Active fire (AF) remote sensing primarily focuses on identifying the location, timing and radiative strength (Fire Radiative Power; FRP) of fires that are actually consuming vegetation and/or organic soil at the time the observations were made.

Active fire (AF) detection and characterization is based on remote sensing of some of the approximately 20 MJ.kg-1 of energy released when vegetation and organic soil burns (Sullivan, 2008) Of the total energy released, only about 10 -20% is released as electromagnetic radiation. This radiative energy rate is far higher than from the same area of ambient land however, and its spectral distribution follows Planck's Radiation law and its derivative Wien's Displacement (Wooster, 2021). Based on this radiative analysis, there was a work to define Active Fire detection algorithms that could identify pixels that contain active fires. Those algorithms were defined to discriminate pixels with active fires from non-fire pixels, taken into consideration potential confusing effects such as sun glint, non-burning hot areas or regions with high local thermal contrast.

All in all, active fire detection has evolved over time, from reporting the timing and location of actively burning fires to include measures such as fire effective temperature, area and fire radiative power (FRP).

# Using Satellite Images for Fire Detection

From Earth Observation (EO) satellites, we can get near real-time (NRT) data streams that can help on detection and monitoring of active fires. And there are expectations that future satellite missions, including higher spatial resolution Geostationary (GEO) systems and an increase in the number of Active Fire capable Low Earth Orbit (LEO) system, including those of small satellite constellations, will provide more opportunities to improve fire detection.

As of 2018, some authors started to map fires based on data from Sentinel-2 sensor (Fernando Rodriguez-Jimenez, 2023). The Sentinel-2 mission is part of the Copernicus Earth observation program, with twin satellites Sentinel 2A launched in 2015, and Sentinel 2B in 2017. Sentinel-2 carries multispectral imaging instruments on board. The
spatial resolution for the output product chosen is 20 m, due to the spectral bands using the most usual indices for mapping burned areas and their severity. Fires are detected through the application of the Relativized Burn Ratio (RBR) (Sean Parks, 2014). This index is calculated from the Normalized Burn Ratio (NBR) using Band 8A and Band 12 in the pre-fire and post-fire images.

The new generation of geostationary satellites provides observations every 10 to 15 min at an improved spatial resolution (2--3 km) making it possible to detect short-lived fires not detectable by polar-orbiting satellites and to track in detail the evolution of the fire line and fire radiative power (Oliva, 2020). In addition, they have enhanced sensors that provide information on 12--16 spectral bands with improved radiometry of 10--14 bits. They introduce a substantial improvement in spatial, temporal, spectral, and radiometric resolution over their predecessors, which correspondingly relates to an enhanced capability for fire detection.

While an advantage of geostationary sensors is their temporal resolution, the polar-orbiting sensors have finer spatial resolution, which ensures better performance at locating and mapping thermal anomalies. AF detections are produced using different polar-orbiting sensors, such as MODIS , the Visible Infrared Imaging Radiometer Suite (VIIRS), FengYun-3C VIRR, Landsat-8 OLI, TET-1, and Firebird.

Currently, the best compromise between spatial and temporal resolution is provided by the operational active fire product derived from the 375 m VIIRS bands on board the National Polar-Orbiting Partnership (NPP) satellite since 2013 and on board the NOAA-20 since 2017. Its improved resolution, frequent acquisitions, and higher sensitivity to burning pixels allow the direct estimation of the burned area by aggregation of consecutive fire detections and the estimation of fire-driven deforestation .With two satellites in operation, the 375 m AF product time lapse between acquisitions is reduced to a few hours ensuring a higher frequency of observations providing fire behavior to fire managers and atmospheric modelers with higher resolution active fire data.

All in all, satellite earth observations are used more and more to monitor, estimate fire impacts by assessing the Burned Area (BA), their spatial variation as shape, size or orientation. New algorithms have been developed for measuring Burned Area, incorporating machine learning approaches. Let's show an example using Moderate Resolution Imaging Spectroradiometer (MODIS) sensor image (Rui Ba, 2019 ), from the MODIS data of several wildfires in the American states of Nevada, Washington and California in 2016. They utilized a back-propagation neural network (BPNN) to learn the spectral differences of different types from the training sample sets and obtain the output burned area map.

![](images/image2.png)

# Our Approach to Wildfire Detection

### Hypothesis

Our hypothesis is that combining Sentinel-2 multispectral imagery with state-of-the-art deep learning segmentation architectures and high-quality labeled datasets from CEMS can produce a system that:

-   Detects active wildfires and burned areas with Fire IoU \> 0.70.
-   Provides severity grading (5-class) at a mean IoU comparable to
    published remote sensing benchmarks.
-   Generalizes beyond training geography (Mediterranean to Australia
    and back).
-   Runs inference fast enough for near-real-time alerting.

The data selection criteria required: Sentinel-2 imagery (same sensor as the operational system, minimizing domain shift), expert-labeled masks, geographic diversity, rich annotations (binary DEL, severity GRA, cloud masks, land cover), and accessibility through standard APIs. We selected CEMS-Wildfire as the primary training dataset and Sen2Fire as a geographic augmentation source.

The complete pipeline follows nine steps: data ingestion → preprocessing → patch extraction → augmentation → Phase 1 binary training → Phase 2 severity fine-tuning → evaluation → inference → application. Experiment tracking was performed with Weights & Biases (W&B).

Our aim with this project is to find the best fit in terms of datasets and deep learning architectures to the task of classifying and segmenting wildfires. Our expectation is to be closer as we can to a near real time detection of wildfires. Based on the previous research about which data can be used for fire detection, our hypothesis is that we can leverage satellite images from specific satellite sensors that are able to provide enough image resolution.

Imagine you have satellite photos of forests that caught fire. Your goal is to teach a computer to look at a satellite photo and say \"fire happened here, here, and here\" - like coloring in the burned areas. Once trained, our model can look at a satellite image it\'s never seen and draw the fire boundaries automatically.

![](images/image3.png)

### Framework: Cloud Infrastructure and Libraries

#### Cloud Infrastructure

All training runs were executed on **Google Cloud Platform (GCP)**, using instances located in Europe to minimize data transfer latency to Copernicus and Sentinel-2 data sources.

  | Resource | Specification       |                                                                                                                                                    
  |----------|---------------------|
  | vCPUs    | 4                   |                                                                                                                                                    
  | RAM      | 16 GB               |                                                                                                                                                  
  | GPU      | NVIDIA L4 (24 GB VRAM) |
  | Disk     | 500 GB              |  

The NVIDIA L4 GPU provided sufficient VRAM for all training runs, including the memory-intensive ResNet50 + U-Net++ model (\~10--12 GB peak VRAM). Training was managed through shell scripts and Python entry points using **uv** as the package and virtual environment manager.

#### Libraries and Frameworks

| **Category**              | **Library / Tool** |
|--------------------------|--------------------|
| Deep learning            | **PyTorch** — model definition, training loops, loss functions, and inference |
| Segmentation models      | **Segmentation Models PyTorch (SMP)** — pretrained encoder-decoder architectures (U-Net, U-Net++, DeepLabV3+) |
| Data / arrays            | **NumPy** — array manipulation, patch extraction, data augmentation |
| Geospatial I/O           | **rasterio** — reading and writing multi-band GeoTIFF files (Sentinel-2 imagery and masks) |
| Hyperparameter tuning    | **Ray Tune** — distributed hyperparameter search (4 trials, top-3 re-trained) |
| Experiment tracking      | **Weights & Biases (W&B)** — metric logging, run comparison, GPU monitoring |
| Application              | **Streamlit** — interactive web application for inference and visualization |
| Satellite data access    | **planetary-computer + pystac-client** — STAC API queries to Microsoft Planetary Computer for Sentinel-2 L2A imagery |
| Package management       | **uv** — fast dependency resolution and virtual environment management |

### Datasets

#### From Where to Gather Data

We have been looking at different sources of satellite data, and we have selected data from new sensors as Sentinel-2, because of its availability for medium-high spatial resolution of optical satellite data. This helps a more detailed assessment of small fires than other Burned Area products. Sentinel-2 is offered by the Copernicus Program, and offers a time series data of burned areas, with high revisit frequency, with improved spatial and spectral resolution of the MSI optical sensor (Filipponi, 2019).

Sentinel-2 is the most used satellite for mapping the precise perimeter of a fire. They use SWIR (Short-Wave Infrared) Bands. While standard cameras only see smoke, Sentinel-2's Bands 11 and 12 can \"see\" the heat of the fire through the smoke. On a map, active fire fronts appear as glowing orange or red lines. (CEMS, Sentinel-2 Documentation, n.d.).

Then, they can run a burn Scar Analysis by comparing images from before and after a fire, scientists calculate the NBR (Normalized Burn Ratio). This identifies which areas were hit hardest and helps plan for forest restoration.

All in all, Sentinel-2 provides high-resolution images (10 to 20 meters), which is sharp enough to see individual roads, firebreaks, and structures.

The satellite imagery comes from Sentinel-2 L2A, which spans across 12 distinct bands of the light spectrum, with resolutions of 10 to 20 meters. The Sentinel Level-2A data undergoes an atmospheric correction process to adjust for the reflectance values influenced by the moisture in the atmosphere. These images are accessed via the Microsoft Planetary Computer STAC API.

#### Dataset Sources and Selection

The Copernicus Emergency Management Service (CEMS) is a component of the European Union\'s Copernicus Program. It provides rapid geospatial information during emergencies, and damage assessment for events such as floods, wildfires, and earthquakes. In particular, Copernicus Rapid Mapping provides on-demand mapping services in cases of various natural disasters, offering detailed and up-to-date geospatial information that assists in disaster management and risk assessment (CEMS, CEMS Rapid Mapping, n.d.).

The Copernicus Emergency Management Service (CEMS) Wildfire dataset spans from June 2017 to April 2023. The dataset includes Sentinel-2 images related to wildfires, along with their respective severity and delineation masks. Additionally, the dataset is enhanced with cloud and landcover masks, providing more valuable information for future training of a semantic segmentation model. The dataset comprises over 500 high-quality images, suitable for subsequent semantic segmentation model training. The dataset is available on [Huggingface](https://huggingface.co/datasets/links-ads/wildfires-cems).
(Huggingface, n.d.).

  -----------------------------------------------------------------------
  ![](images/image4_resized.png)                                          
  -----------------------------------------------------------------------


#### CEMS Dataset

The **Copernicus Emergency Management Service (CEMS) Wildfire dataset** is the primary training source. It is authoritative, expert-labeled, and covers European fire events on the same Sentinel-2 sensor used for operational inference.

**Source:** Copernicus Emergency Management Service, distributed via HuggingFace (links-ads/wildfires-cems).

**Coverage:**

-   275 fire activations (approximately 560 image tiles).
-   19 European countries, with emphasis on the Mediterranean region.
-   Temporal range: 2017-2023.
-   Temporal split used: 2017-2021 train, 2022 validation, 2023 test (split at activation level to prevent data leakage).

**Spectral content:**

-   12 Sentinel-2 L2A bands at 10-20 m spatial resolution.
-   7 bands selected for training: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B8A (NIR-narrow), B11 (SWIR1), B12 (SWIR2).
-   NDVI computed and appended as 8th channel. This one provides with a vegetation information that it is very useful for our proposal.

**Label types:**

-   **DEL (Delineation):** Binary mask 0 = not burned, 1 = burned. Used for Phase 1 (binary fire detection).

-   **GRA (Grading):** 5-class severity mask 0 = no damage, 1 = negligible, 2 = moderate, 3 = high, 4 = destroyed. Used for Phase 2 (severity assessment). 

-   **CM (Cloud Mask):** 4-class cloud mask generated with CloudSen12 0 = clear, 1 = clouds, 2 = light clouds, 3 = shadow.

-   **LC (Land Cover):** Land cover mask from ESRI and ESA WorldCover sources.

![](images/image6.png)

*Severity colormap visualization*

![](images/image7.png)

*Cloud masks from CloudSen12 model for activation EMSR382*

Each activation provides files following the naming convention:

EMSR{id}\_AOI{n}\_{seq}\_S2L2A.tif ← Sentinel-2 image (multi-band
GeoTIFF)\
EMSR{id}\_AOI{n}\_{seq}\_DEL.tif ← Binary delineation mask\
EMSR{id}\_AOI{n}\_{seq}\_GRA.tif ← Severity grading mask (0--4)\
EMSR{id}\_AOI{n}\_{seq}\_CM.tif ← Cloud mask\
EMSR{id}\_AOI{n}\_{seq}\_LC.tif ← Land cover mask

Labels were produced by human experts at CEMS: analysts compared pre-fire and post-fire Sentinel-2 imagery and manually drew polygons around burned areas, which were then rasterized into DEL and GRA masks. This gives the dataset high annotation quality human-verified ground truth rather than automated predictions.

**Storage:** approximately 37 GB combined (23 GB DEL patches, 14 GB GRA patches after patch extraction).

#### Sen2Fire Dataset

The **Sen2Fire dataset** provides geographic diversity by covering Australian bushfires, complementing the European-focused CEMS data.


**Coverage:**

-   2,466 patches from 4 bushfire scenes, New South Wales (NSW), Australia, 2019--2020 fire season

-   Geographic splits: scene1 + scene2 = train (1,458 patches), scene3 = validation (504 patches), scene4 = test (504 patches)

**Spectral content:**

-   12 Sentinel-2 L2A bands + 1 Sentinel-5P aerosol index (13 channels total)

-   Patch size: 512×512 pixels; center-cropped to 256×256 to match CEMS pipeline

-   Same 7-band subset as CEMS is selected; NDVI computed from Red/NIR; output is 8 channels

**Label types:**

-   Binary fire mask only (0/1), derived from MODIS MOD14A1 V6.1

-   **No severity (GRA) labels**. In this dataset, we only have binary information, fire/no-fire.

**Cloud handling:**

-   No cloud mask included in the dataset

-   Rule-based cloud scoring using B02/B03 brightness and optional B10 (water vapour) heuristics

-   By default, "s2cloudless" model is applied; patches with cloud fraction \> 30% are excluded


**Storage:** approximately 6 GB.

**Why Sen2Fire matters:** The 4 Australian bushfire scenes from the 2019--2020 season represent a different fire regime from European Mediterranean fires. Training on both datasets forces the model to learn spectral fire signatures that generalize across continents, rather than overfitting to European vegetation and fire behavior patterns.

### Data Pipeline

The data pipeline converts raw GeoTIFF (CEMS) and .npz (Sen2Fire) files into normalized 256×256 patch arrays suitable for training. It is implemented primarily in run_pipeline.py and patch_generator.py.

**Band selection and NDVI:**

Seven Sentinel-2 bands are selected that carry the most discriminative information for fire detection:

  | Index | Band         | Wavelength | Role                       |
  |-------|--------------|------------|----------------------------|                       
  | 0     | B02 Blue     | 490 nm     | Water, smoke               |
  | 1     | B03 Green    | 560 nm     | Vegetation                 |                                                                                                                    
  | 2     | B04 Red      | 665 nm     | Chlorophyll, burn scars    |                                                                                                                    
  | 3     | B08 NIR      | 842 nm     | Healthy vegetation         |                                                                                                                    
  | 4     | B8A NIR-narrow | 865 nm   | Vegetation boundary        |                                                                                                                    
  | 5     | B11 SWIR1    | 1610 nm    | Fire and burn detection    |
  | 6     | B12 SWIR2    | 2190 nm    | Fire and burn detection    |                                                                                                                    
  | 7     | NDVI         | computed   | Vegetation index           | 

NDVI (Normalized Difference Vegetation Index) is appended as the 8th channel, computed as:

NDVI = (NIR - Red) / (NIR + Red) where NIR is the reflectance value of the Near Infrared.

High NDVI indicates healthy vegetation; low NDVI indicates burned areas, water, bare soil, and shadow. NDVI was recommended as a key discriminator for separating burn scars from other low-reflectance surfaces. Using NDVI as an explicit channel proved beneficial it helps the model discriminate fire damage from spectrally similar
confounders.

**Sliding window extraction:**

Full Sentinel-2 scenes are approximately 10,980×10,980 pixels far too large for direct GPU processing. The pipeline extracts 256×256 patches with a stride of 128 pixels (50% overlap) during training. This overlap provides a data augmentation effect: each fire area at patch boundaries is seen from multiple contexts by the model.

During inference, a stride of 192 pixels (25% overlap) is used, and Gaussian-weighted blending is applied to avoid block artifacts at patch boundaries (Chevalier method: each patch prediction is weighted by a 2D Gaussian map, higher at center, and overlapping predictions are averaged).

**Cloud filtering:**

Patches with more than 50% cloud coverage (from the CEMS CM mask) are rejected during patch extraction. For Sen2Fire, where no cloud mask exists, a rule-based or s2cloudless score is computed and patches above the threshold are excluded.

**Output format:**

Extracted patches are saved as NumPy binary files:

-   \*\_image.npy - shape (256, 256, 8), float32, normalized to [0, 1]

-   \*\_mask.npy - shape (256, 256), uint8, values 0/1 (DEL) or 0--4 (GRA)

Two output directories are produced:

-   patches/ - DEL binary masks (used in Phase 1)

-   patches_gra/ - GRA severity masks (used in Phase 2)

**Class weights:**

The fire class is severely underrepresented in most patches (most pixels are non-fire). Class weights are computed to compensate for this imbalance and are applied in the loss function.

**Augmentation:**

Augmentation is applied only during training, with geometric transforms applied identically to image and mask:

| Transform                  | Parameters                        |
  |----------------------------|-----------------------------------|                       
  | Random horizontal flip     | p = 0.5                           |                                                                                                                    
  | Random vertical flip       | p = 0.5                           |                                                                                                                    
  | Random 90° rotation        | p = 0.5                           |                                                                                                                    
  | Random brightness/contrast | factor ∈ [0.8, 1.2], p = 0.3     |                                                                                                                     
  | Gaussian noise             | σ ∈ [0, 0.02], p = 0.2           | 

SWIR/NIR bands are excluded from color-space augmentation, as they encode physical reflectance properties that should not be distorted.

**Dataset split strategy:**

Splits are done at the **activation level** (not patch level) to prevent data leakage all patches from one fire event are assigned to the same split.

| Split | CEMS                    | Sen2Fire                        |
  |-------|-------------------------|---------------------------------|     
  | Train | Activations 2017–2021   | scene1 + scene2 (1,458 patches) |
  | Val   | Activations 2022        | scene3 (504 patches)            |                                                                                                                 
  | Test  | Activations 2023        | scene4 (504 patches)            |


# Model Architectures and Training

Four families of models were developed and compared: SMP pretrained encoder-decoder models (primary), U-Net from scratch (secondary), CNN from scratch (secondary), and YOLOv8-Seg (secondary). All models receive 8-channel input (7 Sentinel-2 bands + NDVI) and are trained on combined CEMS + Sen2Fire data.

We selected a primary metric for each model deppending of the task that it will tackle and some secundary metrics as Fire Dice, Fire Recall, Mean IoU, and Detection F1.

The loss function used is a combination of Binary Cross Entropy + Dice loss (total = 0.5 × BCE + 0.5 × Dice).

We have some experiment trackings in Weights & Biases (fire-detection project). A total of 59 training runs (46 finished) were tracked.

### CNN from Scratch

#### Architecture

The CNN from scratch is a **patch-level binary classifier** (not a pixel-wise segmentation model). Given a 256×256×8 patch, it outputs a single probability: "does this patch contain fire?"

Architecture: 1×1 spectral convolution → 3 convolutional blocks → global average pooling → fully connected classifier with dropout. Approximately **97,489 parameters**.

This is a simpler task than pixel-wise segmentation: instead of predicting thousands of labels per patch, the model makes a single binary decision per patch. The patch-level label is derived from the mask (if any pixel in the mask \> 0, the patch label is 1).

#### Hypothesis

A compact CNN classifier from scratch can learn to distinguish fire-containing patches from non-fire patches using 8-channel multispectral input, providing a fast, lightweight baseline for fire detection that does not require pixel-level annotation.

#### Experiment Configuration

| Parameter      | Value                                              |
  |----------------|----------------------------------------------------|
  | Data           | CEMS DEL + Sen2Fire (combined)                     |                                                                                                               
  | Input          | 256×256×8 patch → single binary label              |
  | Architecture   | Custom CNN, ~97K params, no pretrained weights     |                                                                                                               
  | Loss           | Binary Cross Entropy                               |
  | Optimizer      | AdamW                                              |                                                                                                               
  | Max epochs     | 50                                                 |
  | Primary metric | Validation F1                                      |                                                                                                               
  | Tuning         | Ray Tune, 4 trials, top-3 re-trained               |


**Hyperparameter search space:**

| Hyperparameter  | Search Space                        |
  |-----------------|-------------------------------------|
  | `learning_rate` | loguniform(5e-5, 5e-4)              |                                                                                                                             
  | `weight_decay`  | loguniform(1e-6, 1e-3)              |                                                                                                                             
  | `dropout`       | choice([0.1, 0.2, 0.3, 0.4])        | 

```bash
uv run python train.py 
--patches-dir ./patches 
--sen2fire-dir ../data-sen2fire 
--output-dir ./output/scratch 
--tune true 
--tune-target scratch 
--tune-samples 4 
--tune-top-k 3 
--epochs 50 
--wandb 
--project fire-detection 
--results-csv ./training_results.csv 
--device auto
```

#### Results

| Run  | Val F1       | Val Precision | Val Recall | Val AUC | Best Epoch |
  |------|--------------|---------------|------------|---------|------------|
  | top1 | **0.6016**   | 0.3600        | 0.6345     | 0.9446  | 29         |                                                                                                           
  | top2 | 0.5779       | 0.3401        | 0.6430     | 0.9405  | 34         |                                                                                                           
  | top3 | 0.5492       | 0.3922        | 0.5805     | 0.9283  | 34         | 

Best result: **Val F1 0.6016, Val AUC 0.9446**.

Training time: approximately 2,441--2,599 seconds per run (~40 minutes).

#### Conclusions

The CNN classifier achieves moderate patch-level classification performance (F1 0.60). The high AUC (0.94) indicates the model has learned a meaningful ranking of fire probability even if the binary threshold produces imprecise F1. However, the low precision (0.36) indicates many false positives: patches are flagged as fire-containing even when the actual fire pixels occupy only a small fraction of the patch. This is a known limitation of patch-level classification a single fire pixel in a 256×256 patch generates a positive label, forcing the model to respond to very weak signals. For operational use, pixel-wise segmentation is clearly superior. The CNN Scratch is included as a baseline to understand the difficulty of the classification task.


### YOLO

#### Architecture

YOLOv8-Seg is a single-stage instance detection and segmentation model adapted for 8-channel multispectral input. Unlike U-Net (which produces a single merged binary mask), YOLOv8-Seg outputs bounding boxes with confidence scores plus instance-level segmentation masks, treating each connected fire area as a separate instance.

Architecture: CSPDarknet backbone → FPN neck → dual-head (detection + segmentation). Approximately **3.2M parameters** (YOLOv8n-seg).

The first convolutional layer was adapted to accept 8-channel input. Pretrained weights from COCO (ImageNet-derived) initialize the backbone.

#### Hypothesis

A detection-style approach (YOLOv8-Seg) can identify individual fire instances in satellite patches with competitive mAP50-95, potentially offering instance-level fire tracking capabilities not available with pure segmentation approaches.

#### Experiment Configuration

| Parameter    | Value                                                    |
  |--------------|----------------------------------------------------------|
  | Data         | CEMS DEL + Sen2Fire, exported to Ultralytics YOLO format |                                                                                                           
  | Input        | 256×256×8 (8-channel TIFF)                               |                                                                                                           
  | Architecture | YOLOv8n-seg, ~3.2M params                               |                                                                                                            
  | Metric       | mAP50-95                                                 |                                                                                                           
  | Tuning       | Ray Tune, 4 trials, top-3 re-trained                     |                                                                                                           
  | Flags        | `--yolo`'  |  

**Hyperparameter search space:**

| Hyperparameter  | Search Space             |
  |-----------------|--------------------------|
  | `lr0`           | loguniform(5e-4, 1e-2)   |                                                                                                                                        
  | `weight_decay`  | loguniform(1e-4, 1e-2)   |
  | `batch`         | choice([8, 16, 32])       | 

```bash
uv run python train.py
--patches-dir ../patches
--sen2fire-dir ../data-sen2fire
--output-dir ./output/yolo
--tune true
--tune-target yolo
--tune-samples 4
--tune-top-k 3
--epochs 50
--wandb
--project fire-detection
--results-csv ./training_results.csv
--device auto
--yolo-imgsz 256 
--yolo-batch 8
```

#### Results

| Run  | mAP50-95     | F1     | Precision | Recall | Training Time (s) |
  |------|--------------|--------|-----------|--------|-------------------|
  | top1 | **0.4397**   | 0.6098 | 0.7130    | 0.5328 | 3,813             |                                                                                                             
  | top2 | 0.4391       | 0.6079 | 0.7078    | 0.5328 | 4,021             |                                                                                                             
  | top3 | 0.4355       | 0.6021 | 0.7240    | 0.5153 | 4,181             |     

Best result: **mAP50-95 0.4397**.

Training time: the slowest model family approximately 63--70 minutes per run.

#### Conclusions

YOLO achieves the weakest results (mAP50-95 0.44) of all models evaluated. The main reason is a significant **domain gap** between COCO (the source of YOLO\'s pretrained weights) and satellite wildfire imagery. COCO contains everyday RGB photographs of objects at human scale; our data consists of 8-channel multispectral Sentinel-2 images of burned landscapes at 10 m spatial resolution. The backbone never saw anything resembling fire scars, smoke plumes, or multispectral reflectance patterns during pretraining. As a result, the pretrained features transfer poorly and the model must relearn relevant representations with very limited domain-specific data.

Beyond the domain gap, the detection-style task formulation is a poor fit for burned area mapping. Burned areas are large, spatially diffuse regions that span most of a tile --- not discrete objects with clear bounding boxes. YOLO\'s detection head is designed for locating individual objects, so it struggles with fire perimeters that are irregular and occupy the majority of the scene. Pixel-wise segmentation models are fundamentally better suited to this problem.



### U-Net from Scratch

#### Architecture

The U-Net from scratch is a compact segmentation model built with no pretrained weights. It is designed to establish a strong baseline for binary fire segmentation while minimizing parameter count.

**Block:** Two sequential Conv3×3 → BatchNorm → ReLU layers (standard U-Net block).

**Encoder (contracting path):**

-   Block 1: 8 → 16 channels, MaxPool → 128×128

-   Block 2: 16 → 32 channels, MaxPool → 64×64

-   Block 3 (bottleneck): 32 → 64 channels

**Decoder (expanding path):**

-   Stage 1: ConvTranspose2d (64 → 32) → concat skip from Block 2 → Conv block → 128×128×32

-   Stage 2: ConvTranspose2d (32 → 16) → concat skip from Block 1 → Conv block → 256×256×16

**Output head:** 1×1 Conv (16 → 2 classes).

**Total parameters: \~118,000** (approximately 415× fewer than ResNet50 + U-Net++).

Skip connections carry high-resolution spatial features from encoder to decoder, preserving precise burn boundary information that would otherwise be lost during downsampling. The architecture intentionally mirrors the SMP U-Net structure but starts from random weights, making it a controlled comparison for the value of ImageNet pretraining.

![](images/image14.png)

*Generic U-Net architecture for fire detection*

![](images/image15.png)

*U-Net scratch architecture used in experiments*

#### Hyperparameter Tuning

Hyperparameter search was performed using Ray Tune with a three-phase
flow:

1.  **Search phase:** 12 trials with random hyperparameter configurations from the search space. This ones are no logged in W&B to avoid some noise.

2.  **Selection phase:** All trials ranked by the primary metric (Fire IoU); top-3 configurations selected.

3.  **Re-training phase:** Each of the 3 best configurations re-trained in full with W&B and CSV logging.

This approach avoids flooding W&B with noise from exploratory runs and ensures only meaningful results are logged.

**Hyperparameter search space:**

| Hyperparameter   | Search Space                  | Description                            |
  |------------------|-------------------------------|----------------------------------------|
  | `learning_rate`  | loguniform(5e-5, 5e-4)        | Initial learning rate                  |                                                                                         
  | `weight_decay`   | loguniform(1e-6, 1e-3)        | L2 regularization                      |                                                                                         
  | `batch_size`     | choice([8, 16, 32])           | Batch size (affects BatchNorm statistics) | 

**Configuration:** 4 tuning trials (tune-samples), top 3 re-trained (tune-top-k), 50 epochs maximum, early stopping on validation Fire IoU, CEMS + Sen2Fire combined training.

```bash
uv run python train.py
--patches-dir ../patches
--sen2fire-dir ../data-sen2fire
--output-dir ./output/unet_scratch
--tune true
--tune-target unet_scratch
--tune-samples 4
--tune-top-k 3
--epochs 50
--wandb
--project fire-detection
--results-csv ./training_results.csv
--device auto
```

#### Hypothesis

A compact U-Net with no pretrained encoder, tuned by Ray Tune, can achieve competitive binary fire segmentation on combined CEMS + Sen2Fire data. Without pretrained weights, the model must learn all spectral-to-feature mappings from fire data only, but its architectural bias (skip connections, encoder-decoder structure) should still enable good Fire IoU given sufficient training data and appropriate hyperparameters.

#### Experiment Configuration

| Parameter      | Value                                                     |
  |----------------|-----------------------------------------------------------|
  | Data           | CEMS DEL + Sen2Fire (combined)                            |                                                                                                        
  | Input          | 256×256×8 (7 bands + NDVI)                                |
  | Architecture   | Custom U-Net, ~118K params, no pretrained weights         |                                                                                                        
  | Loss           | 0.5 × BCE + 0.5 × Dice                                    |
  | Optimizer      | AdamW                                                     |                                                                                                        
  | Max epochs     | 50 (early stopping)                                       |
  | Primary metric | Fire IoU                                                  |                                                                                                        
  | Tuning         | Ray Tune, 4 trials, top-3 re-trained                      |
  | Mask type      | Binary DEL                                                |    

#### Results

| Run  | Fire IoU     | Fire Dice    | Mean IoU     | Best Epoch | LR       | WD       | Batch |
  |------|--------------|--------------|--------------|------------|----------|----------|-------|
  | top3 | **0.7453**   | **0.8541**   | **0.8527**   | 10         | 1.2e-04  | 2.2e-05  | 32    |                                                                                      
  | top1 | 0.7430       | 0.8526       | 0.8518       | 6          | 1.8e-04  | 9.7e-05  | 32    |                                                                                      
  | top2 | 0.7188       | 0.8364       | 0.8362       | 3          | 2.8e-04  | 3.0e-05  | 32    |  

Best result: **Fire IoU 0.745, Fire Dice 0.854, Mean IoU 0.853**.

#### Conclusions

The hypothesis is confirmed and the results are remarkably strong. A 118K-parameter model with no pretrained weights achieves Fire IoU 0.745 only 0.034 below the 49M-parameter ResNet50 + U-Net++ (0.779). This demonstrates that the U-Net architectural inductive bias (skip connections, encoder-decoder structure) is highly effective for satellite fire segmentation even without transfer learning. The combination of CEMS + Sen2Fire provides enough fire examples for the model to learn robust spectral patterns from scratch. The high Mean IoU (0.853) indicates good performance on the background class as well, confirming the model does not collapse to always predicting non-fire. This model is an excellent choice when model size matters for edge deployment, embedded inference, or resource-constrained environments.


### SMP Models with Pretrained Encoders

#### Overview

The Segmentation Models PyTorch (SMP) framework provides a high-level API for building encoder-decoder segmentation networks with pretrained backbones. It supports 12 architectures (U-Net, U-Net++, DeepLabV3+, and others) and over 800 pretrained encoders from the ImageNet benchmark. Using pretrained encoders provides the model with spectral and spatial features learned from millions of natural images, which transfer effectively to satellite imagery via fine-tuning.

**Architectures tested:**

| Encoder        | Decoder    | Role                                        |
  |----------------|------------|---------------------------------------------|
  | ResNet50       | U-Net++    | Best Fire IoU (primary recommendation)      |                                                                                                         
  | ResNet50       | DeepLabV3+ | Large-fire context via atrous convolutions  |                                                                                                         
  | ResNet34       | U-Net      | Balanced baseline                           |                                                                                                         
  | ResNet18       | U-Net      | Fast, lightweight                           |                                                                                                         
  | EfficientNet-B2| U-Net++    | Best quality in EfficientNet family         |                                                                                                         
  | MobileNetV2    | U-Net      | Edge deployment                             |


All models use 8 input channels (standard ImageNet first conv layer is adapted from 3 to 8 channels) and a two-phase training strategy.

### ResNet50 + U-Net++ Architecture (Primary)

The ResNet50 + U-Net++ combination is the primary recommended model. It combines the powerful feature extraction of a deep residual network with the fine-grained boundary localization of the nested U-Net decoder.

**Encoder --- ResNet50:**

ResNet50 is a 50-layer convolutional network using bottleneck residual blocks. It progressively downsamples the input while extracting increasingly abstract features:

Input 256×256×8\
→ Conv 7×7 (stride 2) → 128×128×64\
→ MaxPool (stride 2) → 64×64×64\
→ Layer 1 (3 blocks) → 64×64×256\
→ Layer 2 (4 blocks) → 32×32×512\
→ Layer 3 (6 blocks) → 16×16×1024\
→ Layer 4 (3 blocks) → 8×8×2048 (bottleneck)

The bottleneck design pushes feature maps up to 2048 channels. This wide representation encodes progressively more abstract combinations of spectral bands. In Sentinel-2 imagery, the distinction between an active fire pixel, a highly reflective urban surface, and sun glint over water is subtle in RGB but separable in the full spectral space. ResNet50\'s deeper representations learn those discriminative patterns more reliably than shallower alternatives.

The first convolutional layer is adapted from 3 input channels (ImageNet standard) to 8 input channels, preserving pretrained weights for the first 3 channels and randomly  initializing the additional 5.

**Decoder --- U-Net++:**

U-Net++ replaces the direct skip connections of standard U-Net with a dense nested structure of intermediate nodes. Each decoder node receives: (1) the upsampled output from below, (2) the encoder feature map at the same resolution, and (3) all sibling outputs at the same decoder depth. This dense connectivity reduces the semantic gap between encoder and decoder features, improving boundary precision particularly important for detecting small fire hotspots and precise burn perimeters.

U-Net++: Nested dense skip connections\
X(i,j) = H(concat(X(i,k) for k\<j, upsample(X(i+1,j-1))))

The decoder adds approximately 24M parameters on top of the 25M encoder, giving a total of approximately **49M parameters** for the full dual-head model.

**Output --- dual head:**

After Phase 2 training, the model has two output heads sharing the encoder and decoder:

Decoder output\
├── Binary head (1×1 conv, 2 classes) → fire / no-fire map (256×256)\
└── Severity head (1×1 conv, 5 classes) → damage level map (256×256)

![](images/image8.png)

*ResNet50 + U-Net++ architecture*

![](images/image9.png)

*ResNet50 + U-Net++ decoder detail*

### Two-Phase Training Pipeline

The training follows a two-phase strategy designed to maximize the use of available labeled data and minimize interference between the binary and severity tasks.

#### Phase 1 - Binary Fire Detection

**Hypothesis:**

Combining CEMS DEL (European fires, expert labeled) with Sen2Fire (Australian fires, binary labels) will produce a more generalizable binary fire detection model than training on either dataset alone. The geographic diversity forces the model to learn spectral fire signatures that generalize across different vegetation types, fire regimes, and atmospheric conditions.

**Experiment Configuration:**

| Parameter      | Value                                                  |
  |----------------|--------------------------------------------------------|
  | Data           | CEMS DEL patches + Sen2Fire (combined with `ConcatDataset`) |                                                                                                      
  | Input          | 256×256×8 (7 bands + NDVI)                             |                                                                                                           
  | Model          | Single-head binary (encoder + decoder + binary head, 2 classes) |                                                                                                  
  | Loss           | 0.5 × BCE + 0.5 × Dice                                 |                                                                                                           
  | Optimizer      | AdamW                                                  |                                                                                                           
  | Learning rate  | 2.5e-5                                                 |
  | Batch size     | 16                                                     |                                                                                                           
  | Max epochs     | 50 (with early stopping)                               |                                                                                                           
  | Primary metric | Validation Fire IoU                                    |
  | Script         | `train_combined_binary.py`                             | 


Command to run the experiments:

```bash
uv run python train_combined_binary.py
--patches-dir ../patches
--sen2fire-dir ../data-sen2fire
--output-dir ./output/v3_combined_binary_resnet50_unetpp
--encoder resnet50
--architecture unetplusplus
--epochs 50
```

**Results:**

| Architecture                | Fire IoU     | Det F1 | Fire Recall |
  |-----------------------------|--------------|--------|-------------|
  | resnet50_unetplusplus       | **0.7791**   | 0.8455 | 0.9286      | 27             |                                                                                                
  | resnet50_deeplabv3plus      | 0.7724       | 0.8587 | 0.9181      | ---            |                                                                                                
  | resnet18_unet               | 0.7654       | 0.8351 | 0.9218      | ---            |                                                                                                
  | resnet34_unet               | 0.7650       | 0.8524 | 0.9286      | 17             |                                                                                                
  | efficientnet-b2_unetplusplus| 0.7612       | 0.8711 | 0.9356      |
  | mobilenet_v2_unet           | 0.7572       | 0.8612 | 0.9141      | ---            |   

Training time for ResNet50 + U-Net++: approximately 30 minutes on a single NVIDIA L4 GPU.

![](images/image11.png)

*W&B results of ResNet50 + U-Net++*





#### Conclusions

The hypothesis is confirmed. ResNet50 + U-Net++ achieves the highest Fire IoU (0.779), placing our results in the upper range of published Sentinel-2 fire segmentation benchmarks (typically 0.65-0.90). The combined CEMS + Sen2Fire training provides geographic diversity that improves generalization. High Fire Recall (0.93) is particularly important operationally missing a fire is worse than a false alarm. The pretrained ResNet50 encoder provides robust feature representations that transfer effectively from ImageNet to 8-channel satellite imagery.

#### Phase 2 - Severity Fine-Tuning

**Hypothesis:**

After Phase 1 has trained a robust binary fire detection model, freezing the encoder and binary head and training only a new severity head on CEMS GRA data will add 5-class severity assessment without degrading binary detection performance. This approach leverages the fact that severity labels only exist in CEMS, while binary labels exist in both CEMS and Sen2Fire.

**Experiment Configuration:**

| Parameter     | Value                                                          |
  |---------------|----------------------------------------------------------------|
  | Data          | CEMS GRA patches only (5 severity classes)                     |                                                                                                    
  | Input         | 256×256×8 (same as Phase 1)                                    |
  | Model         | `FireDualHeadModel`: encoder + decoder + binary head (frozen) + severity head (new, trained) |                                                                      
  | Loss          | CrossEntropy (weighted for class imbalance) on severity head   |
  | Learning rate | 5e-4                                                           |                                                                                                    
  | Batch size    | 16                                                             |                                                                                                    
  | Max epochs    | 30                                                             |                                                                                                    
  | Frozen        | Encoder, decoder, binary head                                  |                                                                                                    
  | Trained       | Severity head only (1×1 Conv2d, 5 classes)                     |                                                                                                    
  | Script        | `train_severity_finetune.py`                                   |   

The command to execute the experiment is:

```bash
uv run python train_severity_finetune.py
--checkpoint ./output/v3_combined_binary_resnet50_unetpp/checkpoints/best_model.pt
--patches-dir ../patches_gra
--output-dir ./output/v3_finetune_severity_resnet50_unetpp
--epochs 30
```

**Results:**

  | Architecture                | Mean IoU     | Fire IoU (severity) |
  |-----------------------------|--------------|---------------------|
  | resnet50_unetplusplus       | **0.3444**   | **0.4069**          | 15             |                                                                                                 
  | efficientnet-b2_unetplusplus| 0.3388       | 0.3863              | ---            |                                                                                                 
  | resnet34_unet               | 0.3352       | 0.3715              | ---            |                                                                                                 
  | resnet50_deeplabv3plus      | 0.3329       | 0.3092              | ---            |                                                                                                 
  | mobilenet_v2_unet           | 0.3317       | 0.3638              |
  | resnet18_unet               | 0.3240       | 0.3641              | ---            |  

Mean IoU of 0.34 and fire IoU of 0.41 are in line with expectations for 5-class severity grading. Severity mapping is harder than binary detection because:                                                                                                                                                                              
  1. Classes are fine-grained (negligible vs moderate vs high damage)                                                                                                                 
 2. The "no damage" class dominates (class imbalance)                                                                                                                                
  3. Expert annotations can disagree on boundaries  

Training time: approximately 10--20 minutes per model.

![](images/image13.png)

*ResNet34 U-Net++ vs U-Net baseline comparison*

#### Conclusions

The two-phase approach works as designed. Binary fire detection performance is preserved from Phase 1 the frozen encoder and binary head continue to produce accurate fire maps. The severity head learns a meaningful 5-class severity map with Mean IoU 0.34 and Fire IoU (severity) 0.41. These values are in line with expectations for 5-class remote sensing segmentation: the \"no damage\" class dominates the scene, class boundaries are fine-grained, and expert annotations can disagree at severity boundaries. The dual-head design produces a single model that outputs both binary fire maps and severity maps in one forward pass, enabling efficient inference in the application.

A notable finding from W&B GPU monitoring: the DeepLabV3+ severity fine-tuning run showed GPU utilization as low as 300--500 MHz SM clock (compared to 1500-2000 MHz for U-Net variants), indicating a data pipeline bottleneck rather than a compute bottleneck. This explains its comparatively lower severity performance.

#### Compute Resources

| Phase     | Model            | GPU Memory   | Training Time        |
  |-----------|------------------|--------------|----------------------|
  | Phase 1   | ResNet34 + U-Net | ~6–8 GB      | ~15 min (17 epochs)  | MB           |                                                                                              
  | Phase 1   | ResNet50 + U-Net++| ~10–12 GB   | ~30 min (27 epochs)  | MB          |                                                                                              
  | Phase 2   | Either           | ~6–12 GB     | ~10–20 min           | ---             |                                                                                              
  | Inference | ResNet34 + U-Net | ~2 GB        | ~10–20 ms/patch      | ---             |                                                                                              
  | Inference | ResNet50 + U-Net++| ~4 GB       | ~20–40 ms/patch      | ---             | 

**Compute infrastructure:** Google Cloud Platform, instance g2-standard-4 (4 vCPUs, 16 GB RAM), NVIDIA L4 GPU, 500 GB disk, Debian 11.

### Results

#### Summary Tables

We then compare the results of our models with those of similar projects cited in the literature. This gives us a clearer picture of how the results obtained perform. We break down the results by task performed

Binary Classification:
| Model                | Val F1    |
  |----------------------|-----------|
  | CNN from Scratch     | 0.60      |                                                                                                                                                  
  | Literature baseline  | 0.97 [(1)](#literature-mentioned)  | 
  

Object Detection:

| Model               | mAP50-95  |
  |---------------------|-----------|
  | YOLO v8             | 0.44      |                                                                                                                                                   
  | Literature baseline | 0.83 [(2)](#literature-mentioned)  | 

  Binary Segmentation:

  | Model                      | Fire IoU     | Fire Dice | Fire Recall |                                                                                                               
  |----------------------------|--------------|-----------|-------------|                                                                                                               
  | ResNet50 + U-Net++         | **0.7791**   | 0.8455    | 0.9286      |                                                                                                               
  | ResNet50 + DeepLabV3+      | 0.7724       | 0.8587    | 0.9181      |                                                                                                               
  | ResNet18 + U-Net           | 0.7654       | 0.8351    | 0.9218      |                                                                                                               
  | ResNet34 + U-Net           | 0.7650       | 0.8524    | 0.9286      |                                                                                                               
  | EfficientNet-B2 + U-Net++  | 0.7612       | 0.8711    | 0.9356      |                                                                                                               
  | MobileNetV2 + U-Net        | 0.7572       | 0.8612    | 0.9141      |                                                                                                               
  | U-Net from Scratch         | 0.7453       | 0.8541    | 0.8782      |                                                                                                               
  | Literature baseline        |      0.87[(3)](#literature-mentioned)        |     --      |      --       | 

  Severity Segmentation:
  | Model                      | Fire IoU (severity) | Fire Recall | Mean IoU     |                                                                                                     
  |----------------------------|---------------------|-------------|--------------|                                                                                                     
  | ResNet50 + U-Net++         | **0.4069**          | 0.6246      | **0.3444**   |                                                                                                     
  | EfficientNet-B2 + U-Net++  | 0.3863              | 0.5965      | 0.3388       |                                                                                                     
  | ResNet34 + U-Net           | 0.3715              | 0.5655      | 0.3352       |                                                                                                     
  | ResNet50 + DeepLabV3+      | 0.3092              | 0.5124      | 0.3329       |                                                                                                     
  | MobileNetV2 + U-Net        | 0.3638              | 0.5579      | 0.3317       |                                                                                                     
  | ResNet18 + U-Net           | 0.3641              | 0.5753      | 0.3240       |                                                                                                     
  | Literature baseline        |                     |             | 0.75 [(4)](#literature-mentioned)     | 




### Discussion

The gap between the best pretrained model (ResNet50 U-Net++, Fire IoU 0.779) and the best scratch model (U-Net Scratch, Fire IoU 0.745) is only 0.034, despite a 415× difference in parameters. This highlights the strength of the U-Net inductive bias for satellite fire segmentation. Pretrained encoders add a small but consistent performance boost and better generalization, justifying their use in the final application.

Pixel-wise segmentation clearly outperforms detection (YOLO mAP50-95 0.44) for burned area mapping bounding boxes cannot capture the irregular, diffuse geometry of fire perimeters. Severity segmentation (Mean IoU 0.34) is harder than binary detection by design: five classes with ambiguous boundaries and a dominant \"no damage\" class. The CNN patch classifier\'s low precision (0.36) reflects the limitations of patch-level labels, where a single fire pixel labels an entire 256×256 patch as positive.

## Resource Consumption

![](images/image18.png)

*W&B GPU system metrics dashboard*

GPU utilization during training:

-   U-Net scratch: most consistent high utilization (50-75%).

-   ResNet50 + U-Net++: sustained 60-75% GPU utilization.

-   YOLO: 60--100W power draw with high variability.

-   DeepLabV3+ (severity): GPU SM clock dropped to 300--500 MHz due to data pipeline bottleneck.


# Model Selection for Application

Two models were selected for integration into the fire detection application:

### ResNet50 + U-Net++ (Primary)

**Fire IoU 0.779** - best-performing model overall. Selected when Fire IoU is the priority.

-   Dual-head output: binary fire map + 5-level severity map in a single forward pass.

-   \~49M parameters, 188 MB checkpoint, \~20-40 ms/patch inference on GPU.

-   Requires \~4 GB GPU memory for inference.

### U-Net from Scratch (Secondary)

**Fire IoU 0.745** strong results despite being trained entirely from scratch with no pretrained weights. Selected as a lightweight alternative when GPU memory or model size is constrained.

-   \~118K parameters 415× smaller than ResNet50 U-Net++ .
-   Binary fire detection only (no severity output).
-   Fast inference, very small checkpoint footprint.

Despite operating without any transfer learning, the U-Net from scratch comes remarkably close to the best pretrained model (only 0.034 Fire IoU below ResNet50 U-Net++). This demonstrates that the U-Net encoder-decoder architecture with skip connections is highly effective for satellite fire segmentation even when learned entirely from the task data.

### Why the Other Models Were Not Selected

**CNN Scratch** performs patch-level classification, not pixel-wise segmentation. It cannot produce per-pixel fire masks required for area estimation, perimeter calculation, or map overlays. Its F1 of 0.60 with precision 0.36 also reflects operational unreliability.

**YOLO** was not selected because: 
  - (1) mAP50-95 0.44 is the lowest result of all models; 
  - (2) the detection-style output (bounding boxes) does not support precise fire perimeter calculations; 
  - (3) YOLO\'s pretrained COCO weights provide poor initialization for 8-channel satellite fire data; 
  - (4) training was the most computationally expensive (70+ minutes per run); 
  - (5) pixel-wise segmentation is fundamentally better suited for burned area mapping.

# Application

### Overview

The fire detection application is a web-based interactive tool built with **Streamlit** and **PyTorch**. It allows users to select a geographic region of interest, fetch Sentinel-2 imagery from Microsoft Planetary Computer (via the STAC API), run fire detection inference using the trained models, and visualize binary fire maps and 5-level severity maps as toggleable layers over the satellite imagery.

The application was tested on the Paüls wildfire (southern Catalonia, July 2025), which burned approximately 3,200 ha. The app correctly distinguished pre-fire imagery (June 2025, no fire detected) from post-fire imagery (July 8+, burn scars detected with plausible severity gradients). Eighteen thousand people were confined in that area during the event.

### How It Works

1.  **Region selection:** User draws on the interactive map, selects a preset location (Catalonia, California, Portugal, Greece, Australia), or enters coordinates.

2.  **Image retrieval:** The app queries the Microsoft Planetary Computer STAC API for Sentinel-2 L2A imagery within the specified date range and cloud-cover filter.

3.  **Preprocessing:** The 7 selected bands are extracted, NDVI is computed, and the full scene is tiled into 256×256 patches.

4.  **Inference:** The selected model (ResNet50 U-Net++ or ResNet34 U-Net) runs forward passes on all patches. Predictions are Gaussian-blended and stitched back to the scene extent.

5.  **Visualization:** The user can toggle binary fire map and severity map layers over the original RGB satellite imagery using linked zoom/pan.

### Features

-   **Interactive region selection** --- Draw on map, use preset locations, or enter coordinates

-   **Sentinel-2 imagery** --- Fetch satellite data from Planetary Computer with date range and cloud-cover filters

-   **Real-time fire detection** --- U-Net inference for burn scars with binary fire map and 5-level severity map

-   **Synced image viewer** --- Original vs fire overlay with linked zoom/pan

-   **Analysis history** --- Filter by fire/date, view past runs, load parameters to re-run

-   **Statistics dashboard** --- Total analyses, detection rate, recent fires, data cleanup

-   **Multi-model support** --- Choose among trained models from a dropdown

-   **Quantitative metrics** --- Total affected area, fire percentage, perimeter, centroid, bounding box

![](images/image19.png)

*App architecture schema --- Streamlit + PyTorch + Planetary Computer*

## Technical Architecture

User (browser)\
│\
▼\
Streamlit frontend (app.py)\
│\
├── STAC API query → Microsoft Planetary Computer\
│ └── Sentinel-2 L2A imagery\
│\
├── Preprocessing\
│ └── Band selection + NDVI + patch extraction\
│\
├── Inference (inference.py)\
│ └── FireDualHeadModel (PyTorch)\
│ ├── Binary segmentation map\
│ └── Severity segmentation map\
│\
├── Gaussian blending → full-scene maps\
│\
├── Visualization → Leaflet map overlay\
│\
└── Storage (storage.py)\
└── History: binary + severity + metadata (.npz)

### Application Screenshots

  ----------------------------------------------------------------------------------------------------------------------------------
  ![](images/image20.png)
  ------------------------------------------------------------------------ ---------------------------------------------------------

  ----------------------------------------------------------------------------------------------------------------------------------

![](images/image22.png)

![](images/image23.png)

*Paüls wildfire detection --- Sentinel-2 imagery with fire overlay*

![](images/image24.png)

*App metadata panel --- cloud cover, resolution and ground coverage (no fires detected)*

# Code
Our repository is divided into several branches, each of which handles different tasks.
**feature-v3 branch**: app, dual-head and SMP with pre-trained encoders
**vegetation branch**: CNN built from scratch, YOLO model, U-Net built from scratch and their training pipeline (hyperparameter tuning)
**montse branch**: various tools for joint training and tracking


# Conclusions

This project demonstrates that deep learning on multispectral satellite imagery can achieve strong performance for wildfire burned area detection and severity assessment. The key findings are:

**1. Pretrained encoder-decoder models outperform all alternatives.**
ResNet50 + U-Net++ achieves Fire IoU 0.779 (binary) and Mean IoU 0.344 (severity), placing our results in the upper range of published Sentinel-2 wildfire segmentation benchmarks (0.65-0.90). The two-phase training strategy binary detection on CEMS + Sen2Fire, followed by severity fine-tuning on CEMS GRA only is effective: Phase 1 learns
broad geographic fire detection, and Phase 2 adds severity assessment without degrading binary performance.

**2. A compact U-Net from scratch is surprisingly competitive.** With only \~118K parameters and no pretrained weights, the U-Net scratch model achieves Fire IoU 0.745. The gap to ResNet50 + U-Net++ (0.779) is modest despite a 415× parameter difference. This demonstrates that the U-Net architectural bias is a strong prior for satellite fire
segmentation. For resource-constrained scenarios, the scratch model is a practical alternative.

**Three key lessons learned:**

**Lesson 1: Domain gap matters model architecture and pretraining must match the task.** YOLO\'s pretrained COCO weights represent a significant domain gap when applied to 8-channel multispectral satellite fire imagery. Natural image features learned on RGB COCO do not transfer well to the spectral-spatial patterns of satellite fire signatures. SMP models, also pretrained on ImageNet RGB, transfer better because their decoder architecture is fundamentally designed for pixel-wise segmentation a better match to burned area mapping than YOLO\'s detection-style formulation. When adapting pretrained models to new domains, both the initialization and the output head design must be considered.

**Lesson 2: Metric selection is critical for imbalanced datasets.** Fire pixels constitute a small fraction of most patches. Models that optimize for pixel accuracy can achieve \>85% accuracy by always predicting non-fire. Fire IoU, by explicitly measuring the overlap between predicted and true fire regions, directly penalizes both false alarms and missed fires. We selected Fire IoU as the primary metric for model selection and learning rate scheduling. This choice proved decisive: the DeepLabV3+ severity run that optimized only global loss (without monitoring Fire IoU) effectively learned to predict mostly non-fire and appeared to perform acceptably by overall accuracy while achieving near-zero Fire IoU. Fire Recall was also tracked as a secondary metric in fire detection, missing a fire is operationally worse than a false alarm.

**Lesson 3: Dataset geographic diversity is key.** Combining CEMS (European Mediterranean fires) with Sen2Fire (Australian bushfires) consistently improved generalization compared to training on CEMS alone. The geographic diversity forced the model to learn spectral fire signatures that generalize across different vegetation types, atmospheric conditions, and fire regimes. This finding supports the strategy of using multiple geographically diverse datasets even when they differ in label richness (Sen2Fire has binary labels only; CEMS has full severity grading).

**Limitations and future work:**

Severity segmentation metrics (Mean IoU 0.34) are moderate, not state-of-the-art. Potential improvements include: more GRA-annotated training data, loss weighting for rare severity classes (high and destroyed damage are underrepresented), and post-processing with conditional random fields (CRF) for boundary refinement. The CEMS
dataset is Mediterranean-focused; performance on other fire-prone regions should be validated before operational deployment. The current application is designed for post-fire burned area mapping; extending to active fire detection during an event would require higher temporal resolution data and different training labels.

Future directions include multi-temporal analysis for fire spread detection, automated alerting when fires are detected above a severity threshold, and extension to additional geographic regions using the existing pipeline with minimal modification.


# Bibliography

Anh Tran 1, M. T. (2025). Land8Fire: A Complete Study on Wildfire
Segmentation. *Remote Sensing 17(16)*.

Calum X. Cunningham1\*, J. T. (2025). Climate-linked escalation of
societally disastrous 1 wildfires. *Science Vol 390, Issue 6768*, 53-58.

CEMS. (n.d.). *CEMS Rapid Mapping*. Retrieved from
emergency.copernicus.eu

CEMS. (n.d.). *Sentinel-2 Documentation*. Retrieved from
sentinels.copernicus.eu

Chevalier, G. (n.d.). *Smoothly Blend image patches*. Retrieved from
https://github.com/Vooban/Smoothly-Blend-Image-Patches

Fernando Rodriguez-Jimenez, H. L.-A. (2023). PLS-PM analysis of forest
fires using remote sensing tools. The case of Xur´es in the
Transboundary Biosphere Reserve. *Ecological informatics Volume 75*.

Filipponi, F. (2019). Exploitation of Sentinel-2 Time Series to Map
Burned Areas at the National Level: A Case Study on the 2017 Italy
Wildfires. *Remote Sensing*.

Ghandorh, H. e. (2023). Uni-temporal Sentinel-2 imagery for wildfire
detection using deep learning semantic segmentation models. *Geomatics,
Natural Hazards and Risk, Taylor & Francis.*

Giglio, L. B. (2018). The Collection 6 MODIS burned area mapping
algorithm and product. *Remote Sensing of Environment 217*, 72-85.

Guangqing Zhai, L. D. (2025). From spark to suppression: An overview of
wildfire monitoring, progression. *International Journal of Applied
Earth Observation and Geoinformation, vol 140*.

Günay, O. T. (2009). Video based wildfire detection at. *Fire Saf. J.,
44 (6)*, 860-868.

He, K. Z. (2016). Deep Residual Learning for Image Recognition. *CVPR
2015*, 771-774.

Huggingface. (n.d.). *Huggingface dataset*. Retrieved from
huggingface.co/datasets/links-ads/wildfires-cems

Lisa Knopp \*, M. ,. (2020). A Deep Learning Approach for Burned Area
Segmentation with Sentinel-2 Data. *Remote sensing 12(15)*, 2422.

Merlo, M. (n.d.). *CEMS Wildfire dataset*. Retrieved from
github.com/MatteoM95/CEMS-Wildfire-Dataset

Minghao Qiu, J. L.-N. (2025 ). Wildfire smoke exposure and mortality
burden in the USA under climate change. *Nature*, Number 647,
pages 935--943.

Oliva, E. C. (2020). Satellite Remote Sensing Contributions to Wildland
Fire Science and management. *Fire Science Management* , 81-96.

Rego, F. C. (2006). Modelling the effects of distance on the probability
of fire. *Int. J. Wildland Fire 15 (2)*, 197-202.

Rui Ba, W. S. (2019 ). Integration of Multiple Spectral Indices and a
Neural Network for Burned Area Mapping Based on MODIS Data. *Remote
Sensing* .

Sean Parks, G. D. (2014). A New Metric for Quantifying Burn Severity:
The Relativized Burn Ratio. *Remote sensing vol 6*, 1827-1844.

Sullivan, P. C. (2008). *Grassfires: Fuel, Weather and Fire Behaviour.*
CSIRO Publishing.

Tang Sui, Q. H. (n.d.). BiAU-Net: Wildfire burnt area mapping using
bi-temporal Sentinel-2 imagery and U-Net with attention mechanism.
*International Journal of Applied Earth Observation and Geoinformation*.

Wooster, M. J. (2021). Satellite Remote Sensing of Active 1 Fires:
History and Current Status, Applications and Future Requirements.
*Remote Sens. Environ. Vol 267*.

Zhou, Z. S. (2018). UNet++: A Nested U-Net Architecture for Medical
Image Segmentation. . *DLMIA @ MICCAI 2018. Springer LNCS 11045,*, 3-11.

# Resource links



 | Resource                          | URL                                                                                                                  |
  |-----------------------------------|----------------------------------------------------------------------------------------------------------------------|
  | CEMS-Wildfire Dataset (links-ads) | https://huggingface.co/datasets/links-ads/wildfires-cems                                                             |                          
  | CEMS-Wildfire Dataset (9334hq)    | https://huggingface.co/datasets/9334hq/wildfires-cems                                                               |                           
  | CEMS-Wildfire GitHub              | https://github.com/MatteoM95/CEMS-Wildfire-Dataset                                                                   |                          
  | CEMS-HLS (Alternative)            | https://huggingface.co/datasets/morenoj11/CEMS-HLS                                                                   |                          
  | EO4WildFires Dataset              | https://huggingface.co/datasets/AUA-Informatics-Lab/eo4wildfires                                                     |                          
  | Sen2Fire Dataset                  | https://github.com/Orion-AI-Lab/Sen2Fire (check arXiv paper)                                                         |                          
  | Sen2Fire Paper                    | https://arxiv.org/abs/2403.17884                                                                                      |                         
  | Land8Fire Dataset                 | https://www.mdpi.com/2072-4292/17/16/2776                                                                            |                          
  | TS-SatFire Dataset                | https://arxiv.org/abs/2412.11555                                                                                      |                         
  | S2-WCD Dataset                    | https://ieee-dataport.org/documents/sentinel-2-wildfire-change-detection-s2-wcd                                      |                          
  | FireSR Dataset                    | https://zenodo.org/records/13384289                                                                                   |                         
  | FLOGA Dataset                     | https://arxiv.org/abs/2311.03339                                                                                      |                         
  | Active Fire (Landsat-8)           | https://arxiv.org/abs/2101.03409                                                                                      |                         
  | Copernicus Data Space Browser     | https://browser.dataspace.copernicus.eu/                                                                             |
  | Copernicus Data Space API         | https://documentation.dataspace.copernicus.eu/                                                                       |                          
  | Sentinel Hub API                  | https://docs.sentinel-hub.com/                                                                                        |
  | Sentinel Hub Python               | https://sentinelhub-py.readthedocs.io/                                                                               |                          
  | sentinelsat Python Library        | https://sentinelsat.readthedocs.io/                                                                                   |
  | pystac-client                     | https://pystac-client.readthedocs.io/                                                                                |                          
  | NASA FIRMS                        | https://firms.modaps.eosdis.nasa.gov/                                                                                |                          
  | NASA FIRMS API                    | https://firms.modaps.eosdis.nasa.gov/api/                                                                            |                          
  | EFFIS Portal                      | https://effis.jrc.ec.europa.eu/                                                                                       |                         
  | EFFIS Active Fires                | https://effis.jrc.ec.europa.eu/apps/effis.current-situation/active-fires                                             |
  | Eye on the Fire API               | https://eyeonthefire.com/data-sources                                                                                 |                         
  | Ambee Fire API                    | https://www.getambee.com/api/fire                                                                                     |
  | Generalitat Fire Data             | https://agricultura.gencat.cat/ca/ambits/medi-natural/incendis-forestals/                                            |                          
  | Generalitat Fire Perimeters (CSV) | https://analisi.transparenciacatalunya.cat/api/views/bks7-dkfd/rows.csv?accessType=DOWNLOAD                         |                           
  | Generalitat Civil Protection Risk Map | https://datos.gob.es/en/catalogo/a09002970-mapa-de-proteccion-civil-de-cataluna-riesgo-de-incendios-forestales   |                          
  | ICGC CatLC Dataset                | https://www.icgc.cat/en/Geoinformation-and-Maps/Maps/Dataset-Land-cover-map-CatLC                                   |                           
  | ICGC CatLC FTP Download           | https://ftp.icgc.cat/descarregues/CatLCNet                                                                           |                          
  | PREVINCAT Server                  | https://previncat.ctfc.cat/en/index.html                                                                             |                          
  | PREVINCAT MDPI Paper              | https://www.mdpi.com/2072-4292/12/24/4124 (Note: May require direct browser access)                                  |                          
  | Copernicus EMS Catalonia Fire (2022) | https://data.jrc.ec.europa.eu/dataset/7d5a5041-efac-4762-b9d1-c0b290ab2ce7                                       |                           
  | WUI Map Catalonia                 | https://pdxscholar.library.pdx.edu/esm_fac/215/                                                                     |                           
  | Fire Sondes Data (Catalonia)      | https://zenodo.org/records/6424854                                                                                   | 

### Literature Mentioned
(1) Karaş, İ. R., Sevinç, H. K., Ibrahim, H., Mohamed, M., Şahin, S., and Oumate, M. O.: Deep Learning-Driven Wildfire Detection in Türkiye Using Sentinel-2 Imagery and Convolutional Neural Networks, Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLVIII-4/W18-2025, 179–185, https://doi.org/10.5194/isprs-archives-XLVIII-4-W18-2025-179-2026, 2026.

(2) Akhyar, A., Lee, J., Shahar, N., Saputro, A. H., & Zulkifley, M. A. (2025). Automated forest fire detection in ecological monitoring using enhanced deep learning networks. Scientific Reports, 16(1). https://doi.org/10.1038/s41598-025-31707-6

(3) Zheng, Y., Guo, P., Tian, X., & Ye, Y. (2025). A forest fire identification and monitoring model based on improved YOLOv8. Scientific Reports, 15(1), 37018. https://doi.org/10.1038/s41598-025-17893-3

(4) Sykas, D., Zografakis, D., & Demestichas, K. (2024). Deep learning approaches for wildfire severity prediction: A comparative study of image segmentation networks and visual transformers on the EO4WildFires dataset. Fire, 7(11), 374. https://doi.org/10.3390/fire7110374

*Project repository: <https://github.com/essenciary/aidlfire>*
