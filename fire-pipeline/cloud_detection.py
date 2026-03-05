"""
Cloud detection for patches without a cloud mask (e.g. Sen2Fire).

Provides:
1. Rule-based cloud score from reflectance (no extra deps)
2. Optional s2cloudless-based mask when sentinelhub is available
"""

from pathlib import Path
from typing import Literal

import numpy as np

from constants import RED_INDEX_7, NIR_INDEX_7

# Sen2Fire 12-band order: B1..B12. Indices for simple cloud proxy (Blue, Green; high = cloudy)
SEN2FIRE_BLUE_INDEX = 1   # B02
SEN2FIRE_GREEN_INDEX = 2  # B03
SEN2FIRE_NIR_INDEX = 7    # B08


def cloud_score_rule_based(
    image: np.ndarray,
    *,
    blue_index: int = 0,
    green_index: int = 1,
    nir_index: int = 3,
    use_nir_ratio: bool = True,
) -> float:
    """
    Compute a simple cloud score for a patch (rule-based, no model).

    Clouds are typically bright in blue/green and can have high NIR.
    Score is higher for more "cloud-like" patches.

    Args:
        image: (H, W, C) float in [0, 1] or int (will be normalized)
        blue_index: Channel index for blue band
        green_index: Channel index for green band
        nir_index: Channel index for NIR band
        use_nir_ratio: If True, reduce score when NIR is very high (vegetation)

    Returns:
        Score in [0, 1]; higher = more likely cloudy. Typical threshold 0.2–0.4.
    """
    if image.dtype != np.float32 and image.dtype != np.float64:
        img = image.astype(np.float32) / np.iinfo(image.dtype).max
    else:
        img = np.clip(image, 0.0, 1.0).astype(np.float32)

    h, w, c = img.shape
    if blue_index >= c or green_index >= c or nir_index >= c:
        return 0.0

    blue = img[:, :, blue_index].mean()
    green = img[:, :, green_index].mean()
    nir = img[:, :, nir_index].mean()

    # Bright in visible -> cloudy
    bright = (blue + green) / 2.0
    if use_nir_ratio and nir > 1e-6:
        # Very high NIR often means vegetation; reduce score
        bright = bright / (1.0 + nir * 0.5)
    return float(np.clip(bright, 0.0, 1.0))


def cloud_score_sen2fire_12band(image_12ch: np.ndarray) -> float:
    """
    Cloud score for Sen2Fire 12-channel patches (H, W, 12).

    Uses B02 (blue) and B03 (green); optional NIR to down-weight vegetation.
    """
    if image_12ch.shape[-1] < 8:
        return 0.0
    return cloud_score_rule_based(
        image_12ch,
        blue_index=SEN2FIRE_BLUE_INDEX,
        green_index=SEN2FIRE_GREEN_INDEX,
        nir_index=SEN2FIRE_NIR_INDEX,
        use_nir_ratio=True,
    )


def cloud_score_7ch(image_7ch: np.ndarray) -> float:
    """
    Cloud score for 7-channel CEMS-style patches (B02, B03, B04, B08, B8A, B11, B12).

    Blue=0, Green=1, NIR=3.
    """
    return cloud_score_rule_based(
        image_7ch,
        blue_index=0,
        green_index=1,
        nir_index=3,
        use_nir_ratio=True,
    )


def get_cloud_fraction_s2cloudless(
    image: np.ndarray,
    *,
    band_order: Literal["s2l2a", "s2l1c"] = "s2l2a",
) -> float | None:
    """
    Get cloud fraction using s2cloudless (Sentinel Hub) if available.

    Args:
        image: (H, W, C) float reflectance in [0, 1]. Expects 12 bands in S2 order
            (B01..B12). We build 11 bands for s2cloudless (all_bands=False) by using
            indices 0,1,3,4,7,8,9,10,11 and duplicating 11 (B12) for the missing 13th band.
        band_order: 's2l2a' or 's2l1c'

    Returns:
        Cloud fraction in [0, 1], or None if s2cloudless is not installed or
        band count doesn't match (caller should use rule-based fallback).
    """
    try:
        from s2cloudless import S2PixelCloudDetector
    except ImportError:
        return None

    # s2cloudless all_bands=False expects 11 bands (MODEL_BAND_IDS: indices 0,1,3,4,7,8,9,10,11,12 from 13-band L1C).
    # Sen2Fire has 12 bands (0-11 = B01..B12). We don't have index 12, so we supply 11 bands by
    # using indices [0,1,3,4,7,8,9,10,11] and duplicating 11 (B12) as the 11th channel so the detector gets 11 bands.
    if image.shape[-1] < 12:
        return None
    band_indices_11 = [0, 1, 3, 4, 7, 8, 9, 10, 11, 11]  # 11 channels; last is B12 duplicated
    bands = np.clip(image[:, :, band_indices_11].astype(np.float32), 0, 1)
    if bands.shape[-1] != 11:
        return None
    try:
        detector = S2PixelCloudDetector(
            threshold=0.4, average_over=4, dilation_size=2, all_bands=False
        )
        if bands.ndim == 3:
            cloud_probs = detector.get_cloud_probability_maps(np.expand_dims(bands, 0))
            return float(np.mean(cloud_probs[0]))
    except ValueError:
        # Band count mismatch; fall back to rule-based
        return None
    return None


def get_cloud_fraction_sen2fire(image_12ch: np.ndarray) -> float | None:
    """
    Cloud fraction for Sen2Fire 12-channel patch using s2cloudless when available.

    Args:
        image_12ch: (H, W, 12) float reflectance [0, 1] in S2 band order (B1..B12).

    Returns:
        Cloud fraction in [0, 1] if s2cloudless is available, else None (caller should
        use rule-based cloud_score_sen2fire_12band).
    """
    if image_12ch.shape[-1] < 12:
        return None
    return get_cloud_fraction_s2cloudless(image_12ch, band_order="s2l2a")
