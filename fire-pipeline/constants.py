"""
Shared constants and utilities for the fire detection pipeline.

This module provides a single source of truth for:
- Sentinel-2 band configuration
- Class names and labels
- Device selection utilities
- Patch configuration defaults
"""

from typing import Literal, TYPE_CHECKING

# Torch is optional - only needed for device utilities
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


# =============================================================================
# Sentinel-2 Band Configuration
# =============================================================================

# Band indices to select from 12-band Sentinel-2 L2A imagery (0-indexed)
# Maps original bands to our 7-channel format
BAND_INDICES = (1, 2, 3, 7, 8, 10, 11)

# Band names in order of our 7-channel format
BAND_NAMES = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")

# Indices of Red and NIR within the 7-channel array (for NDVI: (NIR - Red) / (NIR + Red))
RED_INDEX_7 = 2   # B04
NIR_INDEX_7 = 3   # B08

# Human-readable band descriptions
BAND_DESCRIPTIONS = (
    "B02 (Blue, 490nm)",
    "B03 (Green, 560nm)",
    "B04 (Red, 665nm)",
    "B08 (NIR, 842nm)",
    "B8A (NIR narrow, 865nm)",
    "B11 (SWIR1, 1610nm)",
    "B12 (SWIR2, 2190nm)",
    "NDVI (vegetation index)",
)

# Number of input channels for the model (7 bands + NDVI when vegetation is enabled)
NUM_INPUT_CHANNELS = 8
NUM_SPECTRAL_BANDS = 7

# Names for all model input channels (7 bands + NDVI)
INPUT_CHANNEL_NAMES = (*BAND_NAMES, "NDVI")


# =============================================================================
# Class Configuration
# =============================================================================

# DEL (Delineation) - Binary fire detection
DEL_CLASS_NAMES = ("background", "fire")
DEL_NUM_CLASSES = 2

# GRA (Grading) - Severity levels
GRA_CLASS_NAMES = ("no_damage", "negligible", "moderate", "high", "destroyed")
GRA_NUM_CLASSES = 5

# Mapping from num_classes to class names
CLASS_NAMES_MAP = {
    2: DEL_CLASS_NAMES,
    5: GRA_CLASS_NAMES,
}


def get_class_names(num_classes: int) -> tuple[str, ...]:
    """Get class names for the given number of classes.

    Args:
        num_classes: Number of classes (2 for DEL, 5 for GRA)

    Returns:
        Tuple of class names

    Raises:
        ValueError: If num_classes is not 2 or 5
    """
    if num_classes not in CLASS_NAMES_MAP:
        raise ValueError(f"num_classes must be 2 or 5, got {num_classes}")
    return CLASS_NAMES_MAP[num_classes]


# =============================================================================
# Patch Configuration Defaults
# =============================================================================

DEFAULT_PATCH_SIZE = 256
DEFAULT_STRIDE_TRAIN = 224  # patch overlap for training (256 - 224 = 32 pixels overlap)
DEFAULT_STRIDE_INFERENCE = 256  # No overlap for inference
DEFAULT_MAX_CLOUD_COVER = 0.5


# =============================================================================
# Device Utilities
# =============================================================================

DeviceType = Literal["auto", "cuda", "mps", "cpu"]


def get_device(device: DeviceType = "auto") -> "torch.device":
    """
    Get the appropriate torch device.

    Args:
        device: Device specification
            - "auto": Automatically select best available (CUDA > MPS > CPU)
            - "cuda": Use CUDA GPU
            - "mps": Use Apple Metal Performance Shaders
            - "cpu": Use CPU

    Returns:
        torch.device instance

    Raises:
        ImportError: If torch is not installed
        RuntimeError: If requested device is not available
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for device utilities. "
            "Install with: uv sync --extra train"
        )

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")
        return torch.device("cuda")
    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this system")
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name(device: "torch.device") -> str:
    """Get a human-readable name for the device.

    Raises:
        ImportError: If torch is not installed
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for device utilities.")

    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(device.index or 0)})"
    elif device.type == "mps":
        return "Apple MPS"
    else:
        return "CPU"


# =============================================================================
# Visualization Colors
# =============================================================================

# Colors for DEL mask visualization (hex colors)
DEL_COLORS = ("#2d5016", "#ff4444")  # Dark green (no fire), Red (fire)
DEL_LABELS = ("No Fire", "Burned")

# Colors for GRA mask visualization
GRA_COLORS = ("#2d5016", "#b4fe8e", "#fed98e", "#fe9929", "#cc4c02")
GRA_LABELS = ("No Damage", "Negligible", "Moderate", "High", "Destroyed")

# Colors for cloud mask visualization
CLOUD_COLORS = ("#67BEE0", "#DCDCDC", "#B4B4B4", "#3C3C3C")
CLOUD_LABELS = ("Clear", "Cloud", "Light Cloud", "Shadow")


def get_mask_colors(
    mask_type: Literal["DEL", "GRA", "CM"],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Get colors and labels for mask visualization.

    Args:
        mask_type: Type of mask ("DEL", "GRA", or "CM")

    Returns:
        Tuple of (colors, labels)
    """
    if mask_type == "DEL":
        return DEL_COLORS, DEL_LABELS
    elif mask_type == "GRA":
        return GRA_COLORS, GRA_LABELS
    elif mask_type == "CM":
        return CLOUD_COLORS, CLOUD_LABELS
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
