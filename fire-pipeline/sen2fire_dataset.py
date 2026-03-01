"""
Sen2Fire dataset loader for fine-tuning on Australian wildfire data.

Sen2Fire: 2,466 patches from NSW Australia (2019–2020), binary fire only, no cloud masks.
- Train: scene1 + scene2 (1,458); Val: scene3 (504); Test: scene4 (504)
- Patches are 512×512; we center-crop to 256×256 to match CEMS pipeline.
- Maps 12 S2 bands to 7 (same as CEMS) and optionally adds NDVI (8 channels).
- Cloud filtering via rule-based score (and optional s2cloudless).
"""

from pathlib import Path
from typing import Callable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

from constants import NUM_INPUT_CHANNELS, RED_INDEX_7, NIR_INDEX_7
from cloud_detection import cloud_score_sen2fire_12band, get_cloud_fraction_sen2fire

# Sen2Fire 12-band array: indices for our 7-band subset (B02, B03, B04, B08, B9, B11, B12)
SEN2FIRE_7BAND_INDICES = [1, 2, 3, 7, 8, 10, 11]

# Default reflectance scale for Sen2Fire (DN often 0–10000)
SEN2FIRE_REFLECTANCE_SCALE = 10000.0

# Splits: paper says train = areas 1+2, val = 3, test = 4
SEN2FIRE_SPLIT_SCENES = {
    "train": ["scene1", "scene2"],
    "val": ["scene3"],
    "test": ["scene4"],
}


def _load_sen2fire_npz(
    npz_path: Path,
    *,
    reflectance_scale: float = SEN2FIRE_REFLECTANCE_SCALE,
    include_ndvi: bool = True,
    crop_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one Sen2Fire .npz and return (image_256, mask_256).

    - image: (12, 512, 512) int16 -> (crop_size, crop_size, 7 or 8) float32 [0, 1]
    - mask: (512, 512) uint8 -> (crop_size, crop_size) int64, 0/1
    """
    data = np.load(npz_path)
    image_12 = data["image"]   # (12, 512, 512)
    label = data["label"]     # (512, 512) 0/1

    # Normalize to [0, 1]
    img = image_12.astype(np.float32) / reflectance_scale
    img = np.clip(img, 0.0, 1.0)
    # (12, 512, 512) -> (512, 512, 12)
    img = np.transpose(img, (1, 2, 0))

    # Select 7 bands
    img_7 = img[:, :, SEN2FIRE_7BAND_INDICES]  # (512, 512, 7)

    if include_ndvi:
        red = img_7[:, :, RED_INDEX_7]
        nir = img_7[:, :, NIR_INDEX_7]
        ndvi_raw = (nir - red) / (nir + red + 1e-8)
        ndvi_01 = np.clip((ndvi_raw + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        img_7 = np.concatenate([img_7, ndvi_01[:, :, np.newaxis]], axis=2)  # (512, 512, 8)

    # Center crop to crop_size
    h, w = img_7.shape[:2]
    assert h >= crop_size and w >= crop_size
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    image_crop = img_7[top:top + crop_size, left:left + crop_size, :]
    mask_crop = label[top:top + crop_size, left:left + crop_size].astype(np.int64)

    return image_crop, mask_crop


class Sen2FireDataset(Dataset):
    """
    PyTorch Dataset for Sen2Fire patches.

    Loads .npz files from scene dirs, center-crops to 256×256, maps to 7 or 8 channels,
    and optionally excludes cloudy patches (rule-based score).
    """

    def __init__(
        self,
        root_dir: Path | str,
        split: str = "train",
        *,
        include_ndvi: bool = True,
        crop_size: int = 256,
        max_cloud_score: float | None = 0.5,
        use_s2cloudless: bool = True,
        reflectance_scale: float = SEN2FIRE_REFLECTANCE_SCALE,
        transform: Callable | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Sen2FireDataset")

        self.root_dir = Path(root_dir)
        self.split = split
        self.include_ndvi = include_ndvi
        self.crop_size = crop_size
        self.max_cloud_score = max_cloud_score
        self.use_s2cloudless = use_s2cloudless
        self.reflectance_scale = reflectance_scale
        self.transform = transform

        scene_names = SEN2FIRE_SPLIT_SCENES.get(split, [split])
        self.samples: list[Path] = []
        for scene in scene_names:
            scene_dir = self.root_dir / scene
            if not scene_dir.exists():
                continue
            for npz in sorted(scene_dir.glob("*.npz")):
                if "_image" in npz.name or "_mask" in npz.name:
                    continue
                self.samples.append(npz)

        if max_cloud_score is not None and self.samples:
            self._filter_cloudy()

    def _filter_cloudy(self) -> None:
        """Exclude patches with cloud score above threshold. Uses s2cloudless when available (default), else rule-based."""
        kept: list[Path] = []
        for npz in self.samples:
            data = np.load(npz)
            img_12 = data["image"]  # (12, 512, 512)
            img_hwc = np.transpose(img_12.astype(np.float32) / self.reflectance_scale, (1, 2, 0))
            img_hwc = np.clip(img_hwc, 0.0, 1.0)
            score = None
            if self.use_s2cloudless:
                score = get_cloud_fraction_sen2fire(img_hwc)
            if score is None:
                score = cloud_score_sen2fire_12band(img_hwc)
            if score <= (self.max_cloud_score or 1.0):
                kept.append(npz)
        self.samples = kept

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        npz_path = self.samples[idx]
        image, mask = _load_sen2fire_npz(
            npz_path,
            reflectance_scale=self.reflectance_scale,
            include_ndvi=self.include_ndvi,
            crop_size=self.crop_size,
        )
        if self.transform:
            # Transform may expect (H,W,C) and mask (H,W)
            result = self.transform(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        return image, mask
