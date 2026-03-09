"""
PyTorch Dataset for CEMS Wildfire Patches

Loads pre-generated patches for training fire detection/severity models.
Supports data augmentation via Albumentations.
"""

import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # Fallback for when torch isn't installed

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


# =============================================================================
# Augmentation Pipelines
# =============================================================================

def get_training_augmentation(
    brightness_limit: float = 0.1,
    contrast_limit: float = 0.1,
    noise_var_limit: tuple[float, float] = (0.0, 0.01),
    p_geometric: float = 0.5,
    p_spectral: float = 0.3,
) -> "A.Compose":
    """
    Standard training augmentation pipeline based on wildfire_detection_spec.md.

    Geometric augmentations:
    - Random 90° rotations (0°, 90°, 180°, 270°)
    - Horizontal and vertical flips

    Spectral augmentations:
    - Brightness adjustment (±10%)
    - Contrast adjustment (±10%)
    - Gaussian noise (σ = 0.01)

    Args:
        brightness_limit: Max brightness change (default ±10%)
        contrast_limit: Max contrast change (default ±10%)
        noise_var_limit: Gaussian noise variance range
        p_geometric: Probability of each geometric transform
        p_spectral: Probability of each spectral transform

    Returns:
        Albumentations Compose object
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for augmentation. Install with: pip install albumentations")

    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=p_geometric),
        A.VerticalFlip(p=p_geometric),
        A.RandomRotate90(p=p_geometric),

        # Spectral augmentations (applied per-channel for multi-spectral)
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=p_spectral,
        ),
        A.GaussNoise(
            var_limit=noise_var_limit,
            p=p_spectral,
        ),
    ])


def get_strong_augmentation(
    brightness_limit: float = 0.15,
    contrast_limit: float = 0.15,
    noise_var_limit: tuple[float, float] = (0.0, 0.02),
    p_geometric: float = 0.7,
    p_spectral: float = 0.5,
    elastic_alpha: float = 100.0,
    elastic_sigma: float = 10.0,
    p_elastic: float = 0.3,
) -> "A.Compose":
    """
    Stronger augmentation for fire patches (class imbalance Strategy 5).

    Includes all standard augmentations plus:
    - Higher probabilities
    - Elastic deformation
    - Stronger brightness/contrast

    Args:
        brightness_limit: Max brightness change (default ±15%)
        contrast_limit: Max contrast change (default ±15%)
        noise_var_limit: Gaussian noise variance range
        p_geometric: Probability of each geometric transform
        p_spectral: Probability of each spectral transform
        elastic_alpha: Elastic deformation intensity
        elastic_sigma: Elastic deformation smoothness
        p_elastic: Probability of elastic deformation

    Returns:
        Albumentations Compose object
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for augmentation. Install with: pip install albumentations")

    return A.Compose([
        # Geometric augmentations (higher probability)
        A.HorizontalFlip(p=p_geometric),
        A.VerticalFlip(p=p_geometric),
        A.RandomRotate90(p=p_geometric),

        # Elastic deformation (fire-specific, creates variation in fire shapes)
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            p=p_elastic,
        ),

        # Spectral augmentations (stronger)
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=p_spectral,
        ),
        A.GaussNoise(
            var_limit=noise_var_limit,
            p=p_spectral,
        ),
    ])


def get_validation_augmentation() -> None:
    """
    No augmentation for validation/test sets.
    Returns None to skip augmentation.
    """
    return None


# =============================================================================
# Dataset Classes
# =============================================================================

class WildfirePatchDataset(Dataset):
    """PyTorch Dataset for loading wildfire patches with augmentation support.

    Loads pre-generated .npy patches from the patch generation pipeline.
    Supports Albumentations for on-the-fly data augmentation.

    Args:
        patches_dir: Directory containing *_image.npy and *_mask.npy files
        augment: Albumentations Compose object for augmentation (applied to numpy arrays)
        transform: Optional torchvision transform for images (applied after tensor conversion)
        target_transform: Optional torchvision transform for masks (applied after tensor conversion)
        return_metadata: If True, also return patch_id with each sample
        fire_augment: Stronger augmentation for patches containing fire (optional)
        fire_threshold: Minimum fraction of fire pixels to trigger fire_augment (default 0.01 = 1%)

    Example:
        # Basic usage (no augmentation)
        dataset = WildfirePatchDataset("./patches/train")

        # With standard augmentation
        augment = get_training_augmentation()
        dataset = WildfirePatchDataset("./patches/train", augment=augment)

        # With fire-aware augmentation (stronger for fire patches)
        dataset = WildfirePatchDataset(
            "./patches/train",
            augment=get_training_augmentation(),
            fire_augment=get_strong_augmentation(),
            fire_threshold=0.01,
        )
    """

    def __init__(
        self,
        patches_dir: Path | str,
        augment: "A.Compose | None" = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        return_metadata: bool = False,
        fire_augment: "A.Compose | None" = None,
        fire_threshold: float = 0.01,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for WildfirePatchDataset")

        self.patches_dir = Path(patches_dir)
        self.augment = augment
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = return_metadata
        self.fire_augment = fire_augment
        self.fire_threshold = fire_threshold

        # Find all image patches
        self.patch_ids = self._find_patches()

        if len(self.patch_ids) == 0:
            raise ValueError(f"No patches found in {patches_dir}")

        # Cache fire fractions for fire-aware augmentation and weighted sampling
        self._fire_fractions: np.ndarray | None = None

    def _find_patches(self) -> list[str]:
        """Find all patch IDs in the directory."""
        patch_ids = []
        for f in self.patches_dir.glob("*_image.npy"):
            patch_id = f.stem.replace("_image", "")
            mask_file = self.patches_dir / f"{patch_id}_mask.npy"
            if mask_file.exists():
                patch_ids.append(patch_id)
        return sorted(patch_ids)

    def _compute_fire_fractions(self) -> np.ndarray:
        """Compute fire pixel fraction for each patch (lazy loading)."""
        if self._fire_fractions is None:
            fractions = []
            for patch_id in self.patch_ids:
                mask = np.load(self.patches_dir / f"{patch_id}_mask.npy")
                fire_fraction = (mask > 0).sum() / mask.size
                fractions.append(fire_fraction)
            self._fire_fractions = np.array(fractions)
        return self._fire_fractions

    @property
    def fire_fractions(self) -> np.ndarray:
        """Fire pixel fraction for each patch."""
        return self._compute_fire_fractions()

    def get_sample_weights(self, fire_weight: float = 10.0) -> np.ndarray:
        """
        Get sample weights for WeightedRandomSampler.

        Patches with fire get higher weight to address class imbalance.

        Args:
            fire_weight: Weight multiplier for patches containing fire

        Returns:
            Array of weights, one per patch
        """
        fractions = self.fire_fractions
        weights = np.ones(len(fractions))
        weights[fractions > self.fire_threshold] = fire_weight
        return weights

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int):
        patch_id = self.patch_ids[idx]

        # Load image and mask as numpy arrays
        image = np.load(self.patches_dir / f"{patch_id}_image.npy")  # (H, W, C)
        mask = np.load(self.patches_dir / f"{patch_id}_mask.npy")    # (H, W)

        # Apply Albumentations augmentation (on numpy arrays, before tensor conversion)
        if self.augment is not None or self.fire_augment is not None:
            # Determine which augmentation to use
            fire_fraction = (mask > 0).sum() / mask.size

            if self.fire_augment is not None and fire_fraction > self.fire_threshold:
                # Use stronger augmentation for fire patches
                aug = self.fire_augment
            elif self.augment is not None:
                # Use standard augmentation
                aug = self.augment
            else:
                aug = None

            if aug is not None:
                augmented = aug(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

        # Convert to torch tensors
        # Image: (H, W, C) -> (C, H, W) for PyTorch
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.copy()).long()

        # Apply torchvision transforms (optional, for normalization etc.)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.return_metadata:
            return image, mask, patch_id
        return image, mask


class WildfireDataModule:
    """Data module for managing train/val/test splits with augmentation.

    Args:
        patches_root: Root directory containing train/val/test subdirectories
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        train_augment: Augmentation for training (default: standard augmentation)
        val_augment: Augmentation for validation (default: None)
        fire_augment: Stronger augmentation for fire patches (optional)
        fire_threshold: Minimum fire fraction to trigger fire_augment
        transform: Optional torchvision transform for images (e.g., normalization)
        target_transform: Optional torchvision transform for masks
        use_weighted_sampling: Use weighted sampling to oversample fire patches
        fire_sample_weight: Weight for fire patches in weighted sampling

    Example:
        # Basic usage with default augmentation
        dm = WildfireDataModule("./patches", batch_size=32)

        # With fire-aware augmentation and weighted sampling
        dm = WildfireDataModule(
            "./patches",
            batch_size=32,
            train_augment=get_training_augmentation(),
            fire_augment=get_strong_augmentation(),
            use_weighted_sampling=True,
            fire_sample_weight=5.0,
        )

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
    """

    def __init__(
        self,
        patches_root: Path | str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_augment: "A.Compose | None | str" = "default",
        val_augment: "A.Compose | None" = None,
        fire_augment: "A.Compose | None" = None,
        fire_threshold: float = 0.01,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        use_weighted_sampling: bool = False,
        fire_sample_weight: float = 5.0,
        drop_last: bool = False,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for WildfireDataModule")

        self.patches_root = Path(patches_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.target_transform = target_transform
        self.fire_threshold = fire_threshold
        self.use_weighted_sampling = use_weighted_sampling
        self.fire_sample_weight = fire_sample_weight
        self.drop_last = drop_last

        # Handle default augmentation
        if train_augment == "default":
            self.train_augment = get_training_augmentation() if ALBUMENTATIONS_AVAILABLE else None
        else:
            self.train_augment = train_augment
        self.val_augment = val_augment
        self.fire_augment = fire_augment

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def _create_dataset(
        self,
        split: str,
        augment: "A.Compose | None" = None,
        fire_augment: "A.Compose | None" = None,
    ) -> WildfirePatchDataset:
        """Create dataset for a given split."""
        split_dir = self.patches_root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        return WildfirePatchDataset(
            split_dir,
            augment=augment,
            transform=self.transform,
            target_transform=self.target_transform,
            fire_augment=fire_augment,
            fire_threshold=self.fire_threshold,
        )

    @property
    def train_dataset(self) -> WildfirePatchDataset:
        if self._train_dataset is None:
            self._train_dataset = self._create_dataset(
                "train",
                augment=self.train_augment,
                fire_augment=self.fire_augment,
            )
        return self._train_dataset

    @property
    def val_dataset(self) -> WildfirePatchDataset:
        if self._val_dataset is None:
            self._val_dataset = self._create_dataset(
                "val",
                augment=self.val_augment,
            )
        return self._val_dataset

    @property
    def test_dataset(self) -> WildfirePatchDataset:
        if self._test_dataset is None:
            self._test_dataset = self._create_dataset(
                "test",
                augment=None,  # Never augment test set
            )
        return self._test_dataset

    def train_dataloader(self) -> DataLoader:
        sampler = None
        shuffle = True

        if self.use_weighted_sampling:
            weights = self.train_dataset.get_sample_weights(self.fire_sample_weight)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False  # Sampler handles randomization

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# =============================================================================
# Utility Functions
# =============================================================================

def compute_dataset_statistics(patches_dir: Path | str, batch_size: int = 100) -> dict:
    """Compute mean and std for normalization.

    Uses batched computation for efficiency - processes multiple patches at once
    rather than pixel-by-pixel.

    Args:
        patches_dir: Directory containing *_image.npy files
        batch_size: Number of patches to process at once (for memory efficiency)

    Returns:
        Dictionary with 'mean' and 'std' arrays of shape (7,)
    """
    patches_dir = Path(patches_dir)
    image_files = list(patches_dir.glob("*_image.npy"))

    if not image_files:
        raise ValueError(f"No image files found in {patches_dir}")

    print(f"Computing statistics from {len(image_files)} patches...")

    # Two-pass algorithm: first compute mean, then compute variance
    # This is much faster than per-pixel Welford while being numerically stable
    
    # Pass 1: Compute mean
    total_sum = np.zeros(7, dtype=np.float64)
    total_pixels = 0
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_data = [np.load(f) for f in batch_files]
        batch_array = np.stack(batch_data, axis=0)  # (batch, 256, 256, 7)
        
        # Sum over all spatial dimensions and batch
        total_sum += batch_array.sum(axis=(0, 1, 2))
        total_pixels += batch_array.shape[0] * batch_array.shape[1] * batch_array.shape[2]
    
    mean = total_sum / total_pixels
    
    # Pass 2: Compute variance
    total_sq_diff = np.zeros(7, dtype=np.float64)
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_data = [np.load(f) for f in batch_files]
        batch_array = np.stack(batch_data, axis=0)  # (batch, 256, 256, 7)
        
        # Compute squared differences from mean
        diff = batch_array - mean  # Broadcasting
        total_sq_diff += (diff ** 2).sum(axis=(0, 1, 2))
    
    variance = total_sq_diff / total_pixels
    std = np.sqrt(variance)

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "n_patches": len(image_files),
        "n_pixels": total_pixels,
    }


def get_patch_num_channels(patches_dir: Path | str, split: str = "train") -> int:
    """
    Infer number of input channels from patch files (7 or 8).

    Loads one *_image.npy from patches_dir/split and returns shape[-1].
    Use this so the model matches your patch format (with or without NDVI).

    Args:
        patches_dir: Root directory containing train/val/test subdirs
        split: Subdir name (train, val, or test)

    Returns:
        Number of channels (7 or 8)

    Raises:
        ValueError: If no patch found or shape is invalid
    """
    patches_dir = Path(patches_dir)
    split_dir = patches_dir / split
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    image_files = list(split_dir.glob("*_image.npy"))
    if not image_files:
        raise ValueError(f"No *_image.npy files in {split_dir}")
    img = np.load(image_files[0])
    if img.ndim != 3:
        raise ValueError(f"Expected 3D patch (H, W, C), got shape {img.shape}")
    return int(img.shape[-1])


def compute_class_weights(patches_dir: Path | str, num_classes: int = 2) -> np.ndarray:
    """
    Compute class weights for loss function based on pixel frequency.

    Uses inverse frequency weighting: weight = total_pixels / (num_classes * class_pixels)

    Args:
        patches_dir: Directory containing *_mask.npy files
        num_classes: Number of classes (2 for DEL, 5 for GRA)

    Returns:
        Array of shape (num_classes,) with class weights
    """
    patches_dir = Path(patches_dir)
    mask_files = list(patches_dir.glob("*_mask.npy"))

    if not mask_files:
        raise ValueError(f"No mask files found in {patches_dir}")

    print(f"Computing class weights from {len(mask_files)} patches...")

    class_counts = np.zeros(num_classes, dtype=np.int64)

    for f in mask_files:
        mask = np.load(f)
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum()

    total_pixels = class_counts.sum()

    # Inverse frequency weighting
    weights = total_pixels / (num_classes * class_counts + 1e-6)

    # Normalize so minimum weight is 1.0
    weights = weights / weights.min()

    print(f"Class counts: {class_counts}")
    print(f"Class weights: {weights}")

    return weights.astype(np.float32)


if __name__ == "__main__":
    # Test the dataset
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <patches_dir>")
        sys.exit(1)

    patches_dir = Path(sys.argv[1])

    # Compute statistics
    stats = compute_dataset_statistics(patches_dir)
    print(f"\nDataset statistics ({stats['n_patches']} patches, {stats['n_pixels']:,} pixels):")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std:  {stats['std']}")

    # Test dataset loading
    if TORCH_AVAILABLE:
        print("\n--- Testing without augmentation ---")
        dataset = WildfirePatchDataset(patches_dir)
        print(f"Dataset size: {len(dataset)}")

        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")

        # Test with augmentation
        if ALBUMENTATIONS_AVAILABLE:
            print("\n--- Testing with augmentation ---")
            augment = get_training_augmentation()
            dataset_aug = WildfirePatchDataset(patches_dir, augment=augment)

            image_aug, mask_aug = dataset_aug[0]
            print(f"Augmented image shape: {image_aug.shape}")
            print(f"Augmented mask shape: {mask_aug.shape}")

            # Test fire fractions
            print(f"\nFire fractions (first 10): {dataset_aug.fire_fractions[:10]}")

            # Test sample weights
            weights = dataset_aug.get_sample_weights(fire_weight=10.0)
            print(f"Sample weights (first 10): {weights[:10]}")

            # Compute class weights
            print("\n--- Computing class weights ---")
            class_weights = compute_class_weights(patches_dir, num_classes=2)
