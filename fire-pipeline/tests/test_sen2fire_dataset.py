"""Tests for Sen2Fire dataset loader."""

from pathlib import Path

import numpy as np
import pytest

from sen2fire_dataset import (
    SEN2FIRE_7BAND_INDICES,
    SEN2FIRE_SPLIT_SCENES,
    Sen2FireDataset,
    _load_sen2fire_npz,
)


class TestLoadSen2FireNpz:
    """Tests for single .npz loading."""

    def test_load_npz_shape_7ch(self, tmp_path):
        """Load npz with 12 bands, get 7-channel 256 crop."""
        image_12 = np.random.randint(0, 8000, (12, 512, 512), dtype=np.int16)
        label = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
        npz = tmp_path / "test.npz"
        np.savez(npz, image=image_12, label=label)

        img, mask = _load_sen2fire_npz(npz, include_ndvi=False, crop_size=256)

        assert img.shape == (256, 256, 7)
        assert mask.shape == (256, 256)
        assert img.dtype == np.float32
        assert mask.dtype == np.int64
        assert img.min() >= 0 and img.max() <= 1

    def test_load_npz_with_ndvi(self, tmp_path):
        """With include_ndvi=True output has 8 channels."""
        image_12 = np.random.randint(0, 8000, (12, 512, 512), dtype=np.int16)
        label = np.zeros((512, 512), dtype=np.uint8)
        np.savez(tmp_path / "t.npz", image=image_12, label=label)

        img, mask = _load_sen2fire_npz(tmp_path / "t.npz", include_ndvi=True, crop_size=256)

        assert img.shape == (256, 256, 8)
        assert mask.shape == (256, 256)


class TestSen2FireDataset:
    """Tests for Sen2FireDataset."""

    @pytest.fixture
    def sen2fire_root(self, tmp_path):
        """Create minimal Sen2Fire dir with scene1, scene2, scene3, scene4."""
        for scene in ["scene1", "scene2", "scene3", "scene4"]:
            d = tmp_path / scene
            d.mkdir()
            for i in range(3):
                image_12 = np.random.randint(0, 8000, (12, 512, 512), dtype=np.int16)
                label = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
                np.savez(d / f"patch_{i}.npz", image=image_12, label=label)
        return tmp_path

    def test_splits(self, sen2fire_root):
        """Train includes scene1+2, val scene3, test scene4."""
        train_ds = Sen2FireDataset(sen2fire_root, split="train", max_cloud_score=None)
        val_ds = Sen2FireDataset(sen2fire_root, split="val", max_cloud_score=None)
        test_ds = Sen2FireDataset(sen2fire_root, split="test", max_cloud_score=None)

        assert len(train_ds) == 6  # 3 + 3
        assert len(val_ds) == 3
        assert len(test_ds) == 3

    def test_getitem_shape(self, sen2fire_root):
        """Each item is (C, 256, 256) image and (256, 256) mask."""
        ds = Sen2FireDataset(sen2fire_root, split="train", max_cloud_score=None, include_ndvi=True)
        img, mask = ds[0]

        assert img.shape == (8, 256, 256)
        assert mask.shape == (256, 256)

    def test_cloud_filter_reduces_samples(self, sen2fire_root):
        """With max_cloud_score some patches may be filtered out."""
        ds_no_filter = Sen2FireDataset(sen2fire_root, split="train", max_cloud_score=None)
        ds_filter = Sen2FireDataset(
            sen2fire_root, split="train", max_cloud_score=0.5, use_s2cloudless=False
        )
        # Rule-based on random data may keep or drop; we only require filter runs
        assert len(ds_filter) <= len(ds_no_filter)

    def test_no_ndvi_7_channels(self, sen2fire_root):
        """include_ndvi=False gives 7 channels."""
        ds = Sen2FireDataset(
            sen2fire_root, split="train", max_cloud_score=None, include_ndvi=False
        )
        img, _ = ds[0]
        assert img.shape[0] == 7


class TestConstants:
    """Test module constants."""

    def test_7band_indices(self):
        """7-band indices match CEMS band selection."""
        assert len(SEN2FIRE_7BAND_INDICES) == 7
        assert SEN2FIRE_7BAND_INDICES == [1, 2, 3, 7, 8, 10, 11]

    def test_split_scenes(self):
        """Splits map to scene folders."""
        assert SEN2FIRE_SPLIT_SCENES["train"] == ["scene1", "scene2"]
        assert SEN2FIRE_SPLIT_SCENES["val"] == ["scene3"]
        assert SEN2FIRE_SPLIT_SCENES["test"] == ["scene4"]
