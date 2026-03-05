"""Tests for cloud detection (rule-based and s2cloudless)."""

import numpy as np
import pytest

from cloud_detection import (
    cloud_score_rule_based,
    cloud_score_sen2fire_12band,
    cloud_score_7ch,
    get_cloud_fraction_sen2fire,
    get_cloud_fraction_s2cloudless,
)


class TestCloudScoreRuleBased:
    """Tests for rule-based cloud score."""

    def test_low_score_dark_patch(self):
        """Dark patch (low blue/green) should have low cloud score."""
        img = np.zeros((64, 64, 4), dtype=np.float32)
        score = cloud_score_rule_based(img, blue_index=0, green_index=1, nir_index=3)
        assert 0 <= score <= 0.1

    def test_high_score_bright_visible(self):
        """Bright blue/green patch should have high cloud score."""
        img = np.zeros((64, 64, 4), dtype=np.float32)
        img[:, :, 0] = 0.9  # blue
        img[:, :, 1] = 0.9  # green
        img[:, :, 3] = 0.1  # low NIR
        score = cloud_score_rule_based(img, blue_index=0, green_index=1, nir_index=3)
        assert score > 0.3

    def test_uint16_normalized(self):
        """Uint16 input should be normalized to [0,1] and produce valid score."""
        img = np.zeros((32, 32, 4), dtype=np.uint16)
        img[:, :, 0] = 5000
        img[:, :, 1] = 5000
        score = cloud_score_rule_based(img, blue_index=0, green_index=1, nir_index=3)
        assert 0 <= score <= 1


class TestCloudScoreSen2Fire12Band:
    """Tests for Sen2Fire 12-band cloud score."""

    def test_shape_requires_8_channels(self):
        """Expects at least 8 channels (uses blue=1, green=2, nir=7)."""
        img = np.zeros((10, 10, 12), dtype=np.float32)
        score = cloud_score_sen2fire_12band(img)
        assert 0 <= score <= 1

    def test_small_channels_returns_zero(self):
        """Less than 8 channels returns 0."""
        img = np.zeros((10, 10, 5), dtype=np.float32)
        score = cloud_score_sen2fire_12band(img)
        assert score == 0.0


class TestCloudScore7Ch:
    """Tests for 7-channel CEMS-style cloud score."""

    def test_7ch_shape(self):
        """7-channel patch produces valid score."""
        img = np.random.rand(32, 32, 7).astype(np.float32) * 0.5
        score = cloud_score_7ch(img)
        assert 0 <= score <= 1


class TestS2Cloudless:
    """Tests for s2cloudless-based cloud fraction (optional dependency)."""

    def test_get_cloud_fraction_sen2fire_returns_float_or_none(self):
        """get_cloud_fraction_sen2fire returns float in [0,1] or None."""
        img = np.clip(np.random.rand(64, 64, 12).astype(np.float32), 0, 1)
        out = get_cloud_fraction_sen2fire(img)
        if out is not None:
            assert 0 <= out <= 1
        # If s2cloudless not installed, out is None

    def test_get_cloud_fraction_sen2fire_insufficient_channels(self):
        """Less than 12 channels returns None."""
        img = np.zeros((64, 64, 10), dtype=np.float32)
        out = get_cloud_fraction_sen2fire(img)
        assert out is None

    def test_get_cloud_fraction_s2cloudless_insufficient_channels(self):
        """get_cloud_fraction_s2cloudless with <12 ch returns None."""
        img = np.zeros((64, 64, 7), dtype=np.float32)
        out = get_cloud_fraction_s2cloudless(img)
        assert out is None
