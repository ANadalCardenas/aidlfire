"""Unit tests for inference pipeline."""

import numpy as np
import pytest
import torch

from inference import (
    FireInferencePipeline,
    InferenceResult,
    create_visualization,
    create_visualization_from_segmentation,
)


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_get_fire_mask_binary(self):
        """Test binary fire mask extraction."""
        segmentation = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        probabilities = np.random.rand(3, 3, 2).astype(np.float32)

        result = InferenceResult(
            segmentation=segmentation,
            probabilities=probabilities,
            has_fire=True,
            fire_confidence=0.9,
            fire_fraction=0.33,
            severity_counts={"background": 6, "fire": 3},
            num_classes=2,
            image_shape=(3, 3),
        )

        fire_mask = result.get_fire_mask()
        expected = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(fire_mask, expected)

    def test_get_fire_mask_multiclass(self):
        """Test fire mask with severity classes."""
        segmentation = np.array([[0, 1, 2], [3, 4, 0], [0, 0, 0]], dtype=np.uint8)
        probabilities = np.random.rand(3, 3, 5).astype(np.float32)

        result = InferenceResult(
            segmentation=segmentation,
            probabilities=probabilities,
            has_fire=True,
            fire_confidence=0.9,
            fire_fraction=0.44,
            severity_counts={},
            num_classes=5,
            image_shape=(3, 3),
        )

        fire_mask = result.get_fire_mask()
        expected = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(fire_mask, expected)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = InferenceResult(
            segmentation=np.zeros((3, 3), dtype=np.uint8),
            probabilities=np.zeros((3, 3, 2), dtype=np.float32),
            has_fire=False,
            fire_confidence=0.1,
            fire_fraction=0.0,
            severity_counts={"background": 9, "fire": 0},
            num_classes=2,
            image_shape=(3, 3),
        )

        d = result.to_dict()
        assert d["has_fire"] is False
        assert d["fire_confidence"] == 0.1
        assert d["num_classes"] == 2
        assert "segmentation" not in d  # Arrays not serialized


class TestFireInferencePipelinePreprocessing:
    """Tests for preprocessing functions."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline without loading model."""
        # Create a minimal mock pipeline for testing preprocessing
        pipeline = object.__new__(FireInferencePipeline)
        pipeline.patch_size = 256
        pipeline.stride = 256
        pipeline.BAND_INDICES = (1, 2, 3, 7, 8, 10, 11)
        pipeline.config = {"in_channels": 7}
        return pipeline

    def test_preprocess_uint16(self, pipeline):
        """Test preprocessing of uint16 Sentinel-2 data."""
        # Sentinel-2 L2A data is typically 0-10000
        image = np.random.randint(0, 10000, (100, 100, 7), dtype=np.uint16)
        result = pipeline.preprocess_image(image, select_bands=False)

        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_uint8(self, pipeline):
        """Test preprocessing of uint8 data."""
        image = np.random.randint(0, 255, (100, 100, 7), dtype=np.uint8)
        result = pipeline.preprocess_image(image, select_bands=False)

        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_band_selection(self, pipeline):
        """Test band selection from 12-band input."""
        image = np.random.rand(100, 100, 12).astype(np.float32)
        result = pipeline.preprocess_image(image, select_bands=True)

        assert result.shape == (100, 100, 7)

    def test_preprocess_wrong_channels_raises(self, pipeline):
        """Test that wrong number of channels raises error."""
        image = np.random.rand(100, 100, 5).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 7, 8, or 12 channels"):
            pipeline.preprocess_image(image, select_bands=True)


class TestPatchExtraction:
    """Tests for patch extraction and stitching."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for patch testing."""
        pipeline = object.__new__(FireInferencePipeline)
        pipeline.patch_size = 64
        pipeline.stride = 64
        return pipeline

    def test_extract_patches_exact_fit(self, pipeline):
        """Test extraction when image fits exactly."""
        image = np.random.rand(128, 128, 7).astype(np.float32)
        patches, positions = pipeline.extract_patches(image)

        assert patches.shape == (4, 64, 64, 7)  # 2x2 grid
        assert len(positions) == 4
        assert positions[0] == (0, 0)
        assert positions[-1] == (64, 64)

    def test_extract_patches_with_remainder(self, pipeline):
        """Test extraction with non-exact fit."""
        image = np.random.rand(100, 100, 7).astype(np.float32)
        patches, positions = pipeline.extract_patches(image)

        # Should have edge patches
        assert len(patches) > 1
        assert all(p.shape == (64, 64, 7) for p in patches)

    def test_extract_patches_too_small_raises(self, pipeline):
        """Test that small images raise error."""
        image = np.random.rand(32, 32, 7).astype(np.float32)

        with pytest.raises(ValueError, match="smaller than patch_size"):
            pipeline.extract_patches(image)

    def test_stitch_predictions(self, pipeline):
        """Test stitching predictions back together."""
        # Create mock predictions for a 2x2 grid
        predictions = np.random.rand(4, 2, 64, 64).astype(np.float32)
        positions = [(0, 0), (0, 64), (64, 0), (64, 64)]
        output_shape = (128, 128)

        result = pipeline.stitch_predictions(predictions, positions, output_shape)

        assert result.shape == (2, 128, 128)

    def test_stitch_predictions_averaging(self, pipeline):
        """Test that overlapping regions are averaged."""
        pipeline.stride = 32  # 50% overlap

        # Two overlapping patches with known values
        pred1 = np.ones((1, 2, 64, 64)) * 0.2
        pred2 = np.ones((1, 2, 64, 64)) * 0.8
        positions = [(0, 0), (0, 32)]
        output_shape = (64, 96)

        result = pipeline.stitch_predictions(
            np.concatenate([pred1, pred2], axis=0),
            positions,
            output_shape,
        )

        # Non-overlapping region should have original values
        assert np.allclose(result[:, :, :32], 0.2, atol=0.01)
        # Overlapping region should be averaged
        assert np.allclose(result[:, :, 32:64], 0.5, atol=0.01)
        # Other non-overlapping region
        assert np.allclose(result[:, :, 64:], 0.8, atol=0.01)


class TestVisualization:
    """Tests for visualization functions."""

    def test_create_visualization_binary(self):
        """Test visualization with binary segmentation."""
        image = np.random.rand(64, 64, 7).astype(np.float32)
        segmentation = np.zeros((64, 64), dtype=np.uint8)
        segmentation[20:40, 20:40] = 1  # Fire region

        result = InferenceResult(
            segmentation=segmentation,
            probabilities=np.random.rand(64, 64, 2).astype(np.float32),
            has_fire=True,
            fire_confidence=0.9,
            fire_fraction=0.1,
            severity_counts={},
            num_classes=2,
            image_shape=(64, 64),
        )

        vis = create_visualization(image, result, alpha=0.5)

        assert vis.shape == (64, 64, 3)
        assert vis.dtype == np.uint8

    def test_create_visualization_multiclass(self):
        """Test visualization with severity classes."""
        image = np.random.rand(64, 64, 7).astype(np.float32)
        segmentation = np.zeros((64, 64), dtype=np.uint8)
        segmentation[10:20, 10:20] = 1
        segmentation[30:40, 30:40] = 3

        result = InferenceResult(
            segmentation=segmentation,
            probabilities=np.random.rand(64, 64, 5).astype(np.float32),
            has_fire=True,
            fire_confidence=0.9,
            fire_fraction=0.1,
            severity_counts={},
            num_classes=5,
            image_shape=(64, 64),
        )

        vis = create_visualization(image, result, alpha=0.5)

        assert vis.shape == (64, 64, 3)
        assert vis.dtype == np.uint8

    def test_inference_result_dual_head(self):
        """InferenceResult with dual_head uses binary_segmentation for get_fire_mask."""
        binary_seg = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
        severity_seg = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.uint8)
        result = InferenceResult(
            segmentation=binary_seg,
            probabilities=np.random.rand(2, 3, 2).astype(np.float32),
            has_fire=True,
            fire_confidence=0.9,
            fire_fraction=0.5,
            severity_counts={},
            num_classes=2,
            image_shape=(2, 3),
            dual_head=True,
            binary_segmentation=binary_seg,
            severity_segmentation=severity_seg,
            binary_probabilities=np.random.rand(2, 3, 2).astype(np.float32),
            severity_probabilities=np.random.rand(2, 3, 5).astype(np.float32),
        )
        fire_mask = result.get_fire_mask()
        np.testing.assert_array_equal(fire_mask, binary_seg)
        severity_map = result.get_severity_map()
        np.testing.assert_array_equal(severity_map, severity_seg)

    def test_create_visualization_from_segmentation(self):
        """create_visualization_from_segmentation returns RGB overlay."""
        image = np.random.rand(32, 32, 7).astype(np.float32)
        segmentation = np.zeros((32, 32), dtype=np.uint8)
        segmentation[10:20, 10:20] = 1
        vis = create_visualization_from_segmentation(
            image, segmentation, num_classes=2, alpha=0.5
        )
        assert vis.shape == (32, 32, 3)
        assert vis.dtype == np.uint8
