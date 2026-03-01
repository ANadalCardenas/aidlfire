"""Unit tests for storage system."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from storage import StorageManager, AnalysisRecord
from satellite_fetcher import SatelliteImage
from inference import InferenceResult


class TestStorageManager:
    """Tests for StorageManager."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create temporary storage manager."""
        return StorageManager(tmp_path)

    @pytest.fixture
    def mock_satellite_image(self):
        """Create mock satellite image."""
        return SatelliteImage(
            data=np.random.rand(256, 256, 7).astype(np.float32),
            bounds=(-122.5, 37.5, -122.0, 38.0),
            crs="EPSG:4326",
            datetime=datetime.now(),
            scene_id="TEST_SCENE_001",
            cloud_cover=5.0,
            platform="Test-Platform",
        )

    @pytest.fixture
    def mock_inference_result(self):
        """Create mock inference result."""
        return InferenceResult(
            segmentation=np.random.randint(0, 2, (256, 256), dtype=np.uint8),
            probabilities=np.random.rand(256, 256, 2).astype(np.float32),
            has_fire=True,
            fire_confidence=0.85,
            fire_fraction=0.12,
            severity_counts={"background": 225000, "fire": 31000},
            num_classes=2,
            image_shape=(256, 256),
        )

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates required directories."""
        storage = StorageManager(tmp_path)

        assert (tmp_path / "images").exists()
        assert (tmp_path / "results").exists()
        assert (tmp_path / "visualizations").exists()
        assert (tmp_path / "database.db").exists()

    def test_save_and_load_analysis(
        self, storage, mock_satellite_image, mock_inference_result
    ):
        """Test saving and loading an analysis."""
        analysis_id = storage.save_analysis(
            satellite_image=mock_satellite_image,
            inference_result=mock_inference_result,
            metadata={"test": True},
        )

        assert analysis_id is not None
        assert len(analysis_id) == 12  # UUID length

        # Load and verify
        loaded = storage.load_analysis(analysis_id)

        assert loaded is not None
        assert loaded["record"].id == analysis_id
        assert loaded["record"].has_fire is True
        assert loaded["record"].fire_confidence == pytest.approx(0.85)
        assert loaded["image"].shape == (256, 256, 7)
        assert loaded["segmentation"].shape == (256, 256)

    def test_save_with_visualization(
        self, storage, mock_satellite_image, mock_inference_result
    ):
        """Test saving with visualization."""
        visualization = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        analysis_id = storage.save_analysis(
            satellite_image=mock_satellite_image,
            inference_result=mock_inference_result,
            visualization=visualization,
        )

        loaded = storage.load_analysis(analysis_id)

        assert "visualization" in loaded
        assert loaded["visualization"].shape == (256, 256, 3)

    def test_get_record(self, storage, mock_satellite_image, mock_inference_result):
        """Test getting a record by ID."""
        analysis_id = storage.save_analysis(
            satellite_image=mock_satellite_image,
            inference_result=mock_inference_result,
        )

        record = storage.get_record(analysis_id)

        assert record is not None
        assert isinstance(record, AnalysisRecord)
        assert record.scene_id == "TEST_SCENE_001"
        assert record.cloud_cover == 5.0

    def test_get_record_not_found(self, storage):
        """Test getting non-existent record."""
        record = storage.get_record("nonexistent")
        assert record is None

    def test_get_history(self, storage, mock_satellite_image, mock_inference_result):
        """Test getting analysis history."""
        # Save multiple analyses
        for i in range(5):
            storage.save_analysis(
                satellite_image=mock_satellite_image,
                inference_result=mock_inference_result,
            )

        history = storage.get_history(limit=3)

        assert len(history) == 3

    def test_get_history_fire_only(
        self, storage, mock_satellite_image, mock_inference_result
    ):
        """Test filtering history by fire detection."""
        # Save fire analysis
        storage.save_analysis(
            satellite_image=mock_satellite_image,
            inference_result=mock_inference_result,  # has_fire=True
        )

        # Save no-fire analysis
        no_fire_result = InferenceResult(
            segmentation=np.zeros((256, 256), dtype=np.uint8),
            probabilities=np.zeros((256, 256, 2), dtype=np.float32),
            has_fire=False,
            fire_confidence=0.1,
            fire_fraction=0.0,
            severity_counts={},
            num_classes=2,
            image_shape=(256, 256),
        )
        storage.save_analysis(
            satellite_image=mock_satellite_image,
            inference_result=no_fire_result,
        )

        fire_history = storage.get_history(fire_only=True)

        assert len(fire_history) == 1
        assert fire_history[0].has_fire is True

    def test_delete_analysis(self, storage, mock_satellite_image, mock_inference_result):
        """Test deleting an analysis."""
        analysis_id = storage.save_analysis(
            satellite_image=mock_satellite_image,
            inference_result=mock_inference_result,
        )

        # Verify it exists
        assert storage.get_record(analysis_id) is not None

        # Delete
        result = storage.delete_analysis(analysis_id)

        assert result is True
        assert storage.get_record(analysis_id) is None

    def test_delete_nonexistent(self, storage):
        """Test deleting non-existent analysis."""
        result = storage.delete_analysis("nonexistent")
        assert result is False

    def test_get_statistics(self, storage, mock_satellite_image, mock_inference_result):
        """Test getting statistics."""
        # Save some analyses
        for _ in range(3):
            storage.save_analysis(
                satellite_image=mock_satellite_image,
                inference_result=mock_inference_result,
            )

        stats = storage.get_statistics()

        assert stats["total_analyses"] == 3
        assert stats["fire_detections"] == 3  # All have fire
        assert stats["avg_confidence"] > 0

    def test_safe_delete_path_traversal(self, storage, tmp_path):
        """Test that path traversal attempts are blocked."""
        # Create a file outside storage directory
        outside_file = tmp_path.parent / "outside_file.txt"
        outside_file.write_text("should not be deleted")

        # Attempt path traversal
        result = storage._safe_delete(
            storage.images_dir,
            "../../../outside_file.txt"
        )

        assert result is False
        assert outside_file.exists()  # File should still exist

    def test_save_and_load_dual_head_analysis(
        self, storage, mock_satellite_image
    ):
        """Test saving and loading an analysis with dual-head (binary + severity)."""
        from inference import InferenceResult

        h, w = 64, 64
        segmentation = np.random.randint(0, 2, (h, w), dtype=np.uint8)
        probabilities = np.random.rand(h, w, 2).astype(np.float32)
        binary_seg = np.random.randint(0, 2, (h, w), dtype=np.uint8)
        severity_seg = np.random.randint(0, 5, (h, w), dtype=np.uint8)
        binary_probs = np.random.rand(h, w, 2).astype(np.float32)
        severity_probs = np.random.rand(h, w, 5).astype(np.float32)

        result = InferenceResult(
            segmentation=segmentation,
            probabilities=probabilities,
            has_fire=True,
            fire_confidence=0.8,
            fire_fraction=0.1,
            severity_counts={"background": 3000, "fire": 1000},
            num_classes=2,
            image_shape=(h, w),
            dual_head=True,
            binary_segmentation=binary_seg,
            severity_segmentation=severity_seg,
            binary_probabilities=binary_probs,
            severity_probabilities=severity_probs,
        )

        # Resize mock image to match
        mock_satellite_image.data = np.random.rand(h, w, 7).astype(np.float32)
        vis = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        analysis_id = storage.save_analysis(
            satellite_image=mock_satellite_image,
            inference_result=result,
            visualization=vis,
        )

        loaded = storage.load_analysis(analysis_id)
        assert loaded is not None
        assert "binary_segmentation" in loaded
        assert "severity_segmentation" in loaded
        np.testing.assert_array_equal(loaded["binary_segmentation"], binary_seg)
        np.testing.assert_array_equal(loaded["severity_segmentation"], severity_seg)
        assert loaded["record"].metadata is not None
        assert loaded["record"].metadata.get("dual_head") is True


class TestAnalysisRecord:
    """Tests for AnalysisRecord dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = AnalysisRecord(
            id="test123",
            created_at=datetime(2024, 1, 15, 12, 0, 0),
            satellite_datetime=datetime(2024, 1, 14, 10, 30, 0),
            scene_id="SCENE_001",
            bbox=(-122.5, 37.5, -122.0, 38.0),
            center_lon=-122.25,
            center_lat=37.75,
            has_fire=True,
            fire_confidence=0.9,
            fire_fraction=0.15,
            cloud_cover=5.0,
            platform="Sentinel-2A",
            num_classes=2,
        )

        d = record.to_dict()

        assert d["id"] == "test123"
        assert d["has_fire"] is True
        assert d["bbox"] == [-122.5, 37.5, -122.0, 38.0]
        assert "2024-01-15" in d["created_at"]

    def test_optional_fields(self):
        """Test that optional fields default correctly."""
        record = AnalysisRecord(
            id="test123",
            created_at=datetime.now(),
            satellite_datetime=datetime.now(),
            scene_id="SCENE_001",
            bbox=(-122.5, 37.5, -122.0, 38.0),
            center_lon=-122.25,
            center_lat=37.75,
            has_fire=False,
            fire_confidence=0.0,
            fire_fraction=0.0,
            cloud_cover=0.0,
            platform="Test",
            num_classes=2,
        )

        assert record.image_path is None
        assert record.result_path is None
        assert record.visualization_path is None
        assert record.metadata is None
