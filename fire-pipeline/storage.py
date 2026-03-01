"""
Local Storage System for Fire Detection App.

Manages caching of satellite imagery and inference results.

Storage structure:
    cache/
    ├── images/           # Raw satellite imagery (.npy)
    ├── results/          # Inference results (.npy)
    ├── visualizations/   # Generated visualizations (.png)
    └── database.db       # SQLite metadata database

Usage:
    from storage import StorageManager

    storage = StorageManager("./cache")

    # Save analysis
    analysis_id = storage.save_analysis(
        satellite_image=image,
        inference_result=result,
        visualization=vis_array,
    )

    # Load history
    history = storage.get_history(limit=20)

    # Load specific analysis
    analysis = storage.load_analysis(analysis_id)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from satellite_fetcher import SatelliteImage
    from inference import InferenceResult


@dataclass
class AnalysisRecord:
    """Record of a fire detection analysis."""

    id: str
    created_at: datetime
    satellite_datetime: datetime
    scene_id: str

    # Location
    bbox: tuple[float, float, float, float]
    center_lon: float
    center_lat: float

    # Results summary
    has_fire: bool
    fire_confidence: float
    fire_fraction: float

    # Metadata
    cloud_cover: float
    platform: str
    num_classes: int

    # File paths (relative to storage root)
    image_path: str | None = None
    result_path: str | None = None
    visualization_path: str | None = None

    # Additional metadata
    metadata: dict | None = field(default=None)

    def __post_init__(self):
        """Validate record values."""
        if not 0 <= self.fire_confidence <= 1:
            raise ValueError(f"fire_confidence must be in [0, 1], got {self.fire_confidence}")
        if not 0 <= self.fire_fraction <= 1:
            raise ValueError(f"fire_fraction must be in [0, 1], got {self.fire_fraction}")
        if self.cloud_cover < 0 or self.cloud_cover > 100:
            raise ValueError(f"cloud_cover must be in [0, 100], got {self.cloud_cover}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")
        if not -180 <= self.center_lon <= 180:
            raise ValueError(f"center_lon must be in [-180, 180], got {self.center_lon}")
        if not -90 <= self.center_lat <= 90:
            raise ValueError(f"center_lat must be in [-90, 90], got {self.center_lat}")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "satellite_datetime": self.satellite_datetime.isoformat() if self.satellite_datetime else None,
            "scene_id": self.scene_id,
            "bbox": list(self.bbox) if self.bbox else None,
            "center_lon": self.center_lon,
            "center_lat": self.center_lat,
            "has_fire": self.has_fire,
            "fire_confidence": self.fire_confidence,
            "fire_fraction": self.fire_fraction,
            "cloud_cover": self.cloud_cover,
            "platform": self.platform,
            "num_classes": self.num_classes,
            "image_path": self.image_path,
            "result_path": self.result_path,
            "visualization_path": self.visualization_path,
            "metadata": self.metadata,
        }


class StorageManager:
    """
    Manages local storage for satellite imagery and analysis results.

    Uses SQLite for metadata and filesystem for binary data.
    """

    def __init__(self, storage_dir: str | Path):
        """
        Initialize storage manager.

        Args:
            storage_dir: Root directory for storage
        """
        self.storage_dir = Path(storage_dir)

        # Create directory structure
        self.images_dir = self.storage_dir / "images"
        self.results_dir = self.storage_dir / "results"
        self.visualizations_dir = self.storage_dir / "visualizations"

        for dir_path in [self.images_dir, self.results_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db_path = self.storage_dir / "database.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    satellite_datetime TEXT,
                    scene_id TEXT,
                    bbox_west REAL,
                    bbox_south REAL,
                    bbox_east REAL,
                    bbox_north REAL,
                    center_lon REAL,
                    center_lat REAL,
                    has_fire INTEGER,
                    fire_confidence REAL,
                    fire_fraction REAL,
                    cloud_cover REAL,
                    platform TEXT,
                    num_classes INTEGER,
                    image_path TEXT,
                    result_path TEXT,
                    visualization_path TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON analyses(created_at DESC)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_has_fire ON analyses(has_fire)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_satellite_datetime ON analyses(satellite_datetime)
            """)

            conn.commit()

    def save_analysis(
        self,
        satellite_image: "SatelliteImage",
        inference_result: "InferenceResult",
        visualization: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Save a complete analysis to storage.

        Args:
            satellite_image: Source satellite image
            inference_result: Model inference results
            visualization: Optional RGB visualization array
            metadata: Optional additional metadata

        Returns:
            Analysis ID (UUID)
        """
        analysis_id = str(uuid.uuid4())[:12]
        timestamp = datetime.now()

        # Save image data
        image_filename = f"{analysis_id}_image.npy"
        image_path = self.images_dir / image_filename
        np.save(image_path, satellite_image.data)

        # Save inference result
        result_filename = f"{analysis_id}_result.npz"
        result_path = self.results_dir / result_filename
        save_dict = {
            "segmentation": inference_result.segmentation,
            "probabilities": inference_result.probabilities,
        }
        if getattr(inference_result, "dual_head", False):
            if inference_result.binary_segmentation is not None:
                save_dict["binary_segmentation"] = inference_result.binary_segmentation
            if inference_result.severity_segmentation is not None:
                save_dict["severity_segmentation"] = inference_result.severity_segmentation
            if inference_result.binary_probabilities is not None:
                save_dict["binary_probabilities"] = inference_result.binary_probabilities
            if inference_result.severity_probabilities is not None:
                save_dict["severity_probabilities"] = inference_result.severity_probabilities
        np.savez_compressed(result_path, **save_dict)

        # Save visualization if provided
        visualization_path = None
        if visualization is not None:
            from PIL import Image

            vis_filename = f"{analysis_id}_vis.png"
            visualization_path = self.visualizations_dir / vis_filename
            Image.fromarray(visualization).save(visualization_path)

        metadata = metadata or {}
        if getattr(inference_result, "dual_head", False):
            metadata["dual_head"] = True

        # Save to database
        center_lon, center_lat = satellite_image.center

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO analyses (
                    id, created_at, satellite_datetime, scene_id,
                    bbox_west, bbox_south, bbox_east, bbox_north,
                    center_lon, center_lat,
                    has_fire, fire_confidence, fire_fraction,
                    cloud_cover, platform, num_classes,
                    image_path, result_path, visualization_path,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    timestamp.isoformat(),
                    satellite_image.datetime.isoformat() if satellite_image.datetime else None,
                    satellite_image.scene_id,
                    satellite_image.bounds[0],
                    satellite_image.bounds[1],
                    satellite_image.bounds[2],
                    satellite_image.bounds[3],
                    center_lon,
                    center_lat,
                    int(inference_result.has_fire),
                    inference_result.fire_confidence,
                    inference_result.fire_fraction,
                    satellite_image.cloud_cover,
                    satellite_image.platform,
                    inference_result.num_classes,
                    str(image_filename),
                    str(result_filename),
                    str(vis_filename) if visualization_path else None,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()

        return analysis_id

    def load_analysis(self, analysis_id: str) -> dict | None:
        """
        Load a complete analysis by ID.

        Args:
            analysis_id: Analysis UUID

        Returns:
            Dictionary with record, image, result, and visualization
        """
        record = self.get_record(analysis_id)
        if record is None:
            return None

        result = {"record": record}

        # Load image
        if record.image_path:
            image_path = self.images_dir / record.image_path
            if image_path.exists():
                result["image"] = np.load(image_path)

        # Load inference result
        if record.result_path:
            result_path = self.results_dir / record.result_path
            if result_path.exists():
                data = np.load(result_path)
                result["segmentation"] = data["segmentation"]
                result["probabilities"] = data["probabilities"]
                if "binary_segmentation" in data:
                    result["binary_segmentation"] = data["binary_segmentation"]
                if "severity_segmentation" in data:
                    result["severity_segmentation"] = data["severity_segmentation"]
                if "binary_probabilities" in data:
                    result["binary_probabilities"] = data["binary_probabilities"]
                if "severity_probabilities" in data:
                    result["severity_probabilities"] = data["severity_probabilities"]

        # Load visualization
        if record.visualization_path:
            vis_path = self.visualizations_dir / record.visualization_path
            if vis_path.exists():
                from PIL import Image

                result["visualization"] = np.array(Image.open(vis_path))

        return result

    def get_record(self, analysis_id: str) -> AnalysisRecord | None:
        """Get analysis record by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM analyses WHERE id = ?",
                (analysis_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        fire_only: bool = False,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[AnalysisRecord]:
        """
        Get analysis history with optional filters.

        Args:
            limit: Maximum number of records
            offset: Number of records to skip
            fire_only: If True, only return analyses with fire detected
            start_date: Filter by analysis date (after)
            end_date: Filter by analysis date (before)
            bbox: Filter by location (overlapping bbox)

        Returns:
            List of AnalysisRecord objects
        """
        conditions = []
        params = []

        if fire_only:
            conditions.append("has_fire = 1")

        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date.isoformat())

        if bbox:
            # Check for bbox overlap
            west, south, east, north = bbox
            conditions.append("bbox_east >= ? AND bbox_west <= ?")
            conditions.append("bbox_north >= ? AND bbox_south <= ?")
            params.extend([west, east, south, north])

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT * FROM analyses
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_statistics(self) -> dict:
        """Get overall statistics from storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_analyses,
                    SUM(has_fire) as fire_detections,
                    AVG(fire_confidence) as avg_confidence,
                    AVG(cloud_cover) as avg_cloud_cover,
                    MIN(created_at) as first_analysis,
                    MAX(created_at) as last_analysis
                FROM analyses
            """)
            row = cursor.fetchone()

        return {
            "total_analyses": row[0] or 0,
            "fire_detections": row[1] or 0,
            "avg_confidence": row[2] or 0,
            "avg_cloud_cover": row[3] or 0,
            "first_analysis": row[4],
            "last_analysis": row[5],
        }

    def _safe_delete(self, base_dir: Path, relative_path: str) -> bool:
        """
        Safely delete a file, ensuring it's within the expected directory.

        Args:
            base_dir: The base directory the file should be within
            relative_path: Relative path to the file

        Returns:
            True if deleted, False if not found or invalid path
        """
        if not relative_path:
            return False

        file_path = (base_dir / relative_path).resolve()

        # Security check: ensure path is within base_dir
        try:
            file_path.relative_to(base_dir.resolve())
        except ValueError:
            # Path is outside base_dir - potential path traversal attack
            return False

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def delete_analysis(self, analysis_id: str) -> bool:
        """
        Delete an analysis and its associated files.

        Args:
            analysis_id: Analysis UUID

        Returns:
            True if deleted, False if not found
        """
        record = self.get_record(analysis_id)
        if record is None:
            return False

        # Delete files (with path validation)
        self._safe_delete(self.images_dir, record.image_path)
        self._safe_delete(self.results_dir, record.result_path)
        self._safe_delete(self.visualizations_dir, record.visualization_path)

        # Delete from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
            conn.commit()

        return True

    def clear_old_analyses(self, days: int = 30) -> int:
        """
        Delete analyses older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of analyses deleted
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Get IDs to delete
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id FROM analyses WHERE created_at < ?",
                (cutoff,),
            )
            ids_to_delete = [row[0] for row in cursor.fetchall()]

        # Delete each analysis
        for analysis_id in ids_to_delete:
            self.delete_analysis(analysis_id)

        return len(ids_to_delete)

    def _row_to_record(self, row: sqlite3.Row) -> AnalysisRecord:
        """Convert database row to AnalysisRecord."""
        return AnalysisRecord(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            satellite_datetime=datetime.fromisoformat(row["satellite_datetime"]) if row["satellite_datetime"] else None,
            scene_id=row["scene_id"],
            bbox=(
                row["bbox_west"],
                row["bbox_south"],
                row["bbox_east"],
                row["bbox_north"],
            ),
            center_lon=row["center_lon"],
            center_lat=row["center_lat"],
            has_fire=bool(row["has_fire"]),
            fire_confidence=row["fire_confidence"],
            fire_fraction=row["fire_fraction"],
            cloud_cover=row["cloud_cover"],
            platform=row["platform"],
            num_classes=row["num_classes"],
            image_path=row["image_path"],
            result_path=row["result_path"],
            visualization_path=row["visualization_path"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )


if __name__ == "__main__":
    import tempfile

    print("Testing StorageManager...")

    # Create temporary storage
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = StorageManager(tmpdir)

        # Create mock data
        mock_image = SatelliteImage(
            data=np.random.rand(256, 256, 7).astype(np.float32),
            bounds=(-122.5, 37.5, -122.0, 38.0),
            crs="EPSG:4326",
            datetime=datetime.now(),
            scene_id="TEST_SCENE_001",
            cloud_cover=5.0,
            platform="Test-Platform",
        )

        mock_result = InferenceResult(
            segmentation=np.random.randint(0, 2, (256, 256), dtype=np.uint8),
            probabilities=np.random.rand(256, 256, 2).astype(np.float32),
            has_fire=True,
            fire_confidence=0.85,
            fire_fraction=0.12,
            severity_counts={"background": 225280, "fire": 31456},
            num_classes=2,
            image_shape=(256, 256),
        )

        mock_visualization = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Test save
        analysis_id = storage.save_analysis(
            satellite_image=mock_image,
            inference_result=mock_result,
            visualization=mock_visualization,
            metadata={"test": True},
        )
        print(f"  Saved analysis: {analysis_id}")

        # Test load
        loaded = storage.load_analysis(analysis_id)
        print(f"  Loaded analysis: {loaded['record'].id}")
        print(f"  Image shape: {loaded['image'].shape}")
        print(f"  Segmentation shape: {loaded['segmentation'].shape}")

        # Test history
        history = storage.get_history(limit=10)
        print(f"  History count: {len(history)}")

        # Test statistics
        stats = storage.get_statistics()
        print(f"  Statistics: {stats}")

        # Test delete
        deleted = storage.delete_analysis(analysis_id)
        print(f"  Deleted: {deleted}")

        # Verify deletion
        history_after = storage.get_history(limit=10)
        print(f"  History after delete: {len(history_after)}")

    print("\n✅ StorageManager test passed!")
