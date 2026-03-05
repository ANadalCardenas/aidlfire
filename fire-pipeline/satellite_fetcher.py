"""
Satellite Data Fetcher for Fire Detection App.

Fetches Sentinel-2 L2A imagery from Microsoft Planetary Computer.

Usage:
    from satellite_fetcher import SatelliteFetcher

    fetcher = SatelliteFetcher()
    result = fetcher.fetch_region(
        bbox=(-122.5, 37.5, -122.0, 38.0),  # (west, south, east, north)
        date_range=("2024-01-01", "2024-01-31"),
        max_cloud_cover=20,
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class SatelliteImage:
    """Container for fetched satellite imagery."""

    # Image data
    data: np.ndarray  # (H, W, 7) selected bands, normalized
    bounds: tuple[float, float, float, float]  # (west, south, east, north)
    crs: str  # Coordinate reference system

    # Metadata
    datetime: datetime
    scene_id: str
    cloud_cover: float
    platform: str  # e.g., "Sentinel-2A"

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def center(self) -> tuple[float, float]:
        """Return center point (lon, lat)."""
        west, south, east, north = self.bounds
        return ((west + east) / 2, (south + north) / 2)


class SatelliteFetcher:
    """
    Fetches Sentinel-2 imagery from Microsoft Planetary Computer.

    Uses the STAC API to search for imagery and downloads selected bands.

    Band mapping (12 Sentinel-2 bands to 7 model bands):
        - B02 (Blue) -> index 0
        - B03 (Green) -> index 1
        - B04 (Red) -> index 2
        - B08 (NIR) -> index 3
        - B8A (NIR narrow) -> index 4
        - B11 (SWIR1) -> index 5
        - B12 (SWIR2) -> index 6
    """

    # Sentinel-2 bands to download (matching training data)
    BANDS = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]

    # Band resolutions in meters
    BAND_RESOLUTIONS = {
        "B02": 10,
        "B03": 10,
        "B04": 10,
        "B08": 10,
        "B8A": 20,
        "B11": 20,
        "B12": 20,
    }

    def __init__(
        self,
        cache_dir: Path | None = None,
        target_resolution: int = 10,
    ):
        """
        Initialize the satellite fetcher.

        Args:
            cache_dir: Directory to cache downloaded imagery
            target_resolution: Target resolution in meters (10 or 20)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.target_resolution = target_resolution

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._catalog = None
        self._initialized = False

    def _init_planetary_computer(self):
        """Initialize connection to Planetary Computer."""
        if self._initialized:
            return

        try:
            import planetary_computer
            import pystac_client
        except ImportError:
            raise ImportError(
                "planetary-computer and pystac-client are required. "
                "Install with: pip install planetary-computer pystac-client"
            )

        self._catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        self._initialized = True

    def search_scenes(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
        max_cloud_cover: float = 20.0,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search for available Sentinel-2 scenes.

        Args:
            bbox: Bounding box (west, south, east, north) in WGS84
            date_range: Date range as (start, end) in YYYY-MM-DD format
            max_cloud_cover: Maximum cloud cover percentage
            limit: Maximum number of results

        Returns:
            List of scene metadata dictionaries
        """
        self._init_planetary_computer()

        search = self._catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{date_range[0]}/{date_range[1]}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            limit=limit,
        )

        scenes = []
        for item in search.items():
            scenes.append({
                "id": item.id,
                "datetime": item.datetime,
                "cloud_cover": item.properties.get("eo:cloud_cover", 0),
                "platform": item.properties.get("platform", "unknown"),
                "bbox": item.bbox,
                "geometry": item.geometry,
                "item": item,  # Keep reference for downloading
            })

        # Sort by cloud cover (best first)
        scenes.sort(key=lambda x: x["cloud_cover"])

        return scenes

    def fetch_scene(
        self,
        scene: dict,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> SatelliteImage:
        """
        Fetch and process a single scene.

        Args:
            scene: Scene metadata from search_scenes()
            bbox: Optional bbox to clip the image (default: full scene)

        Returns:
            SatelliteImage with processed data
        """
        try:
            import rasterio
            from rasterio.windows import from_bounds
            from rasterio.warp import transform_bounds
            from rasterio.crs import CRS
        except ImportError:
            raise ImportError("rasterio is required for reading imagery")

        item = scene["item"]

        # Collect band data
        band_data = []
        actual_read_bbox = None  # Clipped bbox in source CRS (set when bbox used)
        source_crs = None

        for band_name in self.BANDS:
            asset = item.assets.get(band_name)
            if asset is None:
                raise ValueError(f"Band {band_name} not found in scene {scene['id']}")

            href = asset.href

            try:
                with rasterio.open(href) as src:
                    if bbox:
                        # Transform bbox from WGS84 to source CRS (Sentinel-2 is typically UTM)
                        if src.crs and str(src.crs) != "EPSG:4326":
                            read_bbox = transform_bounds(
                                CRS.from_epsg(4326), src.crs, *bbox
                            )
                        else:
                            read_bbox = bbox
                        # Clip to scene bounds so we never read outside the raster
                        # (otherwise rasterio fills out-of-bounds with 0 → black half-image)
                        src_left, src_bottom, src_right, src_top = src.bounds
                        r_w, r_s, r_e, r_n = read_bbox
                        clip_w = max(r_w, src_left)
                        clip_s = max(r_s, src_bottom)
                        clip_e = min(r_e, src_right)
                        clip_n = min(r_n, src_top)
                        if clip_w >= clip_e or clip_s >= clip_n:
                            raise ValueError(
                                f"Bbox does not intersect scene {scene['id']}. "
                                "Try a smaller region or different dates."
                            )
                        read_bbox = (clip_w, clip_s, clip_e, clip_n)
                        actual_read_bbox = read_bbox
                        source_crs = src.crs
                        # Read only the requested region (now within scene)
                        window = from_bounds(*read_bbox, src.transform)
                        data = src.read(1, window=window)
                        transform = src.window_transform(window)
                    else:
                        data = src.read(1)
                        transform = src.transform

                    crs = str(src.crs)

                    # Resample if needed (20m bands to 10m)
                    if self.BAND_RESOLUTIONS[band_name] > self.target_resolution:
                        from scipy.ndimage import zoom

                        scale = self.BAND_RESOLUTIONS[band_name] / self.target_resolution
                        data = zoom(data, scale, order=1)
            except rasterio.errors.RasterioIOError as e:
                raise RuntimeError(
                    f"Failed to read band {band_name} from {href}: {e}. "
                    "Check network connection or try a different scene."
                ) from e

            band_data.append(data)

        # Ensure all bands have the same shape
        min_h = min(b.shape[0] for b in band_data)
        min_w = min(b.shape[1] for b in band_data)
        band_data = [b[:min_h, :min_w] for b in band_data]

        # Stack bands: (H, W, 7)
        image_data = np.stack(band_data, axis=-1)

        if image_data.size == 0:
            raise ValueError(
                f"Fetched image is empty for bbox {bbox}. "
                "The region may not intersect the scene. Try a larger region or different dates."
            )

        # Normalize to 0-1 (Sentinel-2 L2A values are 0-10000)
        image_data = image_data.astype(np.float32) / 10000.0
        image_data = np.clip(image_data, 0.0, 1.0)

        # Determine actual bounds (clipped to scene if bbox only partially overlapped)
        if bbox:
            if actual_read_bbox is not None and source_crs and str(source_crs) != "EPSG:4326":
                bounds = tuple(
                    transform_bounds(
                        CRS.from_string(str(source_crs)), CRS.from_epsg(4326), *actual_read_bbox
                    )
                )
            else:
                bounds = actual_read_bbox if actual_read_bbox is not None else bbox
        else:
            bounds = tuple(scene["bbox"])

        return SatelliteImage(
            data=image_data,
            bounds=bounds,
            crs=crs,
            datetime=scene["datetime"],
            scene_id=scene["id"],
            cloud_cover=scene["cloud_cover"],
            platform=scene["platform"],
            metadata={
                "bands": self.BANDS,
                "resolution": self.target_resolution,
            },
        )

    def fetch_region(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
        max_cloud_cover: float = 20.0,
    ) -> SatelliteImage | None:
        """
        Fetch the best available image for a region.

        Searches for scenes and returns the one with lowest cloud cover.

        Args:
            bbox: Bounding box (west, south, east, north) in WGS84
            date_range: Date range as (start, end) in YYYY-MM-DD format
            max_cloud_cover: Maximum cloud cover percentage

        Returns:
            SatelliteImage or None if no suitable scene found
        """
        scenes = self.search_scenes(
            bbox=bbox,
            date_range=date_range,
            max_cloud_cover=max_cloud_cover,
            limit=5,
        )

        if not scenes:
            return None

        # Fetch the best scene (lowest cloud cover)
        return self.fetch_scene(scenes[0], bbox=bbox)


class MockSatelliteFetcher:
    """
    Mock satellite fetcher for testing without API access.

    Generates synthetic satellite imagery for development and testing.
    """

    BANDS = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir

    def search_scenes(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
        max_cloud_cover: float = 20.0,
        limit: int = 10,
    ) -> list[dict]:
        """Return mock scene metadata."""
        from datetime import timedelta

        start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
        end_date = datetime.strptime(date_range[1], "%Y-%m-%d")

        scenes = []
        current_date = start_date
        scene_num = 0

        while current_date <= end_date and len(scenes) < limit:
            scenes.append({
                "id": f"MOCK_S2A_{current_date.strftime('%Y%m%d')}_{scene_num:04d}",
                "datetime": current_date,
                "cloud_cover": np.random.uniform(0, max_cloud_cover),
                "platform": "Mock-Sentinel-2A",
                "bbox": bbox,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]],
                },
                "item": None,
            })
            current_date += timedelta(days=5)  # Sentinel-2 revisit time
            scene_num += 1

        scenes.sort(key=lambda x: x["cloud_cover"])
        return scenes

    def fetch_scene(
        self,
        scene: dict,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> SatelliteImage:
        """Generate synthetic satellite imagery."""
        # Generate reasonable sized image
        height, width = 512, 512

        # Create synthetic multi-spectral data
        # Simulate vegetation, water, and burned areas
        np.random.seed(hash(scene["id"]) % (2**32))

        # Base terrain with smooth variations
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        xx, yy = np.meshgrid(x, y)

        terrain = 0.3 + 0.2 * np.sin(xx) * np.cos(yy) + 0.1 * np.random.randn(height, width)
        terrain = np.clip(terrain, 0, 1)

        # Create 7-band synthetic data
        bands = []

        # B02 (Blue): Lower for vegetation
        bands.append(terrain * 0.3 + np.random.randn(height, width) * 0.02)

        # B03 (Green): Higher for vegetation
        bands.append(terrain * 0.5 + np.random.randn(height, width) * 0.02)

        # B04 (Red): Variable
        bands.append(terrain * 0.4 + np.random.randn(height, width) * 0.02)

        # B08 (NIR): High for vegetation, low for burned
        nir = terrain * 0.7 + np.random.randn(height, width) * 0.03

        # Add some "burned" patches
        for _ in range(np.random.randint(0, 3)):
            cx, cy = np.random.randint(50, width - 50), np.random.randint(50, height - 50)
            radius = np.random.randint(20, 60)
            yy_burn, xx_burn = np.ogrid[:height, :width]
            mask = ((xx_burn - cx) ** 2 + (yy_burn - cy) ** 2) < radius ** 2
            nir[mask] *= 0.3  # Low NIR for burned areas

        bands.append(nir)

        # B8A (NIR narrow): Similar to B08
        bands.append(nir * 0.95 + np.random.randn(height, width) * 0.02)

        # B11 (SWIR1): High for burned areas
        swir1 = terrain * 0.4 + np.random.randn(height, width) * 0.02
        swir1[nir < 0.3] = 0.6  # High SWIR where NIR is low (burned)
        bands.append(swir1)

        # B12 (SWIR2): Similar to SWIR1
        bands.append(swir1 * 0.9 + np.random.randn(height, width) * 0.02)

        # Stack and normalize
        image_data = np.stack(bands, axis=-1).astype(np.float32)
        image_data = np.clip(image_data, 0.0, 1.0)

        bounds = bbox if bbox else scene["bbox"]

        return SatelliteImage(
            data=image_data,
            bounds=bounds,
            crs="EPSG:4326",
            datetime=scene["datetime"],
            scene_id=scene["id"],
            cloud_cover=scene["cloud_cover"],
            platform=scene["platform"],
            metadata={
                "bands": self.BANDS,
                "resolution": 10,
                "mock": True,
            },
        )

    def fetch_region(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
        max_cloud_cover: float = 20.0,
    ) -> SatelliteImage | None:
        """Fetch mock imagery for a region."""
        scenes = self.search_scenes(
            bbox=bbox,
            date_range=date_range,
            max_cloud_cover=max_cloud_cover,
            limit=1,
        )

        if not scenes:
            return None

        return self.fetch_scene(scenes[0], bbox=bbox)


def get_fetcher(use_mock: bool = False, cache_dir: Path | None = None):
    """
    Get appropriate satellite fetcher.

    Args:
        use_mock: If True, use mock fetcher (no API needed)
        cache_dir: Optional cache directory

    Returns:
        SatelliteFetcher or MockSatelliteFetcher
    """
    if use_mock:
        return MockSatelliteFetcher(cache_dir=cache_dir)
    return SatelliteFetcher(cache_dir=cache_dir)


if __name__ == "__main__":
    # Test with mock fetcher
    print("Testing MockSatelliteFetcher...")

    fetcher = MockSatelliteFetcher()

    # Search for scenes
    bbox = (-122.5, 37.5, -122.0, 38.0)  # San Francisco Bay Area
    date_range = ("2024-01-01", "2024-01-31")

    scenes = fetcher.search_scenes(bbox, date_range)
    print(f"\nFound {len(scenes)} scenes:")
    for scene in scenes[:3]:
        print(f"  {scene['id']}: {scene['cloud_cover']:.1f}% cloud cover")

    # Fetch best scene
    image = fetcher.fetch_region(bbox, date_range)
    print(f"\nFetched image:")
    print(f"  Shape: {image.data.shape}")
    print(f"  Bounds: {image.bounds}")
    print(f"  Date: {image.datetime}")
    print(f"  Cloud cover: {image.cloud_cover:.1f}%")

    print("\n✅ Mock fetcher test passed!")
