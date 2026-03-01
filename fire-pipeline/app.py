"""
Fire Detection Streamlit App.

Interactive app for satellite-based wildfire detection and severity mapping.

Features:
- Interactive map for region selection
- Satellite image fetching from Planetary Computer
- Real-time fire detection inference
- History of analyses with filtering
- Visualization of results

Run with:
    streamlit run app.py
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Min width for image viewer so it doesn't render as narrow strips
st.markdown(
    """<style>div[data-testid="stPlotlyChart"]{min-width:1000px !important}</style>""",
    unsafe_allow_html=True,
)

from constants import get_class_names
from satellite_fetcher import get_fetcher, SatelliteImage
from storage import StorageManager, AnalysisRecord
from inference import (
    FireInferencePipeline,
    InferenceResult,
    create_visualization,
    create_visualization_from_segmentation,
    create_fire_mask_visualization,
)


# Configuration (can be set via environment variables)
STORAGE_DIR = Path(os.environ.get("FIRE_STORAGE_DIR", "./cache"))
# When MODELS_DIR is set, app shows dropdown of model subdirs; path = MODELS_DIR/<model>/checkpoints/<MODEL_NAME>.pt
_models_dir_str = os.environ.get("FIRE_MODELS_DIR", "")
MODELS_DIR = Path(_models_dir_str) if _models_dir_str else None
MODEL_NAME = os.environ.get("FIRE_MODEL_NAME", "best_model")  # Checkpoint filename without .pt
# Fallback when MODELS_DIR is not used
MODEL_PATH = Path(os.environ.get("FIRE_MODEL_PATH", "./checkpoints/best_model.pt"))
USE_MOCK_FETCHER = os.environ.get("FIRE_USE_MOCK", "true").lower() in ("true", "1", "yes")


@st.cache_resource
def get_storage():
    """Get or create storage manager."""
    return StorageManager(STORAGE_DIR)


@st.cache_resource
def get_model(model_path: Path):
    """Load the fire detection model from the given path."""
    if model_path.exists():
        return FireInferencePipeline(model_path)
    return None


def _get_available_models() -> list[str]:
    """List model names (subdirs) in MODELS_DIR that have checkpoints/<MODEL_NAME>.pt."""
    if not MODELS_DIR or not MODELS_DIR.exists():
        return []
    models = []
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir():
            ckpt = d / "checkpoints" / f"{MODEL_NAME}.pt"
            if ckpt.exists():
                models.append(d.name)
    return models


def _resolve_model_path(selected_model: str | None) -> Path:
    """Resolve the model checkpoint path from MODELS_DIR + selected model, or fallback to MODEL_PATH."""
    if MODELS_DIR and MODELS_DIR.exists() and selected_model:
        return MODELS_DIR / selected_model / "checkpoints" / f"{MODEL_NAME}.pt"
    return MODEL_PATH


def run_inference_mock(image: SatelliteImage) -> InferenceResult:
    """
    Run mock inference when model is not available.

    Generates synthetic results for testing the UI.
    """
    h, w = image.data.shape[:2]

    # Create mock segmentation with some "fire" areas
    np.random.seed(hash(image.scene_id) % (2**32))
    segmentation = np.zeros((h, w), dtype=np.uint8)

    # Add random fire patches
    num_fires = np.random.randint(0, 4)
    for _ in range(num_fires):
        cx = np.random.randint(50, w - 50)
        cy = np.random.randint(50, h - 50)
        radius = np.random.randint(15, 50)

        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2
        segmentation[mask] = np.random.randint(1, 5)  # Severity 1-4

    # Create mock probabilities
    probabilities = np.zeros((h, w, 5), dtype=np.float32)
    for i in range(5):
        probabilities[:, :, i] = (segmentation == i).astype(np.float32)
    probabilities += np.random.rand(h, w, 5).astype(np.float32) * 0.1
    probabilities /= probabilities.sum(axis=2, keepdims=True)

    has_fire = (segmentation > 0).any()
    fire_probs = probabilities[:, :, 1:].sum(axis=2)

    return InferenceResult(
        segmentation=segmentation,
        probabilities=probabilities,
        has_fire=has_fire,
        fire_confidence=float(fire_probs.max()) if has_fire else 0.0,
        fire_fraction=float((segmentation > 0).mean()),
        severity_counts={
            "no_damage": int((segmentation == 0).sum()),
            "negligible": int((segmentation == 1).sum()),
            "moderate": int((segmentation == 2).sum()),
            "high": int((segmentation == 3).sum()),
            "destroyed": int((segmentation == 4).sum()),
        },
        num_classes=5,
        image_shape=(h, w),
    )


def create_rgb_preview(image: SatelliteImage) -> np.ndarray:
    """Create RGB preview from satellite image."""
    # Use bands 2, 1, 0 (Red, Green, Blue)
    rgb = image.data[:, :, [2, 1, 0]]
    rgb = np.clip(rgb * 3.0, 0, 1)  # Boost brightness
    return (rgb * 255).astype(np.uint8)


def resize_for_display(img: np.ndarray, max_size: int = 600) -> np.ndarray:
    """Resize image for display, capping the longest side to max_size."""
    from PIL import Image

    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    pil = Image.fromarray(img)
    return np.array(pil.resize((new_w, new_h), Image.Resampling.LANCZOS))


def render_synced_images(
    img1: np.ndarray,
    img2: np.ndarray,
    label1: str = "Original",
    label2: str = "Fire detection",
    max_size: int = 1000,
) -> None:
    """
    Render two images side-by-side with Plotly, synced zoom/pan.
    Uses px.imshow facet_col for consistent equal sizing of both panels.
    """
    import plotly.express as px

    # Resize both to same size for display (whole image visible)
    img1 = resize_for_display(img1, max_size=max_size)
    img2 = resize_for_display(img2, max_size=max_size)

    # Ensure same dimensions for sync (crop to min)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]

    # Stack as (2, H, W, 3) - facet_col ensures equal sizing for both panels
    stacked = np.stack([img1, img2], axis=0)
    fig = px.imshow(
        stacked,
        facet_col=0,
        binary_string=True,
        facet_col_wrap=2,
        labels={"facet_col": ""},
    )
    # Set facet titles
    for i, label in enumerate([label1, label2]):
        if i < len(fig.layout.annotations):
            fig.layout.annotations[i].text = label

    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, scaleanchor="x")
    display_height = max(450, min(h, 600))
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        height=display_height,
        width=max(1000, w),  # Min 1000px so images don't render as narrow strips
        autosize=True,
        dragmode="pan",
        showlegend=False,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "doubleClick": "reset",
        },
    )


def main():
    """Main app function."""
    # Initialize session state for navigation
    if "page" not in st.session_state:
        st.session_state.page = "New Analysis"

    # Sidebar
    with st.sidebar:
        st.title("🔥 Fire Detection")
        st.markdown("---")

        # Navigation menu with styled buttons
        menu_items = [
            ("🛰️ New Analysis", "New Analysis"),
            ("📜 History", "History"),
            ("📊 Statistics", "Statistics"),
        ]

        for label, page_name in menu_items:
            if st.button(
                label,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if st.session_state.page == page_name else "secondary",
            ):
                st.session_state.page = page_name
                st.rerun()

        st.markdown("---")

        # Model selection
        available_models = _get_available_models()
        if available_models:
            selected = st.selectbox(
                "Model",
                options=available_models,
                key="model_select",
            )
            effective_path = _resolve_model_path(selected)
        else:
            effective_path = _resolve_model_path(None)

        st.session_state.effective_model_path = effective_path
        model = get_model(effective_path)
        if model:
            st.success("Model loaded")
        else:
            st.warning("Model not found - using mock inference")

        # Filters for New Analysis (in sidebar so main area is full width)
        if st.session_state.page == "New Analysis":
            st.subheader("Filters")
            _render_analysis_filters()
            if USE_MOCK_FETCHER:
                with st.expander("Using mock data"):
                    st.caption("Set FIRE_USE_MOCK=false for real Sentinel-2 imagery.")
            st.markdown("---")
            storage = get_storage()
            stats = storage.get_statistics()
            st.metric("Total Analyses", stats["total_analyses"])
            st.metric("Fire Detections", stats["fire_detections"])
        else:
            # Mock fetcher warning (only when not on New Analysis)
            if USE_MOCK_FETCHER:
                st.warning("Using mock satellite data")
                with st.expander("How to use real data"):
                    st.markdown("""
                    To fetch real Sentinel-2 imagery:

                    **Option 1: Environment variable**
                    ```bash
                    export FIRE_USE_MOCK=false
                    streamlit run app.py
                    ```

                    **Option 2: Edit app.py**
                    Set `USE_MOCK_FETCHER = False`

                    Requires `planetary-computer` package.
                    """)

            # Storage info
            storage = get_storage()
            stats = storage.get_statistics()
            st.metric("Total Analyses", stats["total_analyses"])
            st.metric("Fire Detections", stats["fire_detections"])

    # Main content
    page = st.session_state.page
    if page == "New Analysis":
        render_new_analysis()
    elif page == "History":
        render_history()
    else:
        render_statistics()


def _render_analysis_filters() -> None:
    """Render analysis filters in sidebar. Stores bbox in session state for map."""
    input_method = st.radio(
        "Input method",
        ["Draw on map", "Preset Locations", "Coordinates"],
        horizontal=False,
    )

    if input_method == "Draw on map":
        # Use bbox from map drawing if available, else default
        bbox = st.session_state.get("drawn_bbox")
        if bbox is None:
            bbox = (2.0, 41.5, 2.5, 42.0)  # Default: Barcelona area
        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2
        st.caption("Draw a rectangle on the map below to select the analysis area.")
        if st.button("Clear selection", key="clear_draw"):
            st.session_state.pop("drawn_bbox", None)
            st.rerun()
    elif input_method == "Coordinates":
        center_lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-122.0,
            step=0.1,
        )
        center_lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=37.5,
            step=0.1,
        )
        region_size = st.slider(
            "Region Size (°)",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )
        half_size = region_size / 2
        bbox = (
            center_lon - half_size,
            center_lat - half_size,
            center_lon + half_size,
            center_lat + half_size,
        )
    elif input_method == "Preset Locations":
        # Smaller areas (~0.2–0.4°) for better resolution. Bbox: (west, south, east, north)
        presets = {
            # Catalonia (Spain) - ~20–40 km regions
            "Catalonia › Barcelona": (1.9, 41.3, 2.2, 41.5),
            "Catalonia › Girona": (2.7, 41.9, 3.0, 42.2),
            "Catalonia › Tarragona": (1.1, 41.0, 1.4, 41.2),
            "Catalonia › Lleida": (0.6, 41.6, 0.9, 41.8),
            "Catalonia › Costa Brava": (2.9, 41.7, 3.2, 42.0),
            "Catalonia › Pyrenees": (1.0, 42.2, 1.4, 42.5),
            # Other regions
            "California › Bay Area": (-122.5, 37.5, -122.0, 38.0),
            "Portugal › Lisbon": (-9.2, 38.6, -8.9, 38.8),
            "Greece › Athens": (23.6, 37.9, 23.8, 38.1),
            "Australia › Sydney": (150.9, -33.9, 151.3, -33.7),
        }
        preset_keys = list(presets.keys())
        location = st.selectbox("Location", preset_keys, index=0)
        bbox = presets[location]
        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2

    st.caption(f"Bbox: ({bbox[0]:.3f}, {bbox[1]:.3f}) → ({bbox[2]:.3f}, {bbox[3]:.3f})")
    st.session_state.analysis_bbox = bbox
    st.session_state.analysis_center = (center_lat, center_lon)

    # Map in sidebar with town markers
    try:
        import folium
        from streamlit_folium import st_folium

        # Major towns (name, lat, lon) - filtered by bbox
        TOWNS = [
            # Catalonia
            ("Barcelona", 41.3851, 2.1734),
            ("Girona", 41.9794, 2.8214),
            ("Tarragona", 41.1189, 1.2445),
            ("Lleida", 41.6176, 0.6200),
            ("Figueres", 42.2675, 2.9611),
            ("Reus", 41.1569, 1.1086),
            ("Sabadell", 41.5433, 2.1094),
            ("Terrassa", 41.5636, 2.0109),
            ("Manresa", 41.7250, 1.8261),
            ("Mataró", 41.5421, 2.4445),
            ("Vic", 41.9303, 2.2544),
            ("Olot", 42.1822, 2.4891),
            ("Sitges", 41.2370, 1.8113),
            ("Tortosa", 40.8125, 0.5211),
            ("Igualada", 41.5814, 1.6172),
            # California
            ("San Francisco", 37.7749, -122.4194),
            ("Oakland", 37.8044, -122.2712),
            ("San Jose", 37.3382, -121.8863),
            ("Sacramento", 38.5816, -121.4944),
            # Portugal
            ("Lisbon", 38.7223, -9.1393),
            ("Porto", 41.1579, -8.6291),
            ("Faro", 37.0194, -7.9304),
            # Greece
            ("Athens", 37.9838, 23.7275),
            ("Thessaloniki", 40.6401, 22.9444),
            ("Patras", 38.2466, 21.7346),
            # Australia NSW
            ("Sydney", -33.8688, 151.2093),
            ("Newcastle", -32.9283, 151.7817),
            ("Wollongong", -34.4278, 150.8931),
        ]
        west, south, east, north = bbox
        towns_in_view = [
            (n, lat, lon) for n, lat, lon in TOWNS
            if south <= lat <= north and west <= lon <= east
        ]

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles="OpenStreetMap",
        )
        folium.Rectangle(
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            color="red",
            fill=True,
            fillOpacity=0.2,
        ).add_to(m)
        for name, lat, lon in towns_in_view:
            folium.CircleMarker(
                [lat, lon],
                radius=6,
                popup=name,
                tooltip=name,
                color="blue",
                fill=True,
                fillColor="blue",
                fillOpacity=0.6,
                weight=1,
            ).add_to(m)

        # Add Draw plugin when "Draw on map" is selected
        draw_enabled = input_method == "Draw on map"
        if draw_enabled:
            from folium.plugins import Draw

            Draw(
                export=False,
                show_geometry_on_click=False,
                draw_options={
                    "rectangle": {"shapeOptions": {"color": "#3388ff"}},
                    "polygon": False,
                    "polyline": False,
                    "circle": False,
                    "circlemarker": False,
                    "marker": False,
                },
                edit_options={"edit": True, "remove": True},
            ).add_to(m)

        map_data = st_folium(
            m,
            height=220,
            key="analysis_map",
            returned_objects=["last_active_drawing", "all_drawings"],
        )

        # Parse drawn rectangle and update bbox
        if draw_enabled and map_data:
            drawn = map_data.get("last_active_drawing") or (
                (map_data.get("all_drawings") or [])[-1] if map_data.get("all_drawings") else None
            )
            if drawn and isinstance(drawn, dict):
                geom = drawn.get("geometry") or drawn
                coords = geom.get("coordinates") if isinstance(geom, dict) else None
                if coords:
                    # GeoJSON: [[[lon,lat],...]] for polygon/rectangle
                    ring = coords[0] if isinstance(coords[0][0], (list, tuple)) else coords
                    lons = [p[0] for p in ring]
                    lats = [p[1] for p in ring]
                    if lons and lats:
                        west, east = min(lons), max(lons)
                        south, north = min(lats), max(lats)
                        if east - west > 0.01 and north - south > 0.01:  # Min ~1km
                            new_bbox = (west, south, east, north)
                            prev = st.session_state.get("drawn_bbox")
                            if prev is None or abs(prev[0] - west) > 1e-5 or abs(prev[3] - north) > 1e-5:
                                st.session_state.drawn_bbox = new_bbox
                                st.session_state.analysis_bbox = new_bbox
                                st.rerun()
    except ImportError:
        pass

    st.markdown("---")
    st.caption("Date range")
    end_date = st.date_input(
        "End Date",
        value=datetime.now().date(),
        max_value=datetime.now().date(),
    )
    days_back = st.slider("Days to search", min_value=1, max_value=60, value=30)
    start_date = end_date - timedelta(days=days_back)
    max_cloud = st.slider("Max Cloud (%)", min_value=5, max_value=50, value=20)

    st.markdown("---")
    if st.button("🛰️ Fetch & Analyze", type="primary", use_container_width=True):
        run_analysis(
            bbox=bbox,
            date_range=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
            max_cloud_cover=max_cloud,
        )
        st.rerun()


def render_new_analysis():
    """Render the new analysis page - full-width main area for results."""
    st.subheader("Results")

    # Results - full width main area
    if "analysis_result" in st.session_state and st.session_state.analysis_result:
        render_analysis_results(st.session_state.analysis_result)


def run_analysis(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
    max_cloud_cover: float,
):
    """Fetch satellite data, run fire detection, store results in session state."""
    storage = get_storage()
    effective_path = st.session_state.get("effective_model_path", MODEL_PATH)
    model = get_model(effective_path)

    with st.spinner("Fetching satellite imagery..."):
        fetcher = get_fetcher(use_mock=USE_MOCK_FETCHER)

        try:
            image = fetcher.fetch_region(
                bbox=bbox,
                date_range=date_range,
                max_cloud_cover=max_cloud_cover,
            )
        except Exception as e:
            st.error(f"Failed to fetch imagery: {e}")
            return

        if image is None:
            st.warning("No suitable imagery found for the selected region and date range.")
            return

    # Run inference
    with st.spinner("Running fire detection..."):
        if model:
            result = model.predict_from_array(image.data, select_bands=False)
        else:
            result = run_inference_mock(image)

    # Create visualization
    if model:
        visualization = create_visualization(image.data, result)
    else:
        rgb = create_rgb_preview(image)
        visualization = rgb.copy()
        fire_mask = result.segmentation > 0
        visualization[fire_mask] = [255, 0, 0]  # Red overlay for fire

    # Save to storage
    with st.spinner("Saving analysis..."):
        analysis_id = storage.save_analysis(
            satellite_image=image,
            inference_result=result,
            visualization=visualization,
            metadata={"bbox": list(bbox), "date_range": list(date_range)},
        )

    # Store in session state for rendering in main area
    st.session_state.analysis_result = {
        "image": image,
        "result": result,
        "visualization": visualization,
        "analysis_id": analysis_id,
    }


def _loaded_analysis_to_display_data(analysis: dict) -> dict | None:
    """
    Convert loaded analysis (from storage.load_analysis) to the format
    expected by render_analysis_results.
    """
    record = analysis.get("record")
    image_arr = analysis.get("image")
    segmentation = analysis.get("segmentation")
    probabilities = analysis.get("probabilities")
    visualization = analysis.get("visualization")
    binary_segmentation = analysis.get("binary_segmentation")
    severity_segmentation = analysis.get("severity_segmentation")
    binary_probabilities = analysis.get("binary_probabilities")
    severity_probabilities = analysis.get("severity_probabilities")

    if not all([record, image_arr is not None, segmentation is not None]):
        return None
    if visualization is None and not (record.metadata and record.metadata.get("dual_head")):
        return None

    # Build SatelliteImage for display (crs not stored, use placeholder)
    image = SatelliteImage(
        data=image_arr,
        bounds=record.bbox,
        crs="EPSG:4326",
        datetime=record.satellite_datetime,
        scene_id=record.scene_id,
        cloud_cover=record.cloud_cover,
        platform=record.platform or "Unknown",
    )

    # Compute severity_counts from segmentation
    num_classes = record.num_classes
    dual_head = bool(
        record.metadata and record.metadata.get("dual_head")
        and binary_segmentation is not None
        and severity_segmentation is not None
    )
    try:
        class_names = get_class_names(num_classes)
    except ValueError:
        class_names = tuple(f"class_{i}" for i in range(num_classes))
    severity_counts = {name: int((segmentation == i).sum()) for i, name in enumerate(class_names)}

    # Build InferenceResult (probabilities may be missing in older saves)
    if probabilities is None or probabilities.shape[2] != (2 if dual_head else num_classes):
        n = 2 if dual_head else num_classes
        probs = np.zeros((*segmentation.shape, n), dtype=np.float32)
        for c in range(n):
            probs[:, :, c] = (segmentation == c).astype(np.float32)
        probabilities = probs

    result = InferenceResult(
        segmentation=segmentation,
        probabilities=probabilities,
        has_fire=record.has_fire,
        fire_confidence=record.fire_confidence,
        fire_fraction=record.fire_fraction,
        severity_counts=severity_counts,
        num_classes=2 if dual_head else num_classes,
        image_shape=image_arr.shape[:2],
        dual_head=dual_head,
        binary_segmentation=binary_segmentation if dual_head else None,
        severity_segmentation=severity_segmentation if dual_head else None,
        binary_probabilities=binary_probabilities if dual_head else None,
        severity_probabilities=severity_probabilities if dual_head else None,
    )

    # If no saved visualization but dual_head with maps, we'll render from result in render_analysis_results
    if visualization is None and dual_head:
        visualization = create_visualization_from_segmentation(
            image_arr, result.binary_segmentation if result.binary_segmentation is not None else segmentation, 2, alpha=0.5
        )

    return {
        "image": image,
        "result": result,
        "visualization": visualization,
        "analysis_id": record.id,
    }


def render_analysis_results(data: dict, from_history: bool = False) -> None:
    """Render analysis results in the main content area.

    Args:
        data: Dict with image, result, visualization, analysis_id
        from_history: If True, show Close/Delete buttons instead of "Analysis saved!"
    """
    image = data["image"]
    result = data["result"]
    visualization = data["visualization"]
    analysis_id = data["analysis_id"]

    st.markdown("---")
    st.success(f"Found image: {image.scene_id}")

    # Image info
    h, w = image.data.shape[:2]
    # Sentinel-2 ~10m GSD: approximate ground coverage in km
    gsd_m = 10
    ground_km_x = w * gsd_m / 1000
    ground_km_y = h * gsd_m / 1000
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Image Date", image.datetime.strftime("%Y-%m-%d") if image.datetime else "Unknown")
    with col2:
        st.metric("Cloud Cover", f"{image.cloud_cover:.1f}%")
    with col3:
        st.metric("Resolution", f"{h}×{w} px")
    with col4:
        st.metric("Ground coverage", f"~{ground_km_x:.1f}×{ground_km_y:.1f} km")

    with st.expander("ℹ️ About this analysis"):
        st.markdown("""
        **What you're seeing:** The image is the exact area analyzed—cropped from the Sentinel-2 tile that intersects your selected region. The red overlay is the model's pixel-level prediction (no dilation).

        **Model training:** This model was trained on **burn scars** (post-fire damage) from the CEMS dataset, not active flames. It detects areas that show spectral signatures of past burning (low NIR, high SWIR). Some terrain (bare soil, agriculture, shadows) can trigger false positives.

        **Resolution:** Sentinel-2 L2A has ~10 m ground sampling. At 1521×1666 px, that's ~15×17 km. The sidebar bbox may be larger—you see the tile intersection that was actually fetched.
        """)

    st.markdown("---")
    st.subheader("Results")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fire Detected", "🔥 Yes" if result.has_fire else "✅ No")
    with col2:
        st.metric("Confidence", f"{result.fire_confidence:.1%}")
    with col3:
        st.metric("Fire Coverage", f"{result.fire_fraction:.2%}")
    with col4:
        st.metric("Classes", result.num_classes)

    # Synced zoom/pan viewer - full width in main area
    rgb_preview = create_rgb_preview(image)
    st.caption("Scroll to zoom, drag to pan. Both images stay in sync.")

    # Dual-head: optional layers (binary fire + severity)
    if getattr(result, "dual_head", False) and result.binary_segmentation is not None and result.severity_segmentation is not None:
        st.caption("**Dual-head model:** toggle layers below to compare binary fire and severity maps.")
        show_binary = st.checkbox("Show binary fire map", value=True, key="show_binary_layer")
        show_severity = st.checkbox("Show severity map", value=True, key="show_severity_layer")
        if show_binary and show_severity:
            vis_binary = create_visualization_from_segmentation(
                image.data, result.binary_segmentation, 2, alpha=0.5
            )
            vis_severity = create_visualization_from_segmentation(
                image.data, result.severity_segmentation, 5, alpha=0.5
            )
            render_synced_images(rgb_preview, vis_binary, "Original", "Binary fire")
            st.markdown("---")
            render_synced_images(rgb_preview, vis_severity, "Original", "Severity")
        elif show_binary:
            vis_binary = create_visualization_from_segmentation(
                image.data, result.binary_segmentation, 2, alpha=0.5
            )
            render_synced_images(rgb_preview, vis_binary, "Original", "Binary fire")
        elif show_severity:
            vis_severity = create_visualization_from_segmentation(
                image.data, result.severity_segmentation, 5, alpha=0.5
            )
            render_synced_images(rgb_preview, vis_severity, "Original", "Severity")
        else:
            st.info("Enable at least one layer (binary fire or severity) to see the map.")
    else:
        render_synced_images(rgb_preview, visualization, "Original", "Fire detection")

    # When fire detected, show high-contrast fire mask so sparse detections are visible
    if result.has_fire:
        with st.expander("📍 Where are the fires?", expanded=True):
            dilate = st.slider("Dilation (px)", 0, 8, 3, help="Expand fire pixels for visibility. 0 = raw model output.")
            fire_mask_viz = create_fire_mask_visualization(
                result.segmentation,
                num_classes=result.num_classes,
                dilate_pixels=dilate,
            )
            st.caption("Fire pixels (red) on dark background. Set dilation to 0 to see raw model output.")
            st.image(resize_for_display(fire_mask_viz, max_size=1000), width="stretch")

    # Severity breakdown
    if result.severity_counts:
        st.markdown("**Severity Distribution**")
        cols = st.columns(len(result.severity_counts))
        total = sum(result.severity_counts.values())
        for i, (name, count) in enumerate(result.severity_counts.items()):
            with cols[i]:
                pct = count / total * 100 if total > 0 else 0
                # Use more decimals for small values so fire (often <0.1%) is visible
                fmt = f"{pct:.2f}%" if 0 < pct < 0.1 else f"{pct:.1f}%"
                st.metric(name.replace("_", " ").title(), fmt)

    if from_history:
        st.markdown("---")
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("Delete Analysis", type="secondary"):
                get_storage().delete_analysis(analysis_id)
                if "view_analysis" in st.session_state:
                    del st.session_state["view_analysis"]
                st.rerun()
        with col2:
            if st.button("Close"):
                if "view_analysis" in st.session_state:
                    del st.session_state["view_analysis"]
                st.rerun()
    else:
        st.success(f"Analysis saved! ID: {analysis_id}")


def render_history():
    """Render the history page."""
    st.header("Analysis History")

    storage = get_storage()

    # Filters
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            fire_only = st.checkbox("Fire detections only")

        with col2:
            date_filter = st.date_input(
                "Date range",
                value=(datetime.now().date() - timedelta(days=30), datetime.now().date()),
            )

        with col3:
            limit = st.selectbox("Show", [10, 25, 50, 100], index=1)

    # Parse date filter
    if isinstance(date_filter, tuple) and len(date_filter) == 2:
        start_date = datetime.combine(date_filter[0], datetime.min.time())
        end_date = datetime.combine(date_filter[1], datetime.max.time())
    else:
        start_date = None
        end_date = None

    # Load history
    history = storage.get_history(
        limit=limit,
        fire_only=fire_only,
        start_date=start_date,
        end_date=end_date,
    )

    if not history:
        st.info("No analyses found. Run a new analysis to see history.")
        return

    # Display as cards
    for record in history:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

            with col1:
                st.markdown(f"**{record.scene_id}**")
                st.caption(record.created_at.strftime("%Y-%m-%d %H:%M") if record.created_at else "Unknown")

            with col2:
                if record.has_fire:
                    st.markdown("🔥 **Fire**")
                else:
                    st.markdown("✅ Clear")

            with col3:
                st.markdown(f"Conf: {record.fire_confidence:.0%}")

            with col4:
                st.markdown(f"☁️ {record.cloud_cover:.0f}%")

            with col5:
                if st.button("View", key=f"view_{record.id}"):
                    st.session_state["view_analysis"] = record.id

            st.markdown("---")

    # View selected analysis - reuse same view as New Analysis
    if "view_analysis" in st.session_state:
        analysis_id = st.session_state["view_analysis"]
        analysis = storage.load_analysis(analysis_id)

        if analysis:
            display_data = _loaded_analysis_to_display_data(analysis)
            if display_data:
                render_analysis_results(display_data, from_history=True)
            else:
                st.error("Could not load analysis data. Some files may be missing.")
                if st.button("Close"):
                    del st.session_state["view_analysis"]
                    st.rerun()


def render_statistics():
    """Render the statistics page."""
    st.header("Statistics")

    storage = get_storage()
    stats = storage.get_statistics()

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Analyses", stats["total_analyses"])

    with col2:
        st.metric("Fire Detections", stats["fire_detections"])

    with col3:
        detection_rate = (
            stats["fire_detections"] / stats["total_analyses"] * 100
            if stats["total_analyses"] > 0
            else 0
        )
        st.metric("Detection Rate", f"{detection_rate:.1f}%")

    with col4:
        st.metric("Avg Cloud Cover", f"{stats['avg_cloud_cover']:.1f}%" if stats['avg_cloud_cover'] else "N/A")

    # Time range
    if stats["first_analysis"] and stats["last_analysis"]:
        st.markdown("---")
        st.markdown(f"**Analysis Period:** {stats['first_analysis'][:10]} to {stats['last_analysis'][:10]}")

    # Recent detections
    st.markdown("---")
    st.subheader("Recent Fire Detections")

    recent_fires = storage.get_history(limit=10, fire_only=True)

    if recent_fires:
        for record in recent_fires:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**{record.scene_id}**")
                st.caption(f"({record.center_lon:.2f}, {record.center_lat:.2f})")

            with col2:
                st.markdown(f"{record.fire_confidence:.0%}")

            with col3:
                if record.created_at:
                    st.caption(record.created_at.strftime("%m/%d"))
    else:
        st.info("No fire detections recorded yet.")

    # Maintenance
    st.markdown("---")
    st.subheader("Maintenance")

    col1, col2 = st.columns(2)

    with col1:
        days = st.number_input("Delete analyses older than (days)", min_value=7, value=90)

    with col2:
        if st.button("🗑️ Clean Old Data"):
            deleted = storage.clear_old_analyses(days=days)
            st.success(f"Deleted {deleted} old analyses")
            st.rerun()


if __name__ == "__main__":
    main()
