"""
Inference pipeline for the YOLO detection model.

Usage:
    from inference_yolo import YOLOInferencePipeline

    pipeline = YOLOInferencePipeline("checkpoints/best.pt")
    result = pipeline.predict_from_array(image_array)  # (H, W, 8)
    result = pipeline.predict_from_file("satellite_image.tif")

Model output:
    YOLO outputs bounding boxes with confidence scores per patch.
    Each detected box is rasterized into a pixel-level fire probability map:
    the box region is filled with the detection confidence score.
    Non-detected pixels have fire probability 0 (no-fire probability 1).
    The result is (N, 2, H, W) [no-fire, fire] probabilities, compatible
    with the parent class stitch_predictions method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from inference import FireInferencePipeline, InferenceResult


class YOLOInferencePipeline(FireInferencePipeline):
    """
    Inference pipeline for the YOLO detection model.

    Inherits preprocessing, patch extraction, stitching, and file loading
    from FireInferencePipeline. Overrides model loading and patch prediction.

    YOLO outputs bounding boxes per patch. These are rasterized into pixel-level
    probability maps compatible with the parent class stitch_predictions method.

    Args:
        model_path: Path to trained YOLO model checkpoint (.pt)
        device: Device to run inference on (auto, cuda, mps, cpu)
        patch_size: Size of patches for inference (default: 256)
        stride: Stride for patch extraction (default: 256, no overlap)
        batch_size: Batch size for inference
        conf_threshold: Minimum confidence threshold for detections (default: 0.25)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        patch_size: int = 256,
        stride: int = 256,
        batch_size: int = 8,
        conf_threshold: float = 0.25,
    ):
        self.conf_threshold = conf_threshold
        super().__init__(
            model_path=model_path,
            device=device,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
        )

    def _load_model(self, model_path: Union[str, Path]) -> tuple[object, dict]:
        """Load YOLO model from checkpoint."""
        from ultralytics import YOLO

        model = YOLO(str(model_path))

        # Extract config from model metadata if available.
        # YOLO's own "nc" (number of classes) never counts a background class,
        # since detection has no explicit background label — only "no detection".
        # We add +1 here so num_classes is comparable to the segmentation models'
        # convention, where class 0 is background.
        config: dict = {}
        try:
            yaml_cfg = model.model.yaml if hasattr(model.model, "yaml") else {}
            nc = yaml_cfg.get("nc", 1) if isinstance(yaml_cfg, dict) else 1
            ch = yaml_cfg.get("ch", 8) if isinstance(yaml_cfg, dict) else 8
            config["num_classes"] = nc + 1   # +1 to include background class 0
            config["in_channels"] = ch
        except Exception:
            # Metadata isn't always present (depends on Ultralytics version/checkpoint
            # format), so fall back to this project's known defaults instead of failing.
            config["num_classes"] = 2
            config["in_channels"] = 8

        config["dual_head"] = False
        return model, config

    def _ultralytics_device_str(self) -> str:
        """Convert torch.device to Ultralytics-compatible device string."""
        # Ultralytics expects a bare GPU index ("0"), not torch's "cuda:0" form,
        # so this translates whatever torch.device the parent class resolved
        # into the string format YOLO's predict()/train() calls actually accept.
        dev = str(self.device)
        if dev.startswith("cuda"):
            # "cuda:0" -> "0", "cuda" -> "0"
            parts = dev.split(":")
            return parts[1] if len(parts) > 1 else "0"
        return dev  # "cpu" or "mps"

    @torch.no_grad()
    def predict_patches(
        self, patches: np.ndarray
    ) -> tuple[np.ndarray, None, None]:
        """
        Run YOLO inference on patches and rasterize boxes to pixel maps.

        For each patch, YOLO returns bounding boxes with confidence scores.
        Each box region is filled with the detection confidence in the fire
        channel. Non-detected pixels retain fire probability 0.

        Args:
            patches: (N, H, W, C) numpy array

        Returns:
            (N, 2, H, W) array of [no-fire, fire] probabilities, None, None
        """
        n, h, w, _ = patches.shape
        # Pre-allocate a pixel-level probability map per patch so this detection
        # model's output shape matches the segmentation models' (N, 2, H, W)
        # convention, letting both share the same stitch_predictions logic.
        all_probs = np.zeros((n, 2, h, w), dtype=np.float32)
        all_probs[:, 0] = 1.0  # Initialize all pixels as no-fire

        device_str = self._ultralytics_device_str()

        # YOLO's predict() is called per-patch (not batched) because Ultralytics
        # results are returned as a list of per-image Results objects with a
        # variable number of boxes each — looping keeps the box-to-patch mapping
        # simple instead of having to track offsets into a batched output.
        for i in range(n):
            results = self.model.predict(
                source=patches[i],   # (H, W, C) float32
                conf=self.conf_threshold,
                device=device_str,
                verbose=False,
            )

            if not results:
                continue

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            for j in range(len(boxes)):
                conf = float(boxes.conf[j].cpu())
                x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().astype(int)
                # Clip box coordinates to the patch bounds: YOLO can predict
                # boxes that slightly overshoot the image edges.
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                # Take the maximum confidence if boxes overlap, rather than summing
                # or overwriting, so a pixel's fire probability reflects the most
                # confident detection covering it.
                all_probs[i, 1, y1:y2, x1:x2] = np.maximum(
                    all_probs[i, 1, y1:y2, x1:x2], conf
                )
                # Keep the two channels complementary (no-fire = 1 - fire) so
                # downstream code that expects a proper probability pair still works.
                all_probs[i, 0, y1:y2, x1:x2] = 1.0 - all_probs[i, 1, y1:y2, x1:x2]

        return all_probs, None, None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference_yolo.py <model_path> <image_path>")
        sys.exit(1)

    pipeline = YOLOInferencePipeline(sys.argv[1])
    result = pipeline.predict_from_file(sys.argv[2])

    print(f"Has fire: {result.has_fire}")
    print(f"Fire confidence: {result.fire_confidence:.2%}")
    print(f"Fire fraction: {result.fire_fraction:.2%}")
    print(f"Counts: {result.severity_counts}")
