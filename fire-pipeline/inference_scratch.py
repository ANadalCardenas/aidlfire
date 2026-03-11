"""
Inference pipeline for the CNN scratch binary fire classifier (ScratchFireModel).

Usage:
    from inference_scratch import ScratchInferencePipeline

    pipeline = ScratchInferencePipeline("checkpoints/best_model.pt")
    result = pipeline.predict_from_array(image_array)  # (H, W, 8)
    result = pipeline.predict_from_file("satellite_image.tif")

Model output:
    ScratchFireModel outputs a single logit per patch (binary classification).
    For inference, patches are extracted and each patch receives a fire probability
    score. The pixel-level fire map assigns each pixel the probability of its patch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from scratch_model import ScratchFireModel
from inference import FireInferencePipeline, InferenceResult


class ScratchInferencePipeline(FireInferencePipeline):
    """
    Inference pipeline for ScratchFireModel (CNN binary fire classifier).

    Inherits preprocessing, patch extraction, stitching, and file loading
    from FireInferencePipeline. Overrides model loading and patch prediction.

    Args:
        model_path: Path to trained model checkpoint (.pt)
        device: Device to run inference on (auto, cuda, mps, cpu)
        patch_size: Size of patches for inference (default: 256)
        stride: Stride for patch extraction (default: 256, no overlap)
        batch_size: Batch size for inference
    """

    def _load_model(self, model_path: Union[str, Path]) -> tuple[nn.Module, dict]:
        """Load ScratchFireModel from checkpoint."""
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        in_channels = config.get("in_channels", 8)
        dropout = config.get("dropout", 0.3)

        model = ScratchFireModel(in_channels=in_channels, dropout=dropout)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        config["num_classes"] = 2
        config["dual_head"] = False
        return model, config

    @torch.no_grad()
    def predict_patches(
        self, patches: np.ndarray
    ) -> tuple[np.ndarray, None, None]:
        """
        Run inference on patches using ScratchFireModel.

        The model outputs one logit per patch (image-level binary classification).
        We broadcast each patch score to a (2, H, W) probability map so the
        result is compatible with the parent class stitch_predictions method.

        Args:
            patches: (N, H, W, C) numpy array

        Returns:
            (N, 2, H, W) array of [no-fire, fire] probabilities, None, None
        """
        n, h, w, _ = patches.shape
        all_probs = np.zeros((n, 2, h, w), dtype=np.float32)

        for i in range(0, n, self.batch_size):
            batch = patches[i:i + self.batch_size]
            batch_tensor = (
                torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(self.device)
            )
            logits = self.model(batch_tensor)       # (B,)
            fire_prob = torch.sigmoid(logits).cpu().numpy()  # (B,)

            for j, fp in enumerate(fire_prob):
                all_probs[i + j, 0] = 1.0 - fp    # no-fire channel
                all_probs[i + j, 1] = fp           # fire channel

        return all_probs, None, None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference_scratch.py <model_path> <image_path>")
        sys.exit(1)

    pipeline = ScratchInferencePipeline(sys.argv[1])
    result = pipeline.predict_from_file(sys.argv[2])

    print(f"Has fire: {result.has_fire}")
    print(f"Fire confidence: {result.fire_confidence:.2%}")
    print(f"Fire fraction: {result.fire_fraction:.2%}")
    print(f"Counts: {result.severity_counts}")
