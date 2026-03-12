"""
Inference pipeline for the UNet scratch segmentation model.

Usage:
    from inference_unet_scratch import UNetScratchInferencePipeline

    pipeline = UNetScratchInferencePipeline("checkpoints/best_model.pt")
    result = pipeline.predict_from_array(image_array)  # (H, W, 8)
    result = pipeline.predict_from_file("satellite_image.tif")

Model output:
    UNet outputs per-pixel class logits (B, num_classes, H, W).
    Softmax is applied to get per-pixel class probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from unet_scratch import UNet
from inference import FireInferencePipeline, InferenceResult


class UNetScratchInferencePipeline(FireInferencePipeline):
    """
    Inference pipeline for UNet from scratch (pixel-level segmentation).

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
        """Load UNet from checkpoint."""
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        state_dict = checkpoint["model_state_dict"]

        # Infer in_channels from checkpoint (config may be wrong, e.g. hardcoded 7 vs actual 8)
        first_conv_key = "encoder.encBlocks.0.conv1.weight"
        if first_conv_key in state_dict:
            in_channels = int(state_dict[first_conv_key].shape[1])
        else:
            in_channels = config.get("in_channels", 8)

        num_classes = config.get("num_classes", 2)

        model = UNet(in_channels=in_channels, num_classes=num_classes, retainDim=True)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        config["in_channels"] = in_channels  # Ensure preprocessing uses correct channel count
        config["dual_head"] = False
        return model, config

    @torch.no_grad()
    def predict_patches(
        self, patches: np.ndarray
    ) -> tuple[np.ndarray, None, None]:
        """
        Run inference on patches using UNet.

        Args:
            patches: (N, H, W, C) numpy array

        Returns:
            (N, num_classes, H, W) softmax probabilities, None, None
        """
        all_probs = []

        for i in range(0, len(patches), self.batch_size):
            batch = patches[i:i + self.batch_size]
            batch_tensor = (
                torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(self.device)
            )
            logits = self.model(batch_tensor)           # (B, num_classes, H, W)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0), None, None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference_unet_scratch.py <model_path> <image_path>")
        sys.exit(1)

    pipeline = UNetScratchInferencePipeline(sys.argv[1])
    result = pipeline.predict_from_file(sys.argv[2])

    print(f"Has fire: {result.has_fire}")
    print(f"Fire confidence: {result.fire_confidence:.2%}")
    print(f"Fire fraction: {result.fire_fraction:.2%}")
    print(f"Counts: {result.severity_counts}")
