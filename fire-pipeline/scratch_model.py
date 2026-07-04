"""
scratch_model.py

A simple "scratch model" for wildfire detection (binary classification) using 8-channel input.

- Input:  (B, 8, H, W)
- Output: (B,) logits for BCEWithLogitsLoss

"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScratchFireModel(nn.Module):
    """
    Simple CNN classifier (scratch model) for patch-level fire presence.

    Architecture (PyTorch version of your Keras model):
      - 1x1 conv "spectral re-weighting": 8 -> 16
      - Conv blocks: 16->32->64->128 with BN + ReLU + MaxPool
      - Global pooling (adaptive) -> Dropout -> Linear(128->1)
    """

    def __init__(self, in_channels: int = 8, dropout: float = 0.3):
        super().__init__()

        # 1x1 conv acting purely on the channel dimension (no spatial mixing):
        # lets the model learn a weighted combination of the 8 input bands
        # (7 spectral bands + NDVI) before any spatial feature extraction starts.
        # bias=True here because this conv is NOT followed by BatchNorm, so its
        # own bias term is the only shift available at this stage.
        self.spectral = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Conv blocks below use bias=False because BatchNorm2d immediately follows
        # and already learns a per-channel shift; the conv's bias would be redundant.
        self.block1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # halves H and W
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Adaptive average pooling: collapses any H,W down to a single value per
        # channel, so the model isn't tied to one fixed input patch size.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Single logit output (no sigmoid): BCEWithLogitsLoss applies the sigmoid
        # internally, which is more numerically stable than doing it manually.
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,8,H,W)
        x = self.spectral(x)            # channel re-weighting, spatial size unchanged
        x = self.block1(x)              # 16 -> 32 channels, H/2, W/2
        x = self.block2(x)              # 32 -> 64 channels, H/4, W/4
        x = self.block3(x)              # 64 -> 128 channels, H/8, W/8
        x = self.gap(x)                 # (B,128,1,1)
        x = x.flatten(1)                # (B,128)
        x = self.head(x)                # (B,1)
        # squeeze(1) drops the trailing dim of size 1 so the output shape (B,)
        # matches what BCEWithLogitsLoss expects for binary targets.
        return x.squeeze(1)             # (B,)
