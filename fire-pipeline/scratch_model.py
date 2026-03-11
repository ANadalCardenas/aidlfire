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

        self.spectral = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
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

        # Adaptative: works with any H,W
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,8,H,W)
        x = self.spectral(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)                 # (B,128,1,1)
        x = x.flatten(1)                # (B,128)
        x = self.head(x)                # (B,1)
        return x.squeeze(1)             # (B,)
