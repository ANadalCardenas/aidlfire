"""Tests for FireDualHeadModel."""

import pytest
import torch

from model import FireDualHeadModel


@pytest.mark.skipif(
    not hasattr(torch.nn, "Module"),
    reason="PyTorch required",
)
class TestFireDualHeadModel:
    """Tests for dual-head fire model."""

    def test_forward_returns_two_logits(self):
        """Forward returns (binary_logits, severity_logits)."""
        model = FireDualHeadModel(
            encoder_name="resnet18",
            in_channels=7,
            encoder_weights=None,
            architecture="unet",
        )
        model.eval()
        x = torch.randn(2, 7, 64, 64)
        binary_logits, severity_logits = model(x)

        assert binary_logits.shape == (2, 2, 64, 64)
        assert severity_logits.shape == (2, 5, 64, 64)

    def test_predict_binary(self):
        """predict_binary returns (B, H, W) class indices."""
        model = FireDualHeadModel(
            encoder_name="resnet18",
            in_channels=7,
            encoder_weights=None,
        )
        model.eval()
        x = torch.randn(1, 7, 32, 32)
        seg = model.predict_binary(x)
        assert seg.shape == (1, 32, 32)
        assert seg.dtype == torch.int64
        assert seg.min() >= 0 and seg.max() <= 1

    def test_predict_severity(self):
        """predict_severity returns (B, H, W) class indices 0-4."""
        model = FireDualHeadModel(
            encoder_name="resnet18",
            in_channels=7,
            encoder_weights=None,
        )
        model.eval()
        x = torch.randn(1, 7, 32, 32)
        seg = model.predict_severity(x)
        assert seg.shape == (1, 32, 32)
        assert seg.dtype == torch.int64
        assert seg.min() >= 0 and seg.max() <= 4

    def test_freeze_severity_head(self):
        """freeze_severity_head makes severity head parameters not require grad."""
        model = FireDualHeadModel(
            encoder_name="resnet18",
            in_channels=7,
            encoder_weights=None,
        )
        model.freeze_severity_head()

        for name, p in model.named_parameters():
            if "severity_head" in name:
                assert not p.requires_grad
            else:
                assert p.requires_grad

    def test_unfreeze_severity_head(self):
        """unfreeze_severity_head restores grad on severity head."""
        model = FireDualHeadModel(
            encoder_name="resnet18",
            in_channels=7,
            encoder_weights=None,
        )
        model.freeze_severity_head()
        model.unfreeze_severity_head()

        for p in model.parameters():
            assert p.requires_grad

    def test_8_channels(self):
        """Model accepts 8 input channels (e.g. with NDVI)."""
        model = FireDualHeadModel(
            encoder_name="resnet18",
            in_channels=8,
            encoder_weights=None,
        )
        model.eval()
        x = torch.randn(1, 8, 64, 64)
        binary_logits, severity_logits = model(x)
        assert binary_logits.shape == (1, 2, 64, 64)
        assert severity_logits.shape == (1, 5, 64, 64)
