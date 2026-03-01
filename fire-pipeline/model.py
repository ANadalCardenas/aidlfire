"""
Fire Detection Model - Single and dual-head segmentation models.

This module provides:
1. FireSegmentationModel: single head (2 or 5 classes)
2. FireDualHeadModel: binary fire head (2 classes) + severity head (5 classes)
   - Train on CEMS with both heads; freeze severity head when fine-tuning on Sen2Fire

Architecture:
    - Encoder: Pretrained ResNet/EfficientNet (adapts to 7/8 input channels)
    - Decoder: U-Net style decoder with skip connections
    - Output: (B, num_classes, H, W) or (binary_logits, severity_logits) for dual-head
"""

from typing import Literal

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


# Available encoder options (pretrained on ImageNet)
ENCODER_OPTIONS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "mobilenet_v2",
]


def create_segmentation_model(
    encoder_name: str = "resnet34",
    num_classes: int = 2,
    in_channels: int = 7,
    encoder_weights: str | None = "imagenet",
    architecture: Literal["unet", "unetplusplus", "deeplabv3plus"] = "unet",
) -> nn.Module:
    """
    Create a segmentation model for fire detection.

    Args:
        encoder_name: Backbone encoder (resnet34, efficientnet-b0, etc.)
        num_classes: Number of output classes (2 for DEL, 5 for GRA)
        in_channels: Number of input channels (7 for selected Sentinel-2 bands)
        encoder_weights: Pretrained weights ("imagenet" or None)
        architecture: Model architecture (unet, unetplusplus, deeplabv3plus)

    Returns:
        PyTorch segmentation model

    Example:
        model = create_segmentation_model(
            encoder_name="resnet34",
            num_classes=5,  # GRA severity levels
            in_channels=7,
        )
        output = model(images)  # (B, 5, 256, 256)
    """
    if not SMP_AVAILABLE:
        raise ImportError(
            "segmentation-models-pytorch is required. "
            "Install with: uv add segmentation-models-pytorch"
        )

    if architecture == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif architecture == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif architecture == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


class FireSegmentationModel(nn.Module):
    """
    Fire segmentation model with helper methods for both tasks.

    This wraps a segmentation model and provides convenient methods for:
    1. Getting segmentation predictions (pixel-wise)
    2. Getting binary classification (patch-level fire detection)
    3. Getting confidence scores

    Example:
        model = FireSegmentationModel(num_classes=5)

        # Forward pass
        logits = model(images)  # (B, 5, 256, 256)

        # Get predictions
        seg_pred = model.predict_segmentation(images)  # (B, 256, 256)
        has_fire = model.predict_fire_detection(images)  # (B,) boolean
        confidence = model.predict_fire_confidence(images)  # (B,) float
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        num_classes: int = 2,
        in_channels: int = 7,
        encoder_weights: str | None = "imagenet",
        architecture: str = "unet",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.model = create_segmentation_model(
            encoder_name=encoder_name,
            num_classes=num_classes,
            in_channels=in_channels,
            encoder_weights=encoder_weights,
            architecture=architecture,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits of shape (B, num_classes, H, W)
        """
        return self.model(x)

    def predict_segmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get segmentation predictions (class indices per pixel).

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Predictions of shape (B, H, W) with values in [0, num_classes-1]
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get per-class probabilities for each pixel.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Probabilities of shape (B, num_classes, H, W)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict_fire_detection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Binary fire detection: does this patch contain any fire?

        Derived from segmentation by checking if any pixel is predicted
        as fire (class > 0).

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Boolean tensor of shape (B,) - True if fire detected
        """
        seg_pred = self.predict_segmentation(x)
        return (seg_pred > 0).any(dim=(1, 2))

    def predict_fire_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fire detection confidence score.

        Returns the maximum fire probability across all pixels.
        This gives a patch-level confidence that fire is present.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Confidence scores of shape (B,) in [0, 1]
        """
        probs = self.predict_probabilities(x)

        # Sum probabilities for all fire classes (class > 0)
        fire_probs = probs[:, 1:, :, :].sum(dim=1)  # (B, H, W)

        # Max fire probability across all pixels
        max_fire_prob = fire_probs.amax(dim=(1, 2))  # (B,)

        return max_fire_prob

    def predict_fire_fraction(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicted fraction of pixels containing fire.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Fire fraction of shape (B,) in [0, 1]
        """
        seg_pred = self.predict_segmentation(x)
        fire_pixels = (seg_pred > 0).float()
        return fire_pixels.mean(dim=(1, 2))


def _get_decoder_out_channels(segmentation_head: nn.Module) -> int:
    """Get the number of channels feeding into the segmentation head (decoder output)."""
    if hasattr(segmentation_head, "conv"):
        return segmentation_head.conv.in_channels
    if isinstance(segmentation_head, nn.Conv2d):
        return segmentation_head.in_channels
    if isinstance(segmentation_head, nn.Sequential):
        for m in segmentation_head:
            if isinstance(m, nn.Conv2d):
                return m.in_channels
    raise ValueError("Cannot infer decoder output channels from segmentation head")


class FireDualHeadModel(nn.Module):
    """
    Fire model with two heads: binary fire (2 classes) and severity (5 classes).

    - Binary head: fire vs no-fire (for DEL / Sen2Fire).
    - Severity head: 5 GRA levels (for CEMS only); can be frozen when fine-tuning on Sen2Fire.

    Use train_dual_head on CEMS (GRA patches, derive binary from mask>0) then
    fine-tune on Sen2Fire with freeze_severity_head() and only binary loss.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 8,
        encoder_weights: str | None = "imagenet",
        architecture: Literal["unet", "unetplusplus", "deeplabv3plus"] = "unet",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.binary_classes = 2
        self.severity_classes = 5

        base = create_segmentation_model(
            encoder_name=encoder_name,
            num_classes=2,
            in_channels=in_channels,
            encoder_weights=encoder_weights,
            architecture=architecture,
        )
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.binary_head = base.segmentation_head

        decoder_out = _get_decoder_out_channels(self.binary_head)
        self.severity_head = nn.Conv2d(decoder_out, 5, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            binary_logits: (B, 2, H, W)
            severity_logits: (B, 5, H, W)
        """
        features = self.encoder(x)
        decoder_out = self.decoder(features)
        binary_logits = self.binary_head(decoder_out)
        severity_logits = self.severity_head(decoder_out)
        return binary_logits, severity_logits

    def freeze_severity_head(self) -> None:
        """Freeze the severity head (e.g. when fine-tuning on Sen2Fire with binary only)."""
        for p in self.severity_head.parameters():
            p.requires_grad = False

    def unfreeze_severity_head(self) -> None:
        """Unfreeze the severity head."""
        for p in self.severity_head.parameters():
            p.requires_grad = True

    def predict_binary(self, x: torch.Tensor) -> torch.Tensor:
        """Binary segmentation (B, H, W) with values 0 or 1."""
        binary_logits, _ = self.forward(x)
        return binary_logits.argmax(dim=1)

    def predict_severity(self, x: torch.Tensor) -> torch.Tensor:
        """Severity segmentation (B, H, W) with values 0-4."""
        _, severity_logits = self.forward(x)
        return severity_logits.argmax(dim=1)

    def predict_fire_detection(self, x: torch.Tensor) -> torch.Tensor:
        """Patch-level fire detection (B,) boolean."""
        binary = self.predict_binary(x)
        return (binary > 0).any(dim=(1, 2))

    def predict_fire_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Max fire probability over pixels (B,) in [0, 1]."""
        binary_logits, _ = self.forward(x)
        probs = torch.softmax(binary_logits, dim=1)
        return probs[:, 1, :, :].amax(dim=(1, 2))


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice

    Good for imbalanced segmentation where fire pixels are rare.
    """

    def __init__(self, smooth: float = 1e-6, ignore_index: int | None = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) ground truth class indices

        Returns:
            Scalar dice loss
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Compute dice per class
        dims = (0, 2, 3)  # Sum over batch, height, width
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Ignore background class if specified
        if self.ignore_index is not None:
            mask = torch.ones(num_classes, device=dice.device, dtype=torch.bool)
            mask[self.ignore_index] = False
            dice = dice[mask]

        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p) = -α * (1 - p)^γ * log(p)

    Focuses learning on hard examples by down-weighting easy ones.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) ground truth class indices

        Returns:
            Scalar focal loss
        """
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: CrossEntropy + Dice.

    Combines pixel-wise CE loss with region-based Dice loss for
    better segmentation performance.
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
        focal_gamma: float | None = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        if focal_gamma is not None:
            self.ce_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        self.dice_loss = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) ground truth class indices

        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary with individual loss components
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total = self.ce_weight * ce + self.dice_weight * dice

        return total, {"ce_loss": ce.item(), "dice_loss": dice.item()}


def get_model_summary(model: nn.Module, input_size: tuple = (1, 7, 256, 256)) -> str:
    """Get a summary of model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = [
        f"Model: {model.__class__.__name__}",
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Input size: {input_size}",
    ]

    # Get output size
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        output = model(dummy_input)
        summary.append(f"Output size: {tuple(output.shape)}")

    return "\n".join(summary)


if __name__ == "__main__":
    # Test model creation
    print("Testing FireSegmentationModel...\n")

    model = FireSegmentationModel(
        encoder_name="resnet34",
        num_classes=5,  # GRA severity
        in_channels=7,
    )

    print(get_model_summary(model))

    # Test forward pass
    batch = torch.randn(4, 7, 256, 256)

    print("\nTesting forward passes...")
    logits = model(batch)
    print(f"Logits shape: {logits.shape}")

    seg_pred = model.predict_segmentation(batch)
    print(f"Segmentation shape: {seg_pred.shape}")

    has_fire = model.predict_fire_detection(batch)
    print(f"Fire detection: {has_fire}")

    confidence = model.predict_fire_confidence(batch)
    print(f"Fire confidence: {confidence}")

    fire_fraction = model.predict_fire_fraction(batch)
    print(f"Fire fraction: {fire_fraction}")

    # Test losses
    print("\nTesting losses...")
    targets = torch.randint(0, 5, (4, 256, 256))

    dice_loss = DiceLoss()
    print(f"Dice loss: {dice_loss(logits, targets):.4f}")

    focal_loss = FocalLoss(gamma=2.0)
    print(f"Focal loss: {focal_loss(logits, targets):.4f}")

    combined_loss = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    total, components = combined_loss(logits, targets)
    print(f"Combined loss: {total:.4f} (CE: {components['ce_loss']:.4f}, Dice: {components['dice_loss']:.4f})")

    print("\n✅ All tests passed!")
