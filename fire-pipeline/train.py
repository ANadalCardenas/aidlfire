#!/usr/bin/env python3
"""
Training Script for Fire Detection/Segmentation Model.

Single model that performs both:
1. Pixel-wise fire segmentation (with severity levels)
2. Binary fire detection (derived from segmentation)

Usage:
    # Basic training with DEL mask (binary fire/no-fire)
    uv run python train.py --patches-dir ./patches --num-classes 2

    # Training with GRA mask (5 severity levels)
    uv run python train.py --patches-dir ./patches --num-classes 5

    # With W&B logging
    uv run python train.py --patches-dir ./patches --wandb --project fire-detection

    # Resume from checkpoint
    uv run python train.py --patches-dir ./patches --resume checkpoints/best_model.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    WildfireDataModule,
    compute_class_weights,
    get_training_augmentation,
    get_strong_augmentation,
)
from model import FireSegmentationModel, FireDualHeadModel, CombinedLoss
from metrics import CombinedMetrics
from constants import get_device, get_device_name, get_class_names, NUM_INPUT_CHANNELS


def setup_wandb(config: dict, project: str, run_name: str | None = None, wandb_dir: Path | None = None):
    """Initialize Weights & Biases logging."""
    try:
        import os
        import sys

        # Avoid local ./wandb run cache shadowing the wandb package: ensure
        # site-packages is searched before cwd when importing wandb
        if wandb_dir is not None:
            os.environ.setdefault("WANDB_DIR", str(wandb_dir))
        site_packages = [p for p in sys.path if "site-packages" in p]
        if site_packages:
            sys.path.insert(0, site_packages[0])
        import wandb
        if site_packages:
            sys.path.pop(0)

        wandb.init(
            project=project,
            name=run_name,
            config=config,
        )
        return wandb
    except ImportError:
        print("Warning: wandb not installed. Install with: pip install wandb")
        return None


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
    config: dict,
):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    metrics: CombinedMetrics,
    *,
    criterion_severity: nn.Module | None = None,
    dual_head: bool = False,
) -> dict:
    """Train for one epoch."""
    model.train()
    metrics.reset()

    total_loss = 0
    loss_components = {"ce_loss": 0, "dice_loss": 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        if dual_head and isinstance(model, FireDualHeadModel):
            binary_logits, severity_logits = model(images)
            mask_binary = (masks > 0).long()
            result_bin = criterion(binary_logits, mask_binary)
            loss_bin = result_bin[0] if isinstance(result_bin, tuple) else result_bin
            result_sev = criterion_severity(severity_logits, masks)
            loss_sev = result_sev[0] if isinstance(result_sev, tuple) else result_sev
            loss = loss_bin + loss_sev
            logits = binary_logits  # metrics on binary for primary
        else:
            logits = model(images)
            result = criterion(logits, masks)
            if isinstance(result, tuple):
                loss, components = result
                for k, v in components.items():
                    loss_components[k] += v
            else:
                loss = result

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if dual_head and isinstance(model, FireDualHeadModel):
                metrics.update(binary_logits, (masks > 0).long())
            else:
                metrics.update(logits, masks)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_metrics = metrics.compute()
    epoch_metrics["loss"] = total_loss / num_batches
    for k, v in loss_components.items():
        epoch_metrics[k] = v / num_batches

    return epoch_metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    metrics: CombinedMetrics,
    *,
    criterion_severity: nn.Module | None = None,
    dual_head: bool = False,
) -> dict:
    """Validate for one epoch."""
    model.eval()
    metrics.reset()

    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        if dual_head and isinstance(model, FireDualHeadModel):
            binary_logits, severity_logits = model(images)
            mask_binary = (masks > 0).long()
            result_bin = criterion(binary_logits, mask_binary)
            loss_bin = result_bin[0] if isinstance(result_bin, tuple) else result_bin
            result_sev = criterion_severity(severity_logits, masks)
            loss_sev = result_sev[0] if isinstance(result_sev, tuple) else result_sev
            loss = loss_bin + loss_sev
            metrics.update(binary_logits, mask_binary)
        else:
            logits = model(images)
            result = criterion(logits, masks)
            if isinstance(result, tuple):
                loss, _ = result
            else:
                loss = result
            metrics.update(logits, masks)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_metrics = metrics.compute()
    epoch_metrics["loss"] = total_loss / num_batches

    return epoch_metrics


def train(
    patches_dir: Path,
    output_dir: Path,
    num_classes: int = 2,
    encoder_name: str = "resnet34",
    architecture: str = "unet",
    batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    use_class_weights: bool = True,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    use_weighted_sampling: bool = False,
    fire_sample_weight: float = 5.0,
    use_fire_augment: bool = True,
    num_workers: int = 4,
    device: str = "auto",
    resume: Path | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    early_stopping_patience: int = 10,
    save_every: int = 5,
    overwrite_output_dir: bool = False,
    use_dual_head: bool = False,
):
    """
    Main training function.

    Args:
        patches_dir: Directory containing train/val/test patch subdirectories
        output_dir: Directory to save checkpoints and logs
        num_classes: Number of classes (2 for DEL, 5 for GRA)
        encoder_name: Backbone encoder (resnet34, efficientnet-b0, etc.)
        architecture: Model architecture (unet, unetplusplus, deeplabv3plus)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        use_class_weights: Use inverse frequency class weights
        use_focal_loss: Use focal loss instead of cross entropy
        focal_gamma: Focal loss gamma parameter
        use_weighted_sampling: Oversample fire patches
        fire_sample_weight: Weight for fire patches in sampling
        use_fire_augment: Apply stronger augmentation to fire patches
        num_workers: DataLoader workers
        device: Device (auto, cuda, mps, cpu)
        resume: Path to checkpoint to resume from
        wandb_project: W&B project name for logging
        wandb_run_name: W&B run name
        early_stopping_patience: Epochs without improvement before stopping
        save_every: Save checkpoint every N epochs
        use_dual_head: If True, use FireDualHeadModel (binary + severity); requires num_classes=5 (GRA)
    """
    if use_dual_head and num_classes != 5:
        raise SystemExit("--dual-head requires --num-classes 5 (GRA patches).")
    # Setup output directory
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite_output_dir:
            raise SystemExit(
                f"Output directory already exists: {output_dir}\n"
                "Remove it or pass --overwrite-output-dir to allow overwriting."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Setup device using shared utility
    device = get_device(device)
    device_label = get_device_name(device)

    print(f"\n{'='*60}")
    print(f"  FIRE DETECTION TRAINING")
    print(f"{'='*60}")
    print(f"  Device: {device_label}")
    if device.type == "cpu":
        print(f"  (Tip: use --device cuda if you have a GPU; install PyTorch with CUDA for GPU training)")
    print(f"  Patches: {patches_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Classes: {num_classes}")
    print(f"  Encoder: {encoder_name}")
    print(f"  Architecture: {architecture}")
    print(f"{'='*60}\n")

    # Class names for metrics (use shared constants)
    class_names = list(get_class_names(2 if use_dual_head else num_classes))

    # Config for logging
    config = {
        "patches_dir": str(patches_dir),
        "num_classes": num_classes,
        "in_channels": NUM_INPUT_CHANNELS,
        "dual_head": use_dual_head,
        "encoder_name": encoder_name,
        "architecture": architecture,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "use_class_weights": use_class_weights,
        "use_focal_loss": use_focal_loss,
        "focal_gamma": focal_gamma,
        "use_weighted_sampling": use_weighted_sampling,
        "fire_sample_weight": fire_sample_weight,
        "use_fire_augment": use_fire_augment,
        "device": str(device),
    }

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Setup W&B (use output_dir for wandb cache to avoid ./wandb shadowing the package)
    wandb = None
    if wandb_project:
        wandb = setup_wandb(config, wandb_project, wandb_run_name, wandb_dir=output_dir / "wandb")

    # Compute class weights
    class_weights = None
    if use_class_weights:
        print("Computing class weights...")
        weights = compute_class_weights(patches_dir / "train", num_classes=num_classes)
        class_weights = torch.tensor(weights, device=device)
        print(f"Class weights: {weights}")

    # Setup data module
    print("\nSetting up data loaders...")
    fire_augment = get_strong_augmentation() if use_fire_augment else None

    data_module = WildfireDataModule(
        patches_root=patches_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_augment=get_training_augmentation(),
        fire_augment=fire_augment,
        use_weighted_sampling=use_weighted_sampling,
        fire_sample_weight=fire_sample_weight,
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Validation samples: {len(data_module.val_dataset)}")

    # Create model
    print("\nCreating model...")
    if use_dual_head:
        model = FireDualHeadModel(
            encoder_name=encoder_name,
            in_channels=NUM_INPUT_CHANNELS,
            encoder_weights="imagenet",
            architecture=architecture,
        )
        criterion_severity = CombinedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=class_weights)
    else:
        model = FireSegmentationModel(
            encoder_name=encoder_name,
            num_classes=num_classes,
            in_channels=NUM_INPUT_CHANNELS,
            encoder_weights="imagenet",
            architecture=architecture,
        )
        criterion_severity = None

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup loss function
    if use_focal_loss:
        print(f"\nUsing Focal Loss (gamma={focal_gamma})")
        criterion = CombinedLoss(
            ce_weight=0.5,
            dice_weight=0.5,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )
    else:
        print("\nUsing CrossEntropy + Dice Loss")
        criterion = CombinedLoss(
            ce_weight=0.5,
            dice_weight=0.5,
            class_weights=class_weights,
        )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_metric = 0
    if resume:
        print(f"\nResuming from: {resume}")
        checkpoint = load_checkpoint(resume, model, optimizer)
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["metrics"].get("fire_iou", 0)
        if use_dual_head and isinstance(model, FireDualHeadModel):
            model.freeze_severity_head()  # keep severity frozen if it was
        print(f"Resuming from epoch {start_epoch}, best fire_iou: {best_metric:.4f}")

    # Setup metrics (for dual-head we track binary metrics)
    train_metrics = CombinedMetrics(num_classes=2 if use_dual_head else num_classes, class_names=class_names)
    val_metrics = CombinedMetrics(num_classes=2 if use_dual_head else num_classes, class_names=class_names)

    # Training loop
    print("\n" + "=" * 60)
    print("  STARTING TRAINING")
    print("=" * 60 + "\n")

    epochs_without_improvement = 0

    for epoch in range(start_epoch, num_epochs):
        # Train
        train_results = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, train_metrics,
            criterion_severity=criterion_severity,
            dual_head=use_dual_head,
        )

        # Validate
        val_results = validate_epoch(
            model, val_loader, criterion, device, epoch, val_metrics,
            criterion_severity=criterion_severity,
            dual_head=use_dual_head,
        )

        # Update scheduler
        scheduler.step(val_results["fire_iou"])

        # Log results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_results['loss']:.4f}")
        print(f"  Val Loss: {val_results['loss']:.4f}")
        print(f"  Val Fire IoU: {val_results['fire_iou']:.4f}")
        print(f"  Val Fire Recall: {val_results['fire_recall']:.4f}")
        print(f"  Val Detection F1: {val_results['detection_f1']:.4f}")

        # W&B logging
        if wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_results["loss"],
                "train/fire_iou": train_results["fire_iou"],
                "train/detection_f1": train_results["detection_f1"],
                "val/loss": val_results["loss"],
                "val/fire_iou": val_results["fire_iou"],
                "val/fire_recall": val_results["fire_recall"],
                "val/detection_f1": val_results["detection_f1"],
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Save best model
        if val_results["fire_iou"] > best_metric:
            best_metric = val_results["fire_iou"]
            epochs_without_improvement = 0

            save_checkpoint(
                model, optimizer, epoch, val_results,
                checkpoints_dir / "best_model.pt", config
            )
            print(f"  ✓ New best model saved (fire_iou: {best_metric:.4f})")
        else:
            epochs_without_improvement += 1

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_results,
                checkpoints_dir / f"checkpoint_epoch_{epoch}.pt", config
            )

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping: no improvement for {early_stopping_patience} epochs")
            break

    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_results,
        checkpoints_dir / "final_model.pt", config
    )

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Fire IoU: {best_metric:.4f}")
    print(f"  Checkpoints saved to: {checkpoints_dir}")

    if wandb:
        wandb.finish()

    return best_metric


def main():
    parser = argparse.ArgumentParser(
        description="Train fire detection/segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--patches-dir",
        type=Path,
        default=Path("./patches"),
        help="Directory containing train/val/test patches",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints and logs (required)",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Allow overwriting if output directory already exists",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        choices=[2, 5],
        help="Number of classes (2=DEL binary, 5=GRA severity); required 5 for --dual-head",
    )
    parser.add_argument(
        "--dual-head",
        action="store_true",
        help="Use FireDualHeadModel (binary + severity heads); requires --num-classes 5",
    )

    # Model arguments
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        choices=["resnet18", "resnet34", "resnet50", "efficientnet-b0", "efficientnet-b1", "mobilenet_v2"],
        help="Encoder backbone",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="unet",
        choices=["unet", "unetplusplus", "deeplabv3plus"],
        help="Segmentation architecture",
    )

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Loss arguments
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights")
    parser.add_argument("--focal-loss", action="store_true", help="Use focal loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")

    # Sampling arguments
    parser.add_argument("--weighted-sampling", action="store_true", help="Oversample fire patches")
    parser.add_argument("--fire-weight", type=float, default=5.0, help="Fire patch sample weight")
    parser.add_argument("--no-fire-augment", action="store_true", help="Disable fire-specific augmentation")

    # Device arguments
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])

    # Checkpoint arguments
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Logging arguments
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--project", type=str, default="fire-detection", help="W&B project name")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")

    args = parser.parse_args()

    # Run training
    train(
        patches_dir=args.patches_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        encoder_name=args.encoder,
        architecture=args.architecture,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_class_weights=not args.no_class_weights,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        use_weighted_sampling=args.weighted_sampling,
        fire_sample_weight=args.fire_weight,
        use_fire_augment=not args.no_fire_augment,
        num_workers=args.num_workers,
        device=args.device,
        resume=args.resume,
        wandb_project=args.project if args.wandb else None,
        wandb_run_name=args.run_name,
        early_stopping_patience=args.patience,
        save_every=args.save_every,
        overwrite_output_dir=args.overwrite_output_dir,
        use_dual_head=args.dual_head,
    )


if __name__ == "__main__":
    main()
