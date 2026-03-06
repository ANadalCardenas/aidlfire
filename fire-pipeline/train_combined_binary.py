#!/usr/bin/env python3
"""
Phase 1: Train binary fire model on combined CEMS + Sen2Fire data.

Combines CEMS DEL patches (Europe) and Sen2Fire patches (Australia) for
geographic diversity. Uses 8 channels (7 bands + NDVI) by default.

Usage:
    uv run python train_combined_binary.py \\
        --patches-dir ./patches \\
        --sen2fire-dir ../data-sen2fire \\
        --output-dir ./output/combined_binary
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from constants import get_device, get_device_name, get_class_names
from dataset import (
    WildfirePatchDataset,
    get_patch_num_channels,
    get_training_augmentation,
    get_strong_augmentation,
    compute_class_weights,
)
from model import FireSegmentationModel, CombinedLoss
from metrics import CombinedMetrics
from sen2fire_dataset import Sen2FireDataset


def setup_wandb(config: dict, project: str, run_name: str | None = None, wandb_dir: Path | None = None):
    """Initialize Weights & Biases logging."""
    try:
        if wandb_dir is not None:
            os.environ.setdefault("WANDB_DIR", str(wandb_dir))
        site_packages = [p for p in sys.path if "site-packages" in p]
        if site_packages:
            sys.path.insert(0, site_packages[0])
        import wandb
        if site_packages:
            sys.path.pop(0)
        wandb.init(project=project, name=run_name, config=config)
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
) -> None:
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }, path)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    metrics: CombinedMetrics,
) -> dict:
    """Train one epoch (binary only)."""
    model.train()
    metrics.reset()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        result = criterion(logits, masks)
        loss = result[0] if isinstance(result, tuple) else result

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            metrics.update(logits, masks)

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    out = metrics.compute()
    out["loss"] = total_loss / num_batches
    return out


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    metrics: CombinedMetrics,
) -> dict:
    """Validate one epoch."""
    model.eval()
    metrics.reset()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        result = criterion(logits, masks)
        loss = result[0] if isinstance(result, tuple) else result

        metrics.update(logits, masks)
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    out = metrics.compute()
    out["loss"] = total_loss / num_batches
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Train binary fire model on CEMS + Sen2Fire combined",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--patches-dir", type=Path, required=True, help="CEMS DEL patches (train/val/test)")
    parser.add_argument("--sen2fire-dir", type=Path, required=True, help="Sen2Fire root (scene1, scene2, scene3, scene4)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for checkpoints")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-cloud-score", type=float, default=0.5, help="Exclude Sen2Fire patches above this cloud score")
    parser.add_argument("--no-s2cloudless", action="store_true", help="Use rule-based cloud filter only for Sen2Fire")
    parser.add_argument("--no-ndvi", action="store_true", help="Use 7 channels (no NDVI)")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights")
    parser.add_argument("--no-fire-augment", action="store_true", help="Disable fire-specific augmentation")
    parser.add_argument("--encoder", type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "mobilenet_v2"])
    parser.add_argument("--architecture", type=str, default="unet", choices=["unet", "unetplusplus", "deeplabv3plus"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--project", type=str, default="fire-detection", help="W&B project name")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {get_device_name(device)}")

    if not args.patches_dir.exists():
        raise SystemExit(f"CEMS patches dir not found: {args.patches_dir}")
    if not args.sen2fire_dir.exists():
        raise SystemExit(f"Sen2Fire dir not found: {args.sen2fire_dir}")

    args.output_dir = Path(args.output_dir)
    if args.output_dir.exists() and not args.overwrite_output_dir:
        raise SystemExit(f"Output dir exists: {args.output_dir}. Use --overwrite-output-dir to overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    if args.no_ndvi:
        in_channels = 7
        include_ndvi = False
    else:
        in_channels = get_patch_num_channels(args.patches_dir)
        include_ndvi = in_channels == 8

    train_augment = get_training_augmentation()
    fire_augment = get_strong_augmentation() if not args.no_fire_augment else None

    cems_train = WildfirePatchDataset(
        args.patches_dir / "train",
        augment=train_augment,
        fire_augment=fire_augment,
        fire_threshold=0.01,
    )
    cems_val = WildfirePatchDataset(
        args.patches_dir / "val",
        augment=None,
    )

    sen2fire_train = Sen2FireDataset(
        args.sen2fire_dir,
        split="train",
        include_ndvi=include_ndvi,
        max_cloud_score=args.max_cloud_score,
        use_s2cloudless=not args.no_s2cloudless,
        transform=train_augment,
    )
    sen2fire_val = Sen2FireDataset(
        args.sen2fire_dir,
        split="val",
        include_ndvi=include_ndvi,
        max_cloud_score=args.max_cloud_score,
        use_s2cloudless=not args.no_s2cloudless,
    )

    train_dataset = ConcatDataset([cems_train, sen2fire_train])
    val_dataset = ConcatDataset([cems_val, sen2fire_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"\n{'='*60}")
    print("  PHASE 1: COMBINED BINARY TRAINING (CEMS + Sen2Fire)")
    print(f"{'='*60}")
    print(f"  CEMS train: {len(cems_train)} | Sen2Fire train: {len(sen2fire_train)} | Total: {len(train_dataset)}")
    print(f"  CEMS val: {len(cems_val)} | Sen2Fire val: {len(sen2fire_val)} | Total: {len(val_dataset)}")
    print(f"  Channels: {in_channels} (NDVI: {include_ndvi})")
    print(f"{'='*60}\n")

    class_weights = None
    if not args.no_class_weights:
        weights = compute_class_weights(args.patches_dir / "train", num_classes=2)
        class_weights = torch.tensor(weights, device=device)
        print(f"Class weights: {weights}")

    model = FireSegmentationModel(
        encoder_name=args.encoder,
        num_classes=2,
        in_channels=in_channels,
        encoder_weights="imagenet",
        architecture=args.architecture,
    )
    model = model.to(device)

    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    config = {
        "patches_dir": str(args.patches_dir),
        "sen2fire_dir": str(args.sen2fire_dir),
        "num_classes": 2,
        "in_channels": in_channels,
        "dual_head": False,
        "encoder_name": args.encoder,
        "architecture": args.architecture,
        "combined_binary": True,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    wandb_run = None
    if args.wandb:
        wandb_run = setup_wandb(config, args.project, args.run_name, wandb_dir=args.output_dir / "wandb")

    class_names = list(get_class_names(2))
    train_metrics = CombinedMetrics(num_classes=2, class_names=class_names)
    val_metrics = CombinedMetrics(num_classes=2, class_names=class_names)

    best_metric = 0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_results = train_epoch(model, train_loader, criterion, optimizer, device, epoch, train_metrics)
        val_results = validate_epoch(model, val_loader, criterion, device, epoch, val_metrics)

        scheduler.step(val_results["fire_iou"])

        print(f"\nEpoch {epoch} | Train loss: {train_results['loss']:.4f} | Val loss: {val_results['loss']:.4f} | Val Fire IoU: {val_results['fire_iou']:.4f} | Val F1: {val_results['detection_f1']:.4f}")

        if wandb_run:
            wandb_run.log({
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

        if val_results["fire_iou"] > best_metric:
            best_metric = val_results["fire_iou"]
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_results, checkpoints_dir / "best_model.pt", config)
            print("  ✓ Best model saved")
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_results, checkpoints_dir / f"checkpoint_epoch_{epoch}.pt", config)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break

    save_checkpoint(model, optimizer, epoch, val_results, checkpoints_dir / "final_model.pt", config)
    if wandb_run:
        wandb_run.finish()
    print(f"\nDone. Best Val Fire IoU: {best_metric:.4f}")
    print(f"Checkpoints: {checkpoints_dir}")
    # Dynamic Phase 2 suggestion from script inputs
    ckpt_path = checkpoints_dir / "best_model.pt"
    patches_gra = args.patches_dir.parent / "patches_gra"
    out_name = args.output_dir.name
    phase2_name = out_name.replace("combined_binary", "finetune_severity") if "combined_binary" in out_name else f"{out_name}_severity"
    phase2_output = args.output_dir.parent / phase2_name
    print("\nNext step: Phase 2 - Fine-tune severity head on CEMS GRA:")
    print(f"  uv run python train_severity_finetune.py --checkpoint {ckpt_path} --patches-dir {patches_gra} --output-dir {phase2_output}")


if __name__ == "__main__":
    main()
