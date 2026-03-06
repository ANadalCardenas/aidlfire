#!/usr/bin/env python3
"""
Phase 2: Fine-tune severity head on CEMS GRA data.

Loads Phase 1 binary checkpoint (from train_combined_binary.py), builds
FireDualHeadModel, freezes encoder + binary head, and trains only the
severity head on CEMS GRA patches (5 severity classes).

Usage:
    uv run python train_severity_finetune.py \\
        --checkpoint ./output/combined_binary/checkpoints/best_model.pt \\
        --patches-dir ./patches_gra \\
        --output-dir ./output/severity_finetune
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import NUM_INPUT_CHANNELS, get_device, get_device_name, get_class_names
from dataset import (
    WildfireDataModule,
    get_patch_num_channels,
    compute_class_weights,
    get_training_augmentation,
    get_strong_augmentation,
)
from model import FireDualHeadModel, CombinedLoss
from metrics import CombinedMetrics


def load_binary_checkpoint_for_severity_finetune(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[FireDualHeadModel, dict]:
    """
    Load a binary checkpoint and build FireDualHeadModel for severity fine-tuning.

    Loads encoder, decoder, binary_head from checkpoint. Severity head stays
    randomly initialized. Freezes encoder, decoder, binary_head.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    in_channels = config.get("in_channels", NUM_INPUT_CHANNELS)
    encoder_name = config.get("encoder_name", "resnet34")
    architecture = config.get("architecture", "unet")

    model = FireDualHeadModel(
        encoder_name=encoder_name,
        in_channels=in_channels,
        encoder_weights=None,
        architecture=architecture,
    )

    state = checkpoint.get("model_state_dict", {})
    if config.get("dual_head"):
        model.load_state_dict(state, strict=True)
    else:
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_k = k.replace("model.segmentation_head", "binary_head").replace("model.", "")
                new_state[new_k] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=False)

    model = model.to(device)

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False
    for p in model.binary_head.parameters():
        p.requires_grad = False

    return model, config


def train_epoch(
    model: FireDualHeadModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    metrics: CombinedMetrics,
) -> dict:
    """Train one epoch (severity head only)."""
    model.train()
    metrics.reset()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        _, severity_logits = model(images)
        result = criterion(severity_logits, masks)
        loss = result[0] if isinstance(result, tuple) else result

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            metrics.update(severity_logits, masks)

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    out = metrics.compute()
    out["loss"] = total_loss / num_batches
    return out


@torch.no_grad()
def validate_epoch(
    model: FireDualHeadModel,
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

        _, severity_logits = model(images)
        result = criterion(severity_logits, masks)
        loss = result[0] if isinstance(result, tuple) else result

        metrics.update(severity_logits, masks)
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    out = metrics.compute()
    out["loss"] = total_loss / num_batches
    return out


def save_checkpoint(
    model: FireDualHeadModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
    config: dict,
) -> None:
    """Save dual-head checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config_save = {**config, "dual_head": True, "num_classes": 2}
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config_save,
    }, path)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Fine-tune severity head on CEMS GRA (encoder + binary frozen)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Phase 1 binary checkpoint")
    parser.add_argument("--patches-dir", type=Path, required=True, help="CEMS GRA patches (train/val/test)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for checkpoints")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for severity head")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights")
    parser.add_argument("--no-fire-augment", action="store_true", help="Disable fire-specific augmentation")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--overwrite-output-dir", action="store_true")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {get_device_name(device)}")

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.patches_dir.exists():
        raise SystemExit(f"Patches dir not found: {args.patches_dir}")

    args.output_dir = Path(args.output_dir)
    if args.output_dir.exists() and not args.overwrite_output_dir:
        raise SystemExit(f"Output dir exists: {args.output_dir}. Use --overwrite-output-dir to overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    model, config = load_binary_checkpoint_for_severity_finetune(args.checkpoint, device)
    in_channels = config.get("in_channels", NUM_INPUT_CHANNELS)

    fire_augment = get_strong_augmentation() if not args.no_fire_augment else None
    data_module = WildfireDataModule(
        patches_root=args.patches_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_augment=get_training_augmentation(),
        fire_augment=fire_augment,
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print("  PHASE 2: SEVERITY FINE-TUNING (CEMS GRA)")
    print(f"{'='*60}")
    print(f"  Train samples: {len(data_module.train_dataset)}")
    print(f"  Val samples: {len(data_module.val_dataset)}")
    print(f"  Trainable params: {trainable:,} / {total:,} (encoder + binary frozen)")
    print(f"{'='*60}\n")

    class_weights = None
    if not args.no_class_weights:
        weights = compute_class_weights(args.patches_dir / "train", num_classes=5)
        class_weights = torch.tensor(weights, device=device)

    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    config_save = {
        **config,
        "dual_head": True,
        "num_classes": 2,
        "in_channels": in_channels,
        "severity_finetune": True,
    }
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config_save, f, indent=2)

    class_names = list(get_class_names(5))
    train_metrics = CombinedMetrics(num_classes=5, class_names=class_names)
    val_metrics = CombinedMetrics(num_classes=5, class_names=class_names)

    best_metric = 0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_results = train_epoch(model, train_loader, criterion, optimizer, device, epoch, train_metrics)
        val_results = validate_epoch(model, val_loader, criterion, device, epoch, val_metrics)

        scheduler.step(val_results["mean_iou"])

        print(f"\nEpoch {epoch} | Train loss: {train_results['loss']:.4f} | Val loss: {val_results['loss']:.4f} | Val mean IoU: {val_results['mean_iou']:.4f} | Val fire IoU: {val_results['fire_iou']:.4f}")

        if val_results["mean_iou"] > best_metric:
            best_metric = val_results["mean_iou"]
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_results, checkpoints_dir / "best_model.pt", config_save)
            print("  ✓ Best model saved")
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_results, checkpoints_dir / f"checkpoint_epoch_{epoch}.pt", config_save)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break

    save_checkpoint(model, optimizer, epoch, val_results, checkpoints_dir / "final_model.pt", config_save)
    print(f"\nDone. Best Val mean IoU: {best_metric:.4f}")
    print(f"Checkpoints: {checkpoints_dir}")
    print("\nDual-head model ready for inference (binary + severity maps).")


if __name__ == "__main__":
    main()
