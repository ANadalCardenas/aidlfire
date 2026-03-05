#!/usr/bin/env python3
"""
Fine-tune a CEMS-trained (dual-head or single-head) model on Sen2Fire.

- Loads checkpoint from CEMS training. If single-head, builds dual-head and loads
  encoder/decoder/binary head; severity head is randomly initialized and frozen.
- Freezes the severity head so only the binary fire head is trained.
- Trains on Sen2Fire train (scene1+2), validates on scene3, tests on scene4.

Usage:
    uv run python train_sen2fire_finetune.py \\
        --checkpoint ./output/dual_head/checkpoints/best_model.pt \\
        --sen2fire-dir ../data-sen2fire \\
        --output-dir ./output/sen2fire_finetune
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import (
    NUM_INPUT_CHANNELS,
    get_device,
    get_device_name,
    get_class_names,
)
from model import FireDualHeadModel, CombinedLoss
from metrics import CombinedMetrics
from sen2fire_dataset import Sen2FireDataset


def load_dual_head_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[FireDualHeadModel, dict]:
    """
    Load a FireDualHeadModel from checkpoint.

    If the checkpoint is single-head (FireSegmentationModel), build dual-head
    and load encoder/decoder/segmentation_head into encoder/decoder/binary_head;
    severity_head stays random and will be frozen.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    in_channels = config.get("in_channels", NUM_INPUT_CHANNELS)
    encoder_name = config.get("encoder_name", "resnet34")
    architecture = config.get("architecture", "unet")

    dual = FireDualHeadModel(
        encoder_name=encoder_name,
        in_channels=in_channels,
        encoder_weights=None,
        architecture=architecture,
    )

    state = checkpoint.get("model_state_dict", {})
    if config.get("dual_head"):
        dual.load_state_dict(state, strict=True)
    else:
        # Single-head checkpoint: keys are "model.encoder", "model.decoder", "model.segmentation_head"
        new_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_k = k.replace("model.segmentation_head", "binary_head").replace("model.", "")
                new_state[new_k] = v
            else:
                new_state[k] = v
        dual.load_state_dict(new_state, strict=False)

    dual = dual.to(device)
    dual.freeze_severity_head()
    return dual, config


def train_epoch(
    model: FireDualHeadModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    metrics: CombinedMetrics,
) -> dict:
    """Train one epoch (binary head only)."""
    model.train()
    metrics.reset()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        binary_logits, _ = model(images)
        result = criterion(binary_logits, masks)
        if isinstance(result, tuple):
            loss, _ = result
        else:
            loss = result

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            metrics.update(binary_logits, masks)

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

        binary_logits, _ = model(images)
        result = criterion(binary_logits, masks)
        if isinstance(result, tuple):
            loss, _ = result
        else:
            loss = result

        metrics.update(binary_logits, masks)
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
        description="Fine-tune CEMS model on Sen2Fire (binary head only, severity frozen)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to CEMS checkpoint (dual-head or single-head)")
    parser.add_argument("--sen2fire-dir", type=Path, required=True, help="Root directory of Sen2Fire (contains scene1, scene2, scene3, scene4)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-cloud-score", type=float, default=0.5, help="Exclude patches with cloud score above this")
    parser.add_argument("--no-s2cloudless", action="store_true", help="Use rule-based cloud filter only (skip s2cloudless)")
    parser.add_argument("--no-ndvi", action="store_true", help="Use 7 channels (no NDVI)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--overwrite-output-dir", action="store_true")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {get_device_name(device)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sen2Fire dir: {args.sen2fire_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Severity head: frozen (binary head only trained)")

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.sen2fire_dir.exists():
        raise SystemExit(f"Sen2Fire dir not found: {args.sen2fire_dir}")

    args.output_dir = Path(args.output_dir)
    if args.output_dir.exists() and not args.overwrite_output_dir:
        raise SystemExit(f"Output dir exists: {args.output_dir}. Use --overwrite-output-dir to overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    model, config = load_dual_head_from_checkpoint(args.checkpoint, device)
    in_channels = config.get("in_channels", NUM_INPUT_CHANNELS)
    include_ndvi = in_channels == 8 and not args.no_ndvi

    train_ds = Sen2FireDataset(
        args.sen2fire_dir,
        split="train",
        include_ndvi=include_ndvi,
        max_cloud_score=args.max_cloud_score,
        use_s2cloudless=not args.no_s2cloudless,
    )
    val_ds = Sen2FireDataset(
        args.sen2fire_dir,
        split="val",
        include_ndvi=include_ndvi,
        max_cloud_score=args.max_cloud_score,
        use_s2cloudless=not args.no_s2cloudless,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    class_names = list(get_class_names(2))
    train_metrics = CombinedMetrics(num_classes=2, class_names=class_names)
    val_metrics = CombinedMetrics(num_classes=2, class_names=class_names)

    config_save = {
        **config,
        "dual_head": True,
        "num_classes": 2,
        "in_channels": 8 if include_ndvi else 7,
        "sen2fire_finetune": True,
    }
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(config_save, f, indent=2)

    best_metric = 0
    epochs_without_improvement = 0
    epoch = 0
    val_results = {}

    for epoch in range(args.epochs):
        train_results = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, train_metrics
        )
        val_results = validate_epoch(
            model, val_loader, criterion, device, epoch, val_metrics
        )

        scheduler.step(val_results["fire_iou"])

        print(f"\nEpoch {epoch} | Train loss: {train_results['loss']:.4f} | Val loss: {val_results['loss']:.4f} | Val Fire IoU: {val_results['fire_iou']:.4f} | Val F1: {val_results['detection_f1']:.4f}")

        if val_results["fire_iou"] > best_metric:
            best_metric = val_results["fire_iou"]
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
    print(f"\nDone. Best Val Fire IoU: {best_metric:.4f}")
    print(f"Checkpoints: {checkpoints_dir}")


if __name__ == "__main__":
    main()
