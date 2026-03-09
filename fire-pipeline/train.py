#!/usr/bin/env python3
"""
Training Script for Fire Detection/Segmentation Model.

Single model that performs both:
1. Pixel-wise fire segmentation (with severity levels)
2. Binary fire detection (derived from segmentation) + 
YOLOv8-Seg baseline (RGB-only) using an exported version (train/val splits).

Usage:
    # Basic training with DEL mask (binary fire/no-fire)
    uv run python train.py --patches-dir ./patches --num-classes 2

    # Training with GRA mask (5 severity levels)
    uv run python train.py --patches-dir ./patches --num-classes 5

    # With W&B logging
    uv run python train.py --patches-dir ./patches --wandb --project fire-detection

    # Resume from checkpoint
    uv run python train.py --patches-dir ./patches --resume checkpoints/best_model.pt

Outputs:
    - SMP model checkpoints and logs are saved under:
        <output-dir>/encoder_<encoder>/
    - If enabled/used, YOLO baseline artifacts are saved under:
        <output-dir>/yolo_seg/
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from metric_logger import MetricLogger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from utils_report import print_run_summary, append_results_csv


from dataset import (
    WildfireDataModule,
    compute_class_weights,
    get_patch_num_channels,
    get_training_augmentation,
    get_strong_augmentation,
)
from model import FireSegmentationModel, FireDualHeadModel, CombinedLoss, ENCODER_OPTIONS
from scratch_model import ScratchFireModel
from unet_scratch import UNet
from metrics import CombinedMetrics
from constants import get_device, get_device_name, get_class_names

try:
    from ray import tune
    ray_report = tune.report
    try:
        from ray.tune import RunConfig
    except ImportError:
        try:
            from ray.air import RunConfig
        except ImportError:
            RunConfig = tune.RunConfig
except ImportError:
    tune = None
    RunConfig = None
    ray_report = None


# =============================================================================
# UTILITIES
# =============================================================================

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


def setup_tensorboard(output_dir: Path) -> SummaryWriter:
    """Initialize TensorBoard logging."""
    log_dir = output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


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
    max_batches: int | None = None,
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

        if max_batches is not None and num_batches >= max_batches:
            break

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
    max_batches: int | None = None,
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

        if max_batches is not None and num_batches >= max_batches:
            break

    epoch_metrics = metrics.compute()
    epoch_metrics["loss"] = total_loss / num_batches

    return epoch_metrics

def _binary_labels_from_mask(masks: torch.Tensor) -> torch.Tensor:
    """
    masks: (B,H,W) int
    returns y: (B,) float in {0,1} where 1 means "any fire pixel present"
    """
    return (masks > 0).any(dim=(1, 2)).float()


@torch.no_grad()
def _binary_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> dict:
    """
    logits: (B,) raw logits
    y: (B,) float {0,1}
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()

    tp = ((pred == 1) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


def _make_run_name(model: str, args) -> str:
    """Generate a descriptive W&B run name from model name and CLI args."""
    import re

    def _fmt(v: float) -> str:
        s = f"{v:.0e}"
        return re.sub(r"e([+-])0*(\d+)", lambda m: f"e{m.group(1)}{m.group(2)}", s)

    parts = [model, f"e{args.epochs}", f"bs{args.batch_size}",
             f"lr{_fmt(args.lr)}", f"wd{_fmt(args.weight_decay)}"]
    if getattr(args, "focal_loss", False):
        parts.append(f"focal{args.focal_gamma}")
    if getattr(args, "weighted_sampling", False):
        parts.append("ws")
    if getattr(args, "no_fire_augment", False):
        parts.append("noaug")
    return "_".join(parts)


# =============================================================================
# SCRATCH CLASSIFIER TRAINING
# =============================================================================

def train_scratch_classifier(
    patches_dir: Path,
    output_dir: Path,
    sen2fire_dir: Path | None = None,
    batch_size: int = 16,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device: str = "auto",
    dropout: float = 0.3,
    pos_weight: float | None = None,
    report_to_tune: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    results_csv: Path | None = None,
    dataset: str = "1rst dataset",
    max_batches: int | None = None,
    command: str = "",
):

    """
    Minimal training loop for ScratchFireModel (binary classification).
    - Label is derived from mask: y = 1 if any pixel > 0 else 0
    - Saves best checkpoint by lowest val loss
    """
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device_t = get_device(device)

    # Report print
    t0 = time.perf_counter()
    if device_t.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_t)    
   

    wandb = None
    if wandb_project:
        config = {
            "model": "ScratchFireModel",
            "patches_dir": str(patches_dir),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(device_t),
            "dropout": dropout,
            "pos_weight": pos_weight,
        }
        wandb = setup_wandb(config, wandb_project, wandb_run_name, wandb_dir=output_dir / "wandb")

    # Data
    dm = WildfireDataModule(
        patches_root=patches_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_augment=get_training_augmentation(),
        fire_augment=None,
        use_weighted_sampling=False,
        fire_sample_weight=1.0,
    )
    if sen2fire_dir is not None:
        from sen2fire_dataset import Sen2FireDataset
        from torch.utils.data import ConcatDataset
        sen2fire_train = Sen2FireDataset(sen2fire_dir, split="train", use_s2cloudless=False)
        sen2fire_val = Sen2FireDataset(sen2fire_dir, split="val", use_s2cloudless=False)
        combined_train = ConcatDataset([dm.train_dataset, sen2fire_train])
        combined_val = ConcatDataset([dm.val_dataset, sen2fire_val])
        train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(combined_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"Combined dataset: {len(combined_train)} train, {len(combined_val)} val (CEMS + Sen2Fire)")
    else:
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

    # Model
    model = ScratchFireModel(in_channels=get_patch_num_channels(patches_dir), dropout=dropout).to(device_t)
    total_params = sum(p.numel() for p in model.parameters())
    best_ckpt_path = checkpoints_dir / "best_model.pt"   

    # Loss + Optim
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device_t))
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_metrics = {}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = train_prec = train_rec = train_f1 = 0.0
        n = 0

        for images, masks in tqdm(train_loader, desc=f"Scratch Epoch {epoch} [Train]"):
            images = images.to(device_t)
            masks = masks.to(device_t)

            # binary label from mask
            y = _binary_labels_from_mask(masks)

            optimizer.zero_grad()
            logits = model(images) 
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                m = _binary_metrics_from_logits(logits, y)
                train_acc += m["acc"]
                train_prec += m["precision"]
                train_rec += m["recall"]
                train_f1 += m["f1"]

            train_loss += loss.item()
            n += 1

            if max_batches is not None and n >= max_batches:
                break

        train_loss /= max(n, 1)
        train_acc /= max(n, 1)
        train_prec /= max(n, 1)
        train_rec /= max(n, 1)
        train_f1 /= max(n, 1)

        model.eval()
        val_loss = 0.0
        val_acc = val_prec = val_rec = val_f1 = 0.0
        val_probs_all = []
        val_labels_all = []
        n = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Scratch Epoch {epoch} [Val]"):
                images = images.to(device_t)
                masks = masks.to(device_t)

                y = (masks > 0).any(dim=(1, 2)).float()
                logits = model(images)
                loss = criterion(logits, y)
                m = _binary_metrics_from_logits(logits, y)
                val_acc += m["acc"]
                val_prec += m["precision"]
                val_rec += m["recall"]
                val_f1 += m["f1"]

                val_probs_all.append(torch.sigmoid(logits).cpu())
                val_labels_all.append(y.cpu())

                val_loss += loss.item()
                n += 1

                if max_batches is not None and n >= max_batches:
                    break

        val_loss /= max(n, 1)
        val_acc /= max(n, 1)
        val_prec /= max(n, 1)
        val_rec /= max(n, 1)
        val_f1 /= max(n, 1)

        try:
            from sklearn.metrics import roc_auc_score
            import numpy as np
            probs_np = torch.cat(val_probs_all).numpy()
            labels_np = torch.cat(val_labels_all).numpy()
            val_auc = float(roc_auc_score(labels_np, probs_np)) if len(np.unique(labels_np)) > 1 else 0.0
        except Exception:
            val_auc = 0.0

        if report_to_tune:
            ray_report({"val_loss": val_loss, "val_f1": val_f1, "train_loss": train_loss, "epoch": epoch})

        print(
            f"\nScratch Epoch {epoch}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_prec={val_prec:.4f} val_rec={val_rec:.4f} val_auc={val_auc:.4f}"
        )
        if wandb:
            wandb.log({
                "epoch": epoch,

                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/precision": train_prec,
                "train/recall": train_rec,
                "train/f1": train_f1,

                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/precision": val_prec,
                "val/recall": val_rec,
                "val/f1": val_f1,
                "val/auc": val_auc,
            })

        # Save best checkpoint (highest val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_epoch = epoch
            best_metrics = {
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_precision": float(val_prec),
                "val_recall": float(val_rec),
                "val_f1": float(val_f1),
                "val_auc": float(val_auc),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "train_precision": float(train_prec),
                "train_recall": float(train_rec),
                "train_f1": float(train_f1),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=best_metrics,
                path=best_ckpt_path,
                config={
                    "model": "ScratchFireModel",
                    "patches_dir": str(patches_dir),
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "device": str(device_t),
                },
            )
            print(f"  ✓ Saved best scratch model (val_f1={best_val_f1:.4f})")

    train_time_s = time.perf_counter() - t0
    
    print_run_summary(
        title="SCRATCH CLASSIFIER",
        train_time_s=train_time_s,
        best_checkpoint=str(best_ckpt_path),
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        num_params=total_params,
    )

    if results_csv is not None:
        append_results_csv(results_csv, {
            "dataset": dataset,
            "model_name": "cnn_scratch",
            "wandb_run_name": wandb_run_name if wandb_project else "-",
            "VAL_F1": best_metrics.get("val_f1", ""),
            "val_loss": best_metrics.get("val_loss", ""),
            "val_precision": best_metrics.get("val_precision", ""),
            "val_recall": best_metrics.get("val_recall", ""),
            "val_accuracy": best_metrics.get("val_acc", ""),
            "val_auc_roc": best_metrics.get("val_auc", ""),
            "train_time_s": round(train_time_s, 2),
            "num_params": total_params,
            "best_epoch": best_epoch,
            "command": command,
        })

    if wandb:
        wandb.finish()
    return best_val_f1, best_epoch


# =============================================================================
# UNET SCRATCH SEGMENTATION TRAINING
# =============================================================================

def train_unet_scratch_segmentation(
    patches_dir: Path,
    output_dir: Path,
    sen2fire_dir: Path | None = None,
    num_classes: int = 2,
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
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    use_tensorboard: bool = True,
    early_stopping_patience: int = 10,
    save_every: int = 5,
    overwrite_output_dir: bool = False,
    results_csv: Path | None = None,
    dataset: str = "1rst dataset",
    max_batches: int | None = None,
    command: str = "",
) -> float:
    """
    Baseline training loop for the U-Net from scratch (segmentation).
    This is a baseline like YOLO or ScratchFireModel, not part of pretrained encoder sweeps.

    Returns:
        best_metric (best fire IoU)
    """
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

    metric_logger = MetricLogger(output_dir=output_dir)

    # Setup device
    device_t = get_device(device)
    device_label = get_device_name(device_t)

    t0 = time.perf_counter()
    if device_t.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_t)

    best_metrics = {}
    best_ckpt_path = checkpoints_dir / "best_model.pt"

    print(f"\n{'='*60}")
    print(f"  U-NET FROM SCRATCH (BASELINE SEGMENTATION)")
    print(f"{'='*60}")
    print(f"  Device: {device_label}")
    print(f"  Patches: {patches_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Classes: {num_classes}")
    print(f"{'='*60}\n")

    class_names = list(get_class_names(num_classes))

    # Config for logging
    config = {
        "patches_dir": str(patches_dir),
        "num_classes": num_classes,
        "model": "UNetScratch",
        "in_channels": 7,
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
        "device": str(device_t),
    }

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Setup W&B
    wandb = None
    if wandb_project:
        wandb = setup_wandb(config, wandb_project, wandb_run_name, wandb_dir=output_dir / "wandb")

    # Setup TensorBoard
    writer = None
    if use_tensorboard:
        writer = setup_tensorboard(output_dir)
        print(f"\nTensorBoard logging enabled: {output_dir / 'tensorboard'}")
        print(f"  View with: tensorboard --logdir={output_dir / 'tensorboard'}")

    # Compute class weights
    class_weights = None
    if use_class_weights:
        print("Computing class weights...")
        weights = compute_class_weights(patches_dir / "train", num_classes=num_classes)
        class_weights = torch.tensor(weights, device=device_t)
        print(f"Class weights: {weights}")

    # Data
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
    if sen2fire_dir is not None:
        from sen2fire_dataset import Sen2FireDataset
        from torch.utils.data import ConcatDataset
        sen2fire_train = Sen2FireDataset(sen2fire_dir, split="train", use_s2cloudless=False)
        sen2fire_val = Sen2FireDataset(sen2fire_dir, split="val", use_s2cloudless=False)
        combined_train = ConcatDataset([data_module.train_dataset, sen2fire_train])
        combined_val = ConcatDataset([data_module.val_dataset, sen2fire_val])
        train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(combined_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"Combined dataset: {len(combined_train)} train, {len(combined_val)} val (CEMS + Sen2Fire)")
    else:
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        print(f"Training samples: {len(data_module.train_dataset)}")
        print(f"Validation samples: {len(data_module.val_dataset)}")

    # Model (UNet from scratch)
    print("\nCreating scratch U-Net model...")
    model = UNet(in_channels=get_patch_num_channels(patches_dir), num_classes=num_classes, retainDim=True).to(device_t)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss
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

    # Optim + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Metrics
    train_metrics = CombinedMetrics(num_classes=num_classes, class_names=class_names)
    val_metrics = CombinedMetrics(num_classes=num_classes, class_names=class_names)

    # Training loop
    print("\n" + "=" * 60)
    print("  STARTING TRAINING")
    print("=" * 60 + "\n")

    best_metric = 0.0
    best_epoch = -1
    best_val_results = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_results = train_epoch(
            model, train_loader, criterion, optimizer, device_t, epoch, train_metrics,
            max_batches=max_batches,
        )
        val_results = validate_epoch(
            model, val_loader, criterion, device_t, epoch, val_metrics,
            max_batches=max_batches,
        )

        scheduler.step(val_results["fire_iou"])

        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_results['loss']:.4f}")
        print(f"  Val Loss: {val_results['loss']:.4f}")
        print(f"  Val Fire IoU: {val_results['fire_iou']:.4f}")
        print(f"  Val Fire Dice: {val_results['fire_dice']:.4f}")
        print(f"  Val Fire Precision: {val_results['fire_precision']:.4f}")
        print(f"  Val Fire Recall: {val_results['fire_recall']:.4f}")
        print(f"  Val Mean IoU: {val_results['mean_iou']:.4f}")
        print(f"  Val Detection F1: {val_results['detection_f1']:.4f}")

        metric_logger.log(epoch, "train", train_results)
        metric_logger.log(epoch, "val", val_results)

        if writer:
            writer.add_scalar("train/loss", train_results["loss"], epoch)
            writer.add_scalar("train/fire_iou", train_results["fire_iou"], epoch)
            writer.add_scalar("train/detection_f1", train_results["detection_f1"], epoch)

            writer.add_scalar("val/loss", val_results["loss"], epoch)
            writer.add_scalar("val/fire_iou", val_results["fire_iou"], epoch)
            writer.add_scalar("val/fire_dice", val_results["fire_dice"], epoch)
            writer.add_scalar("val/fire_precision", val_results["fire_precision"], epoch)
            writer.add_scalar("val/fire_recall", val_results["fire_recall"], epoch)
            writer.add_scalar("val/mean_iou", val_results["mean_iou"], epoch)
            writer.add_scalar("val/detection_f1", val_results["detection_f1"], epoch)

            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            if "ce_loss" in train_results:
                writer.add_scalar("train/ce_loss", train_results["ce_loss"], epoch)
            if "dice_loss" in train_results:
                writer.add_scalar("train/dice_loss", train_results["dice_loss"], epoch)

        if wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_results["loss"],
                "train/fire_iou": train_results["fire_iou"],
                "train/detection_f1": train_results["detection_f1"],
                "val/loss": val_results["loss"],
                "val/fire_iou": val_results["fire_iou"],
                "val/fire_dice": val_results["fire_dice"],
                "val/fire_precision": val_results["fire_precision"],
                "val/fire_recall": val_results["fire_recall"],
                "val/mean_iou": val_results["mean_iou"],
                "val/detection_f1": val_results["detection_f1"],
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Save best model (by fire IoU)
        if val_results["fire_iou"] > best_metric:
            best_metric = float(val_results["fire_iou"])
            best_epoch = epoch
            best_metrics = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in val_results.items()
            }
            best_metrics["train_loss"] = float(train_results["loss"])
            best_metrics["train_fire_iou"] = float(train_results["fire_iou"])
            best_metrics["train_detection_f1"] = float(train_results["detection_f1"])
            best_metrics["lr"] = float(optimizer.param_groups[0]["lr"])

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_results,
                path=checkpoints_dir / "best_model.pt",
                config=config,
            )
            print(f"  ✓ New best scratch U-Net saved (fire_iou: {best_metric:.4f})")
        else:
            epochs_without_improvement += 1

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_results,
                path=checkpoints_dir / f"checkpoint_epoch_{epoch}.pt",
                config=config,
            )

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping: no improvement for {early_stopping_patience} epochs")
            break

    # Final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics=val_results,
        path=checkpoints_dir / "final_model.pt",
        config=config,
    )

    if writer:
        writer.flush()
        writer.close()

    if wandb:
        wandb.finish()

    train_time_s = time.perf_counter() - t0

    print_run_summary(
        title="SCRATCH U-NET",
        train_time_s=train_time_s,
        best_checkpoint=str(best_ckpt_path),
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        num_params=total_params,
    )

    if results_csv is not None:
        append_results_csv(results_csv, {
            "dataset": dataset,
            "model_name": "unet_scratch",
            "wandb_run_name": wandb_run_name if wandb_project else "-",
            "FIRE_IOU": best_metrics.get("fire_iou", ""),
            "val_loss": best_metrics.get("loss", ""),
            "val_precision": best_metrics.get("fire_precision", ""),
            "val_recall": best_metrics.get("fire_recall", ""),
            "fire_dice": best_metrics.get("fire_dice", ""),
            "mean_iou": best_metrics.get("mean_iou", ""),
            "train_time_s": round(train_time_s, 2),
            "num_params": total_params,
            "best_epoch": best_epoch,
            "command": command,
        })

    return best_metric, best_epoch


# =============================================================================
# SMP SEGMENTATION TRAINING
# =============================================================================

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
    use_tensorboard: bool = True,
    early_stopping_patience: int = 10,
    save_every: int = 5,
    overwrite_output_dir: bool = False,
    use_dual_head: bool = False,
    max_batches: int | None = None,
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
        use_tensorboard: Enable TensorBoard logging
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
    metric_logger = MetricLogger(output_dir=output_dir)


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

    # Infer input channels from patch data (7 or 8) so model matches your patches
    in_channels = get_patch_num_channels(patches_dir)
    print(f"  Input channels (from patches): {in_channels}\n")

    # Class names for metrics (use shared constants)
    class_names = list(get_class_names(2 if use_dual_head else num_classes))

    # Config for logging
    config = {
        "patches_dir": str(patches_dir),
        "num_classes": num_classes,
        "in_channels": in_channels,
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

    # Setup TensorBoard
    writer = None
    if use_tensorboard:
        writer = setup_tensorboard(output_dir)
        print(f"\nTensorBoard logging enabled: {output_dir / 'tensorboard'}")
        print(f"  View with: tensorboard --logdir={output_dir / 'tensorboard'}")

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
            in_channels=in_channels,
            encoder_weights="imagenet",
            architecture=architecture,
        )
        criterion_severity = CombinedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=class_weights)
    else:
        model = FireSegmentationModel(
            encoder_name=encoder_name,
            num_classes=num_classes,
            in_channels=in_channels,
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

    # Dual-head: binary head uses 2 classes, so criterion must have no weights or 2-class weights
    if use_dual_head:
        criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=None)

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
            max_batches=max_batches,
        )

        # Validate
        val_results = validate_epoch(
            model, val_loader, criterion, device, epoch, val_metrics,
            criterion_severity=criterion_severity,
            dual_head=use_dual_head,
            max_batches=max_batches,
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

        metric_logger.log(epoch, "train", train_results)
        metric_logger.log(epoch, "val", val_results)

        # TensorBoard logging
        if writer:
            # Training metrics
            writer.add_scalar("train/loss", train_results["loss"], epoch)
            writer.add_scalar("train/fire_iou", train_results["fire_iou"], epoch)
            writer.add_scalar("train/detection_f1", train_results["detection_f1"], epoch)
            
            # Validation metrics
            writer.add_scalar("val/loss", val_results["loss"], epoch)
            writer.add_scalar("val/fire_iou", val_results["fire_iou"], epoch)
            writer.add_scalar("val/fire_recall", val_results["fire_recall"], epoch)
            writer.add_scalar("val/detection_f1", val_results["detection_f1"], epoch)
            
            # Learning rate
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            
            # Loss components if available
            if "ce_loss" in train_results:
                writer.add_scalar("train/ce_loss", train_results["ce_loss"], epoch)
            if "dice_loss" in train_results:
                writer.add_scalar("train/dice_loss", train_results["dice_loss"], epoch)

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

    if writer:
        writer.flush()
        writer.close()

    if wandb:
        wandb.finish()

    return best_metric


# =============================================================================
# RAY TUNE TRAINABLES
# =============================================================================

def tune_trainable(config, fixed):
    """
    Minimal Ray Tune trainable: call existing train() once per trial
    and report final best_metric.
    """
    best_metric = train(
        patches_dir=fixed["patches_dir"],
        output_dir=fixed["output_dir"] / f"trial_{uuid.uuid4().hex[:8]}",
        num_classes=fixed["num_classes"],
        encoder_name=fixed["encoder_name"],
        architecture=fixed["architecture"],
        batch_size=config.get("batch_size", fixed["batch_size"]),
        num_epochs=fixed["num_epochs"],
        learning_rate=config.get("learning_rate", fixed["learning_rate"]),
        weight_decay=config.get("weight_decay", fixed["weight_decay"]),
        use_class_weights=fixed["use_class_weights"],
        use_focal_loss=fixed["use_focal_loss"],
        focal_gamma=config.get("focal_gamma", fixed["focal_gamma"]),
        use_weighted_sampling=fixed["use_weighted_sampling"],
        fire_sample_weight=fixed["fire_sample_weight"],
        use_fire_augment=fixed["use_fire_augment"],
        num_workers=fixed["num_workers"],
        device=fixed["device"],
        resume=None,
        wandb_project=None,
        wandb_run_name=None,
        early_stopping_patience=fixed["early_stopping_patience"],
        save_every=fixed["save_every"],
        max_batches=fixed.get("max_batches"),
    )

    ray_report({"fire_iou": best_metric})


def tune_scratch_trainable(config, fixed):
    """
    Ray Tune trainable for ScratchFireModel.
    Reports val_loss (lower is better).
    """
    trial_dir = fixed["output_dir"] / f"trial_{uuid.uuid4().hex[:8]}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1, best_epoch = train_scratch_classifier(
        patches_dir=fixed["patches_dir"],
        sen2fire_dir=fixed.get("sen2fire_dir"),
        output_dir=trial_dir,
        batch_size=config.get("batch_size", fixed["batch_size"]),
        num_epochs=fixed["num_epochs"],
        learning_rate=config.get("learning_rate", fixed["learning_rate"]),
        weight_decay=config.get("weight_decay", fixed["weight_decay"]),
        num_workers=fixed["num_workers"],
        device=fixed["device"],
        dropout=config.get("dropout", 0.3),
        pos_weight=config.get("pos_weight", None),
        report_to_tune=True,
        max_batches=fixed.get("max_batches"),
    )

    ray_report({"val_f1": best_val_f1, "best_epoch": best_epoch})


def tune_unet_scratch_trainable(config, fixed):
    """
    Ray Tune trainable for UNet from scratch (segmentation).
    Reports best fire_iou (higher is better).
    """
    trial_dir = fixed["output_dir"] / f"trial_{uuid.uuid4().hex[:8]}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    best_fire_iou, best_epoch = train_unet_scratch_segmentation(
        patches_dir=fixed["patches_dir"],
        sen2fire_dir=fixed.get("sen2fire_dir"),
        output_dir=trial_dir,
        num_classes=fixed["num_classes"],
        batch_size=config.get("batch_size", fixed["batch_size"]),
        num_epochs=fixed["num_epochs"],
        learning_rate=config.get("learning_rate", fixed["learning_rate"]),
        weight_decay=config.get("weight_decay", fixed["weight_decay"]),
        use_class_weights=fixed["use_class_weights"],
        use_focal_loss=fixed["use_focal_loss"],
        focal_gamma=config.get("focal_gamma", fixed["focal_gamma"]),
        num_workers=fixed["num_workers"],
        device=fixed["device"],
        overwrite_output_dir=True,
        max_batches=fixed.get("max_batches"),
    )

    ray_report({"fire_iou": best_fire_iou, "best_epoch": best_epoch})


def tune_yolo_trainable(config, fixed):
    """
    Ray Tune trainable for YOLOv8 detection.
    Reports val mAP@0.5 (higher is better).
    Dataset export is done once before tuning starts; each trial only runs training.
    """
    from yolo_runner import train_and_validate_yolo_det7, YoloDetTrainCfg

    trial_dir = fixed["output_dir"] / f"trial_{uuid.uuid4().hex[:8]}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg = YoloDetTrainCfg(
        imgsz=fixed["imgsz"],
        batch=config.get("batch", fixed["batch_size"]),
        epochs=fixed["num_epochs"],
        device=fixed["device"],
        model_weights=fixed["model_weights"],
        lr0=config.get("lr0", fixed["lr0"]),
        weight_decay=config.get("weight_decay", fixed["weight_decay"]),
    )

    metrics = train_and_validate_yolo_det7(
        data_yaml=fixed["data_yaml"],
        output_dir=trial_dir,
        cfg=cfg,
    )

    map50 = metrics["best_metrics"].get("metrics/mAP50(B)", 0.0)
    ray_report({"map50": map50})


# =============================================================================
# MAIN / ARGUMENT PARSING
# =============================================================================

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
        "--sen2fire-dir",
        type=Path,
        default=None,
        help="Optional Sen2Fire dataset root dir. If provided, combined with CEMS patches for scratch and unet_scratch models.",
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
        "--all-encoders",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Train all encoders instead of a single encoder",
)
    parser.add_argument(
        "--architecture",
        type=str,
        default="unet",
        choices=["unet", "unetplusplus", "deeplabv3plus"],
        help="Segmentation architecture",
    )

    # YOLO detection baseline arguments
    parser.add_argument(
        "--include-yolo",
        action="store_true",
        help="Also train YOLOv8 DETECTION baseline using 7-channel multispectral data",
)

    # Scratch model arguments
    parser.add_argument(
        "--include-scratch",
        action="store_true",
        help="Also train the ScratchFireModel (binary classifier) using 7-channel patches",
    )

    # U-Net scratch model arguments
    parser.add_argument(
        "--include-unet-scratch",
        action="store_true",
        help="Also train The U-Net from scracth model. This one does not use pretrained encoders",
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
    parser.add_argument("--results-csv", type=Path, default=Path("training_results.csv"),
                        help="Path to CSV file where training results are appended (default: training_results.csv)")
    parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable TensorBoard logging (default: enabled)")
    parser.add_argument("--no-tensorboard", action="store_false", dest="tensorboard", help="Disable TensorBoard logging")

    # Hyperparameter tuning arguments
    parser.add_argument(
        "--tune",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Run hyperparameter tuning with Ray Tune",
    )
    parser.add_argument(
        "--tune-samples",
        type=int,
        default=20,
        help="Number of Ray Tune trials to run",
    )
    parser.add_argument(
        "--tune-mode",
        type=str,
        default="random",
        choices=["random", "grid"],
        help="Tuning strategy: random search or small grid",
    )
    parser.add_argument(
        "--tune-target",
        type=str,
        default="seg",
        choices=["seg", "scratch", "unet_scratch", "yolo"],
        help="What to tune: segmentation models (seg), CNN scratch (scratch), UNet scratch (unet_scratch), or YOLO (yolo)",
    )
    parser.add_argument(
        "--tune-top-k",
        type=int,
        default=3,
        help="Number of best trials to re-run with W&B and CSV logging after tuning (default: 3)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny smoke test: 1 epoch, 2 tune trials, top-1 rerun, 5 batches per epoch",
    )

    # YOLO-specific arguments
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=512,
        help="Image size for YOLO training (default: 512; use 256 to reduce memory on low-RAM machines)",
    )
    parser.add_argument(
        "--yolo-batch",
        type=int,
        default=None,
        help="Batch size override for YOLO (default: same as --batch-size; use smaller value to reduce memory)",
    )

    args = parser.parse_args()
    launch_command = "python train.py " + " ".join(sys.argv[1:])

    if args.smoke_test:
        print("[SMOKE TEST] Overriding: epochs=1, tune_samples=2, tune_top_k=1, max_batches=5")
        args.epochs = 1
        args.tune_samples = 2
        args.tune_top_k = 1
    train_all_encoders = args.all_encoders == "true"
    encoders_to_train = (
        ENCODER_OPTIONS if train_all_encoders else [args.encoder]
    )

    dataset_label = "1rst + 2nd dataset" if args.sen2fire_dir else "1rst dataset"
    sen2fire_tag = "+s2f" if args.sen2fire_dir else ""

    # Hyperparameter tuning
    if args.tune == "true":
        if tune is None:
            print("Ray Tune not available. Install with: pip install 'ray[tune]'")
            sys.exit(1)
        
        if args.tune_target == "scratch":
            search_space = {
                "learning_rate": tune.loguniform(5e-5, 5e-4),
                "weight_decay": tune.loguniform(1e-6, 1e-3),
                "dropout": tune.choice([0.1, 0.2, 0.3, 0.4]),
                # optional imbalance tuning (only useful if you know imbalance is large)
                # "pos_weight": tune.choice([1.0, 2.0, 5.0, 10.0]),
                # optional batch tuning:
                # "batch_size": tune.choice([8, 16, 32]),
            }

            fixed = {
                "patches_dir": Path(args.patches_dir).resolve(),
                "sen2fire_dir": Path(args.sen2fire_dir).resolve() if args.sen2fire_dir else None,
                "output_dir": (args.output_dir / "scratch_model" / "tune"),
                "batch_size": args.batch_size,
                "num_epochs": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "num_workers": args.num_workers,
                "device": args.device,
                "max_batches": 5 if args.smoke_test else None,
            }

            fixed["output_dir"].mkdir(parents=True, exist_ok=True)

            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(tune_scratch_trainable, fixed=fixed),
                    resources={"gpu": 1, "cpu": 4},
                ),
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    metric="val_f1",
                    mode="max",
                    num_samples=args.tune_samples if args.tune_mode == "random" else 1,
                    max_concurrent_trials=1,
                ),
                run_config=RunConfig(
                    name="tune_scratch",
                    storage_path=str(Path(fixed["output_dir"]).resolve()),

                ),
            )

            results = tuner.fit()

            # Re-run top-K best configs with W&B and CSV logging
            valid_results = [r for r in results if r.config is not None]
            if not valid_results:
                print("All tuning trials failed. Check error logs in the output directory.")
                return
            top_k = sorted(valid_results, key=lambda r: -r.metrics.get("val_f1", 0.0))[:args.tune_top_k]
            wandb_project = args.project if args.wandb else None

            for rank, result in enumerate(top_k, start=1):
                cfg = result.config
                best_epoch = int(result.metrics.get("best_epoch", args.epochs - 1))
                rerun_epochs = max(1, best_epoch + 1)  # epochs are 0-indexed
                run_name = f"scratch{sen2fire_tag}-tune-top{rank}-lr{cfg['learning_rate']:.1e}-wd{cfg['weight_decay']:.1e}-do{cfg.get('dropout', 0.3):.2f}"
                print(f"\n[Re-run {rank}/{args.tune_top_k}] scratch best config: {cfg} (epochs={rerun_epochs})")
                train_scratch_classifier(
                    patches_dir=args.patches_dir,
                    sen2fire_dir=args.sen2fire_dir,
                    output_dir=args.output_dir / "scratch_model" / f"best_{rank}",
                    batch_size=cfg.get("batch_size", args.batch_size),
                    num_epochs=rerun_epochs,
                    learning_rate=cfg["learning_rate"],
                    weight_decay=cfg["weight_decay"],
                    num_workers=args.num_workers,
                    device=args.device,
                    dropout=cfg.get("dropout", 0.3),
                    pos_weight=cfg.get("pos_weight", None),
                    wandb_project=wandb_project,
                    wandb_run_name=run_name,
                    results_csv=args.results_csv,
                    dataset=dataset_label,
                    max_batches=5 if args.smoke_test else None,
                    command=launch_command,
                )
            return

        if args.tune_target == "unet_scratch":
            # Search space narrowed based on best configs from 1rst dataset tuning:
            # best lr: 1.2e-4 to 2.4e-4, best batch: 16 or 32, weight_decay low to moderate
            search_space = {
                "learning_rate": tune.loguniform(1e-4, 3e-4),
                "weight_decay": tune.loguniform(1e-5, 3e-4),
                "batch_size": tune.choice([16, 32]),
            }
            fixed = {
                "patches_dir": Path(args.patches_dir).resolve(),
                "sen2fire_dir": Path(args.sen2fire_dir).resolve() if args.sen2fire_dir else None,
                "output_dir": (args.output_dir / "unet_scratch" / "tune"),
                "num_classes": args.num_classes,
                "batch_size": args.batch_size,
                "num_epochs": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "use_class_weights": not args.no_class_weights,
                "use_focal_loss": args.focal_loss,
                "focal_gamma": args.focal_gamma,
                "num_workers": args.num_workers,
                "device": args.device,
                "max_batches": 5 if args.smoke_test else None,
            }

            fixed["output_dir"].mkdir(parents=True, exist_ok=True)

            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(tune_unet_scratch_trainable, fixed=fixed),
                    resources={"gpu": 1, "cpu": 4},
                ),
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    metric="fire_iou",
                    mode="max",
                    num_samples=args.tune_samples if args.tune_mode == "random" else 1,
                    max_concurrent_trials=1,
                ),
                run_config=RunConfig(
                    name="tune_unet_scratch",
                    storage_path=str(Path(fixed["output_dir"]).resolve()),

                ),
            )

            results = tuner.fit()

            # Re-run top-K best configs with W&B and CSV logging
            valid_results = [r for r in results if r.config is not None]
            if not valid_results:
                print("All tuning trials failed. Check error logs in the output directory.")
                return
            top_k = sorted(valid_results, key=lambda r: -r.metrics.get("fire_iou", 0.0))[:args.tune_top_k]
            wandb_project = args.project if args.wandb else None

            for rank, result in enumerate(top_k, start=1):
                cfg = result.config
                best_epoch = int(result.metrics.get("best_epoch", args.epochs - 1))
                rerun_epochs = max(1, best_epoch + 1)
                run_name = f"unet-scratch{sen2fire_tag}-tune-top{rank}-lr{cfg['learning_rate']:.1e}-wd{cfg['weight_decay']:.1e}-bs{cfg.get('batch_size', args.batch_size)}"
                print(f"\n[Re-run {rank}/{args.tune_top_k}] unet_scratch best config: {cfg} (epochs={rerun_epochs})")
                train_unet_scratch_segmentation(
                    patches_dir=args.patches_dir,
                    sen2fire_dir=args.sen2fire_dir,
                    output_dir=args.output_dir / "unet_scratch" / f"best_{rank}",
                    num_classes=args.num_classes,
                    batch_size=cfg.get("batch_size", args.batch_size),
                    num_epochs=rerun_epochs,
                    learning_rate=cfg["learning_rate"],
                    weight_decay=cfg["weight_decay"],
                    use_class_weights=not args.no_class_weights,
                    use_focal_loss=args.focal_loss,
                    focal_gamma=args.focal_gamma,
                    num_workers=args.num_workers,
                    device=args.device,
                    wandb_project=wandb_project,
                    wandb_run_name=run_name,
                    results_csv=args.results_csv,
                    dataset=dataset_label,
                    overwrite_output_dir=True,
                    max_batches=5 if args.smoke_test else None,
                    command=launch_command,
                )
            return

        if args.tune_target == "yolo":
            from yolo_dataset_exporter import export_yolo_det7_dataset, ExportDet7Cfg
            from yolo_runner import YoloDetTrainCfg

            # Export dataset once; all trials reuse it
            yolo_channels = get_patch_num_channels(args.patches_dir)
            yolo_export_dir = args.output_dir / "yolo" / "tune"
            yolo_export_dir.mkdir(parents=True, exist_ok=True)
            data_yaml = yolo_export_dir / "yolo_det_8ch_dataset" / "data.yaml"

            if not data_yaml.exists():
                print("Exporting YOLO dataset (done once for all trials)...")
                export_yolo_det7_dataset(
                    patches_dir=args.patches_dir,
                    export_root=yolo_export_dir,
                    num_classes=args.num_classes,
                    export_cfg=ExportDet7Cfg(
                        channels=yolo_channels,
                        mask_to_boxes_mode="components",
                        min_box_area_px=10,
                    ),
                    train_cfg=YoloDetTrainCfg(
                        imgsz=args.yolo_imgsz,
                        batch=args.yolo_batch or args.batch_size,
                        epochs=1,  # dummy run just to trigger export
                        device=args.device,
                    ),
                    num_workers=args.num_workers,
                    sen2fire_dir=args.sen2fire_dir,
                )

            yolo_batch_size = args.yolo_batch or args.batch_size
            search_space = {
                "lr0": tune.loguniform(5e-4, 1e-2),
                "weight_decay": tune.loguniform(1e-4, 1e-2),
                "batch": tune.choice([max(4, yolo_batch_size // 2), yolo_batch_size]),
            }

            fixed = {
                "patches_dir": Path(args.patches_dir).resolve(),
                "output_dir": yolo_export_dir.resolve(),
                "data_yaml": data_yaml.resolve(),
                "num_epochs": args.epochs,
                "batch_size": yolo_batch_size,
                "imgsz": args.yolo_imgsz,
                "device": args.device,
                "model_weights": "yolov8n.pt",
                "lr0": 1e-2,
                "weight_decay": 5e-4,
            }

            # Ultralytics reports its own per-epoch metrics to Ray Tune internally,
            # which don't include our 'map50' key. Disable strict checking to allow this.
            os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(tune_yolo_trainable, fixed=fixed),
                    resources={"gpu": 1, "cpu": 4},
                ),
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    metric="map50",
                    mode="max",
                    num_samples=args.tune_samples if args.tune_mode == "random" else 1,
                    max_concurrent_trials=1,
                ),
                run_config=RunConfig(
                    name="tune_yolo",
                    storage_path=str(Path(yolo_export_dir).resolve()),

                ),
            )

            results = tuner.fit()

            # Re-run top-K best configs with W&B and CSV logging
            from yolo_runner import train_and_validate_yolo_det7, YoloDetTrainCfg
            valid_results = [r for r in results if r.config is not None]
            if not valid_results:
                print("All tuning trials failed. Check error logs in the output directory.")
                return
            top_k = sorted(valid_results, key=lambda r: -r.metrics.get("map50", 0.0))[:args.tune_top_k]
            wandb_project = args.project if args.wandb else None

            for rank, result in enumerate(top_k, start=1):
                cfg = result.config
                run_name = f"yolo{sen2fire_tag}-tune-top{rank}-lr{cfg['lr0']:.1e}-wd{cfg['weight_decay']:.1e}-bs{cfg.get('batch', args.batch_size)}"
                print(f"\n[Re-run {rank}/{args.tune_top_k}] yolo best config: {cfg}")

                yolo_cfg = YoloDetTrainCfg(
                    imgsz=args.yolo_imgsz,
                    batch=cfg.get("batch", yolo_batch_size),
                    epochs=args.epochs,
                    device=args.device,
                    model_weights="yolov8n.pt",
                    lr0=cfg["lr0"],
                    weight_decay=cfg["weight_decay"],
                )

                wandb_run = None
                if wandb_project:
                    wandb_run = setup_wandb(
                        config={**cfg, "model": "yolo", "epochs": args.epochs},
                        project=wandb_project,
                        run_name=run_name,
                        wandb_dir=args.output_dir / "yolo" / f"best_{rank}",
                    )

                metrics = train_and_validate_yolo_det7(
                    data_yaml=data_yaml,
                    output_dir=args.output_dir / "yolo" / f"best_{rank}",
                    cfg=yolo_cfg,
                )

                yolo_m = dict(metrics["best_metrics"])

                if wandb_run:
                    wandb_run.log({
                        "map50": yolo_m.get("metrics/mAP50(B)", 0.0),
                        "map50_95": yolo_m.get("metrics/mAP50-95(B)", 0.0),
                        "precision": yolo_m.get("metrics/precision(B)", 0.0),
                        "recall": yolo_m.get("metrics/recall(B)", 0.0),
                        "f1": yolo_m.get("metrics/f1(B)", 0.0),
                        "train_time_s": metrics["train_time_s"],
                    })
                    wandb_run.finish()

                if args.results_csv is not None:
                    append_results_csv(args.results_csv, {
                        "model_name": "yolo",
                        "wandb_run_name": run_name if wandb_project else "-",
                        "MAP_50_95": yolo_m.get("metrics/mAP50-95(B)", ""),
                        "map_50": yolo_m.get("metrics/mAP50(B)", ""),
                        "val_precision": yolo_m.get("metrics/precision(B)", ""),
                        "val_recall": yolo_m.get("metrics/recall(B)", ""),
                        "f1": yolo_m.get("metrics/f1(B)", ""),
                        "train_time_s": round(float(metrics["train_time_s"]), 2),
                        "num_params": int(metrics["num_params"]),
                        "command": launch_command,
                    })
            return

        # Segmentation tuning
        if args.tune_mode == "grid":
            search_space = {
                "learning_rate": tune.grid_search([5e-5, 1e-4, 2e-4]),
                "weight_decay": tune.grid_search([1e-5, 1e-4]),
            }
        else:
            search_space = {
                "learning_rate": tune.loguniform(5e-5, 5e-4),
                "weight_decay": tune.loguniform(1e-6, 1e-3),
            }
            if args.focal_loss:
                search_space["focal_gamma"] = tune.choice([1.5, 2.0, 2.5])

        # We'll store best results per encoder here
        best_per_encoder = {}

        # If --all-encoders true -> tune each encoder separately
        tune_encoders = ENCODER_OPTIONS if train_all_encoders else [args.encoder]

        for encoder in tune_encoders:
            fixed = {
                "patches_dir": Path(args.patches_dir).resolve(),
                "output_dir": (args.output_dir / f"encoder_{encoder}" / "tune"),
                "num_classes": args.num_classes,
                "encoder_name": encoder,
                "architecture": args.architecture,
                "batch_size": args.batch_size,
                "num_epochs": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "use_class_weights": not args.no_class_weights,
                "use_focal_loss": args.focal_loss,
                "focal_gamma": args.focal_gamma,
                "use_weighted_sampling": args.weighted_sampling,
                "fire_sample_weight": args.fire_weight,
                "use_fire_augment": not args.no_fire_augment,
                "num_workers": args.num_workers,
                "device": args.device,
                "early_stopping_patience": args.patience,
                "save_every": args.save_every,
                "max_batches": 5 if args.smoke_test else None,
            }

            fixed["output_dir"].mkdir(parents=True, exist_ok=True)

            print(f"\n\n{'#' * 80}")
            print(f"TUNING ENCODER: {encoder}")
            print(f"OUTPUT DIR: {fixed['output_dir']}")
            print(f"{'#' * 80}\n")

            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(tune_trainable, fixed=fixed),
                    resources={"gpu": 1, "cpu": 4},
                ),
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    metric="fire_iou",
                    mode="max",
                    num_samples=args.tune_samples if args.tune_mode == "random" else 1,
                    max_concurrent_trials=1,
                ),
                run_config=RunConfig(
                    name=f"tune_{encoder}",
                    storage_path=str(Path(fixed["output_dir"]).resolve()),

                ),
            )

            results = tuner.fit()
            best = results.get_best_result(metric="fire_iou", mode="max")

            best_per_encoder[encoder] = {
                "best_fire_iou": float(best.metrics["fire_iou"]),
                "best_config": best.config,
            }

            print("\nBest hyperparameters:", best.config)
            print("Best Fire IoU:", best.metrics["fire_iou"])

        # Save a global summary JSON
        summary_path = args.output_dir / "tune_encoder_summary.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(best_per_encoder, f, indent=2)

        print("\nSaved tuning summary to:", summary_path)
        return
    else:
        # Run training without Hyperparameter tuning
        wandb_project = args.project if args.wandb else None
        results = {}

    # Skip encoder training if ONLY special models (YOLO/scratch) are requested
    skip_encoders = (args.include_yolo or args.include_scratch or args.include_unet_scratch) and args.all_encoders == "false"
    
    # Runing bag of models
    # SMP models
    if not skip_encoders:
        for encoder in encoders_to_train:
            encoder_output_dir = args.output_dir / f"encoder_{encoder}"

            run_name = args.run_name or f"{encoder}-{args.architecture}-c{args.num_classes}"

            print(f"\n\n{'#' * 80}")
            print(f"TRAINING ENCODER: {encoder}")
            print(f"OUTPUT DIR: {encoder_output_dir}")
            print(f"{'#' * 80}\n")


            best_metric = train(
                patches_dir=args.patches_dir,
                output_dir=encoder_output_dir,
                num_classes=args.num_classes,
                encoder_name=encoder,
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
                resume=None if train_all_encoders else args.resume,
                wandb_project=wandb_project,
                wandb_run_name=run_name,
                early_stopping_patience=args.patience,
                save_every=args.save_every,
                use_dual_head=args.dual_head,
            )

            results[encoder] = best_metric

    # YOLOv8 DETECTION (7-CHANNEL) baseline
    if args.include_yolo:
        from yolo_dataset_exporter import export_yolo_det7_dataset, ExportDet7Cfg
        from yolo_runner import YoloDetTrainCfg
        best_epoch = 0  # default; YOLO doesn't track per-epoch best
        yolo_run_name = args.run_name or _make_run_name("yolo", args)

        yolo_channels = get_patch_num_channels(args.patches_dir)
        print("\n" + "#" * 80)
        print(f"TRAINING YOLOv8 DETECTION ({yolo_channels}-CHANNEL)")
        print("#" * 80 + "\n")

        metrics = export_yolo_det7_dataset(
            patches_dir=args.patches_dir,
            export_root=args.output_dir,
            num_classes=args.num_classes,
            export_cfg=ExportDet7Cfg(
                channels=yolo_channels,
                mask_to_boxes_mode="components",
                min_box_area_px=10,
            ),
            train_cfg=YoloDetTrainCfg(
                imgsz=args.yolo_imgsz,
                batch=args.yolo_batch or args.batch_size,
                epochs=args.epochs,
                device=args.device,
                model_weights="yolov8n.pt",
            ),
            num_workers=args.num_workers,
            sen2fire_dir=args.sen2fire_dir,
        )

        # Report block
        print_run_summary(
            title="YOLO DET",
            train_time_s=float(metrics["train_time_s"]),
            best_checkpoint=str(metrics["best_checkpoint"]),
            best_epoch=best_epoch,
            best_metrics=dict(metrics["best_metrics"]),
            num_params=int(metrics["num_params"]),
        )

        if args.results_csv is not None:
            yolo_m = dict(metrics["best_metrics"])
            append_results_csv(args.results_csv, {
                "dataset": dataset_label,
                "model_name": "yolo",
                "wandb_run_name": yolo_run_name if args.wandb else "-",
                "MAP_50_95": yolo_m.get("metrics/mAP50-95(B)", ""),
                "map_50": yolo_m.get("metrics/mAP50(B)", ""),
                "val_precision": yolo_m.get("metrics/precision(B)", ""),
                "val_recall": yolo_m.get("metrics/recall(B)", ""),
                "f1": yolo_m.get("metrics/f1(B)", ""),
                "train_time_s": round(float(metrics["train_time_s"]), 2),
                "num_params": int(metrics["num_params"]),
                "best_epoch": best_epoch,
                "command": launch_command,
            })
        

    # From Scratch model
    if args.include_scratch:
        print("\n" + "#" * 80)
        print("TRAINING SCRATCH MODEL (BINARY CLASSIFIER)")
        print("#" * 80 + "\n")
        run_name = args.run_name or _make_run_name("cnn_scratch", args)

        train_scratch_classifier(
            patches_dir=args.patches_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            device=args.device,
            wandb_project=wandb_project,
            wandb_run_name=run_name,
            results_csv=args.results_csv,
            command=launch_command,
        )

    # U-Net from scratch (segmentation) baseline
    if args.include_unet_scratch:
        print("\n" + "#" * 80)
        print("TRAINING U-NET FROM SCRATCH (SEGMENTATION BASELINE)")
        print("#" * 80 + "\n")

        unet_run_name = args.run_name or _make_run_name("unet_scratch", args)

        train_unet_scratch_segmentation(
            patches_dir=args.patches_dir,
            output_dir=args.output_dir,
            num_classes=args.num_classes,
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
            wandb_project=(args.project if args.wandb else None),
            wandb_run_name=unet_run_name,
            use_tensorboard=args.tensorboard,
            early_stopping_patience=args.patience,
            save_every=args.save_every,
            overwrite_output_dir=args.overwrite_output_dir,
            results_csv=args.results_csv,
            command=launch_command,
        )

if __name__ == "__main__":
    main()
