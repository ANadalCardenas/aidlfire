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
from model import FireSegmentationModel, CombinedLoss, ENCODER_OPTIONS
from metrics import CombinedMetrics
from constants import get_device, get_class_names
from yolo_runner import train_yolo_seg, YoloCfg

from ray import tune


def setup_wandb(config: dict, project: str, run_name: str | None = None):
    """Initialize Weights & Biases logging."""
    try:
        import wandb
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

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)

        # Compute loss - CombinedLoss returns (loss, dict), others return just loss
        result = criterion(logits, masks)
        if isinstance(result, tuple):
            loss, components = result
            for k, v in components.items():
                loss_components[k] += v
        else:
            loss = result

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        with torch.no_grad():
            metrics.update(logits, masks)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute epoch metrics
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

        logits = model(images)

        # Compute loss - CombinedLoss returns (loss, dict), others return just loss
        result = criterion(logits, masks)
        if isinstance(result, tuple):
            loss, _ = result
        else:
            loss = result

        metrics.update(logits, masks)
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute epoch metrics
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
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Setup device using shared utility
    device = get_device(device)

    print(f"\n{'='*60}")
    print(f"  FIRE DETECTION TRAINING")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Patches: {patches_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Classes: {num_classes}")
    print(f"  Encoder: {encoder_name}")
    print(f"  Architecture: {architecture}")
    print(f"{'='*60}\n")

    # Class names for metrics (use shared constants)
    class_names = list(get_class_names(num_classes))

    # Config for logging
    config = {
        "patches_dir": str(patches_dir),
        "num_classes": num_classes,
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

    # Setup W&B
    wandb = None
    if wandb_project:
        wandb = setup_wandb(config, wandb_project, wandb_run_name)

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
    model = FireSegmentationModel(
        encoder_name=encoder_name,
        num_classes=num_classes,
        in_channels=7,
        encoder_weights="imagenet",
        architecture=architecture,
    )
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
        verbose=True,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_metric = 0
    if resume:
        print(f"\nResuming from: {resume}")
        checkpoint = load_checkpoint(resume, model, optimizer)
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["metrics"].get("fire_iou", 0)
        print(f"Resuming from epoch {start_epoch}, best fire_iou: {best_metric:.4f}")

    # Setup metrics
    train_metrics = CombinedMetrics(num_classes=num_classes, class_names=class_names)
    val_metrics = CombinedMetrics(num_classes=num_classes, class_names=class_names)

    # Training loop
    print("\n" + "=" * 60)
    print("  STARTING TRAINING")
    print("=" * 60 + "\n")

    epochs_without_improvement = 0

    for epoch in range(start_epoch, num_epochs):
        # Train
        train_results = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, train_metrics
        )

        # Validate
        val_results = validate_epoch(
            model, val_loader, criterion, device, epoch, val_metrics
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


def tune_trainable(config, fixed):
    """
    Minimal Ray Tune trainable: call existing train() once per trial
    and report final best_metric.
    """
    best_metric = train(
        patches_dir=fixed["patches_dir"],
        output_dir=fixed["output_dir"] / f"trial_{tune.get_trial_id()}",
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
    )

    tune.report(fire_iou=best_metric)


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
        default=Path("./output"),
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        choices=[2, 5],
        help="Number of classes (2=DEL binary, 5=GRA severity)",
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

    # YOLO baseline arguments
    parser.add_argument(
        "--include-yolo",
        action="store_true",
        help="Also train YOLOv8-Seg RGB baseline after main training",
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

    args = parser.parse_args()
    train_all_encoders = args.all_encoders == "true"
    encoders_to_train = (
        ENCODER_OPTIONS if train_all_encoders else [args.encoder]
    )

    # Hyperparameter tuning
    if args.tune == "true":
        # Build the search space once
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
                "patches_dir": args.patches_dir,
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
            }

            fixed["output_dir"].mkdir(parents=True, exist_ok=True)

            print(f"\n\n{'#' * 80}")
            print(f"TUNING ENCODER: {encoder}")
            print(f"OUTPUT DIR: {fixed['output_dir']}")
            print(f"{'#' * 80}\n")

            tuner = tune.Tuner(
                tune.with_parameters(tune_trainable, fixed=fixed),
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    metric="fire_iou",
                    mode="max",
                    num_samples=args.tune_samples if args.tune_mode == "random" else 1,
                ),
                run_config=tune.RunConfig(
                    name=f"tune_{encoder}",
                    local_dir=str(fixed["output_dir"]),
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

    # Runing bag of models
    # SMP models

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
        )

        results[encoder] = best_metric

    # YOLOv8-Seg RGB BASELINE
    if args.include_yolo:
        from yolo_runner import train_yolo_seg, YoloCfg

        print("\n" + "#" * 80)
        print("TRAINING YOLOv8-SEG RGB BASELINE")
        print("#" * 80 + "\n")

        yolo_metrics = train_yolo_seg(
            patches_dir=args.patches_dir,
            output_dir=args.output_dir / "yolo_seg",
            num_classes=args.num_classes,
            cfg=YoloCfg(
                epochs=args.epochs,
                batch=args.batch_size,
                device=args.yolo_device,
                rgb_channels=(0, 1, 2),  # update if needed
            ),
            num_workers=args.num_workers,
        )

        print("YOLO validation metrics:", yolo_metrics.get("val_results"))


if __name__ == "__main__":
    main()
