#!/usr/bin/env python3
"""
Train all model combinations (encoder + architecture + baselines) without overwriting existing outputs.

This script trains all combinations of:
- Segmentation Models:
  - Encoders: resnet18, resnet34, resnet50, efficientnet-b0, efficientnet-b1, mobilenet_v2
  - Architectures: unet, unetplusplus, deeplabv3plus
- Baseline Models:
  - YOLOv8 (optional, detection/segmentation)
  - Scratch Model (optional, binary classifier)
  - Scratch U-Net Model (optional, segmentation)

Each training run uses a unique output directory with timestamps to prevent overwriting.

Usage:
    uv run python train_all_models.py --patches-dir ./patches --num-classes 2
    uv run python train_all_models.py --patches-dir ./patches --num-classes 2 --include-yolo --include-scratch --include-unet-scratch
    uv run python train_all_models.py --patches-dir ./patches --wandb --project fire-detection

Output:
    ./output/training_run_{timestamp}/
        encoder_{encoder}-architecture_{arch}/
            checkpoints/
            logs/
            metrics/
        yolo_seg/
            checkpoints/
            logs/
        scratch_model/
            checkpoints/
            logs/
        scratch_unet_model/
            checkpoints/
            logs/
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


ENCODERS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnet-b0",
    "efficientnet-b1",
    "mobilenet_v2",
]

ARCHITECTURES = ["unet", "unetplusplus", "deeplabv3plus"]


def train_segmentation_model(
    patches_dir: Path,
    output_dir: Path,
    encoder: str,
    architecture: str,
    num_classes: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    use_class_weights: bool,
    use_focal_loss: bool,
    use_weighted_sampling: bool,
    use_fire_augment: bool,
    num_workers: int,
    device: str,
    wandb_enabled: bool,
    wandb_project: str,
    verbose: bool = True,
) -> bool:
    """
    Train a single segmentation model configuration.
    
    Returns:
        True if training succeeded, False otherwise
    """
    # Build command
    cmd = [
        sys.executable,
        "train.py",
        "--patches-dir", str(patches_dir),
        "--output-dir", str(output_dir),
        "--encoder", encoder,
        "--architecture", architecture,
        "--num-classes", str(num_classes),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(learning_rate),
        "--weight-decay", str(weight_decay),
        "--num-workers", str(num_workers),
        "--device", device,
    ]
    
    if use_class_weights:
        pass  # Enabled by default
    else:
        cmd.append("--no-class-weights")
    
    if use_focal_loss:
        cmd.append("--focal-loss")
    
    if use_weighted_sampling:
        cmd.append("--weighted-sampling")
    
    if not use_fire_augment:
        cmd.append("--no-fire-augment")
    
    if wandb_enabled:
        cmd.append("--wandb")
        cmd.extend(["--project", wandb_project])
        cmd.extend(["--run-name", f"{encoder}-{architecture}"])
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Training Segmentation: {encoder} + {architecture}")
        print(f"Output: {output_dir}")
        print(f"{'='*80}\n")
    
    # Run training
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for {encoder} + {architecture}: {e}")
        return False


def train_yolo_baseline(
    patches_dir: Path,
    output_dir: Path,
    num_classes: int,
    batch_size: int,
    epochs: int,
    num_workers: int,
    device: str,
    wandb_enabled: bool,
    wandb_project: str,
    verbose: bool = True,
) -> bool:
    """
    Train YOLOv8 baseline model ONLY (no encoder training).
    
    Returns:
        True if training succeeded, False otherwise
    """
    # Build command - MUST specify encoder to prevent training all encoders
    # We use resnet18 as dummy since we only want YOLO, not segmentation
    cmd = [
        sys.executable,
        "train.py",
        "--patches-dir", str(patches_dir),
        "--output-dir", str(output_dir),
        "--num-classes", str(num_classes),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--num-workers", str(num_workers),
        "--device", device,
        "--encoder", "resnet18",  # Required to prevent all-encoder training
        "--include-yolo",
    ]
    
    if wandb_enabled:
        cmd.append("--wandb")
        cmd.extend(["--project", wandb_project])
        cmd.extend(["--run-name", "yolo-baseline"])
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Training YOLOv8 Baseline")
        print(f"Output: {output_dir}")
        print(f"{'='*80}\n")
    
    # Run training
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for YOLOv8 baseline: {e}")
        return False


def train_scratch_model(
    patches_dir: Path,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    device: str,
    wandb_enabled: bool,
    wandb_project: str,
    verbose: bool = True,
) -> bool:
    """
    Train scratch model (binary classifier) ONLY (no encoder training).
    
    Returns:
        True if training succeeded, False otherwise
    """
    # Build command - MUST specify encoder to prevent training all encoders
    cmd = [
        sys.executable,
        "train.py",
        "--patches-dir", str(patches_dir),
        "--output-dir", str(output_dir),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(learning_rate),
        "--weight-decay", str(weight_decay),
        "--num-workers", str(num_workers),
        "--device", device,
        "--encoder", "resnet18",  # Required to prevent all-encoder training
        "--include-scratch",
    ]
    
    if wandb_enabled:
        cmd.append("--wandb")
        cmd.extend(["--project", wandb_project])
        cmd.extend(["--run-name", f"scratch-classifier-lr-{learning_rate}-wd-{weight_decay}"])
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Training Scratch Model (Binary Classifier)")
        print(f"Output: {output_dir}")
        print(f"{'='*80}\n")
    
    # Run training
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for Scratch model: {e}")
        return False

def train_unet_scratch_model(
    patches_dir: Path,
    output_dir: Path,
    num_classes: int,
    batch_size: int,
    epochs: int,
    num_workers: int,
    device: str,
    wandb_enabled: bool,
    wandb_project: str,
    verbose: bool = True,
) -> bool:
    """
    Train U-Net scratch model (segmentation) ONLY (no encoder training).
    
    Returns:
        True if training succeeded, False otherwise
    """
    # Build command - MUST specify encoder to prevent training all encoders
    cmd = [
        sys.executable,
        "train.py",
        "--patches-dir", str(patches_dir),
        "--output-dir", str(output_dir),
        "--num-classes", str(num_classes),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--num-workers", str(num_workers),
        "--device", device,
        "--encoder", "resnet18",  # Required to prevent all-encoder training
        "--include-unet-scratch",
    ]
    
    if wandb_enabled:
        cmd.append("--wandb")
        cmd.extend(["--project", wandb_project])
        cmd.extend(["--run-name", "scratch-segmentation"])
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Training U-Net Scratch Model (Segmentation)")
        print(f"Output: {output_dir}")
        print(f"{'='*80}\n")
    
    # Run training
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for U-Net Scratch Model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train all model combinations (segmentation + baselines) without overwriting outputs",
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
        help="Base output directory (will create timestamped subdirectories)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        choices=[2, 5],
        help="Number of classes (2=DEL binary, 5=GRA severity)",
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
    parser.add_argument("--no-fire-augment", action="store_true", help="Disable fire-specific augmentation")
    
    # Sampling arguments
    parser.add_argument("--no-weighted-sampling", action="store_true", help="Disable weighted sampling")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    
    # Logging arguments
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--project", type=str, default="fire-detection", help="W&B project name")
    
    # Segmentation model filtering
    parser.add_argument(
        "--encoders",
        type=str,
        help="Comma-separated list of encoders to train (default: all)",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        help="Comma-separated list of architectures to train (default: all)",
    )
    
    # Baseline model arguments
    parser.add_argument(
        "--include-yolo",
        action="store_true",
        help="Include YOLOv8 baseline training",
    )
    parser.add_argument(
        "--include-scratch",
        action="store_true",
        help="Include scratch model (binary classifier) training",
    )
    parser.add_argument(
        "--include-unet-scratch",
        action="store_true",
        help="Include U-Net scratch model (segmentation) training",
    )
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Train only segmentation models (skip YOLO and scratches)",
    )
    
    # Control arguments
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue training other models if one fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    
    args = parser.parse_args()
    
    print("PATCHES ",args.patches_dir)
    # Determine which encoders/architectures to train
    encoders = ENCODERS
    architectures = ARCHITECTURES
    
    if args.encoders:
        encoders = args.encoders.split(",")
        encoders = [e.strip() for e in encoders]
    
    if args.architectures:
        architectures = args.architectures.split(",")
        architectures = [a.strip() for a in architectures]
    
    # Validate
    for encoder in encoders:
        if encoder not in ENCODERS:
            print(f"❌ Unknown encoder: {encoder}")
            sys.exit(1)
    
    for arch in architectures:
        if arch not in ARCHITECTURES:
            print(f"❌ Unknown architecture: {arch}")
            sys.exit(1)
    
    # Generate timestamp once for all runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate total combinations
    seg_combinations = len(encoders) * len(architectures)
    total_combinations = seg_combinations
    if args.include_yolo:
        total_combinations += 1
    if args.include_scratch:
        total_combinations += 1
    if args.include_unet_scratch:
        total_combinations += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Encoders: {', '.join(encoders)}")
    print(f"Architectures: {', '.join(architectures)}")
    print(f"Segmentation combinations: {seg_combinations}")
    if args.include_yolo:
        print(f"YOLOv8 baseline: YES")
    if args.include_scratch:
        print(f"Scratch model: YES")
    if args.include_unet_scratch:
        print(f"Scratch U-Net model: YES")    
    print(f"Total models: {total_combinations}")
    print(f"Patches dir: {args.patches_dir}")
    print(f"Output base dir: {args.output_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")
    print(f"W&B enabled: {args.wandb}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*80}\n")
    
    if args.dry_run:
        print("[DRY RUN] Commands will be printed but not executed\n")
    
    # Train all combinations
    results = {}
    failed = []
    
    # Train segmentation models
    print("SEGMENTATION MODELS")
    print("-" * 80)
    for encoder in encoders:
        for architecture in architectures:
            combo = f"seg-{encoder}-{architecture}"
            output_dir = args.output_dir / f"training_run_{timestamp}" / f"encoder_{encoder}-architecture_{architecture}"
            
            print(f"\n[{len(results)+1}/{total_combinations}] {combo}")
            
            if args.dry_run:
                cmd = [
                    "python train.py",
                    f"--encoder {encoder}",
                    f"--architecture {architecture}",
                    f"--output-dir {output_dir}",
                    f"--patches-dir {args.patches_dir}",
                    f"--num-classes {args.num_classes}",
                    f"--batch-size {args.batch_size}",
                    f"--epochs {args.epochs}",
                    f"--lr {args.lr}",
                ]
                print(f"Command: {' '.join(cmd)}\n")
                results[combo] = "dry-run"
            else:
                success = train_segmentation_model(
                    patches_dir=args.patches_dir,
                    output_dir=output_dir,
                    encoder=encoder,
                    architecture=architecture,
                    num_classes=args.num_classes,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                    use_class_weights=not args.no_class_weights,
                    use_focal_loss=args.focal_loss,
                    use_weighted_sampling=not args.no_weighted_sampling,
                    use_fire_augment=not args.no_fire_augment,
                    num_workers=args.num_workers,
                    device=args.device,
                    wandb_enabled=args.wandb,
                    wandb_project=args.project,
                )
                results[combo] = "✅ success" if success else "❌ failed"
                if not success:
                    failed.append(combo)
                    if not args.skip_errors:
                        print(f"\n⚠️  NOTICE: Training failed, but models may still be saved.")
                        print(f"     Use --skip-errors to continue training other models.")
                        print(f"     This saves compute resources if you want to retry later.\n")
                        sys.exit(1)
    
    # Train YOLOv8 baseline
    if args.include_yolo and not args.segmentation_only:
        print("\n\nYOLOV8 BASELINE")
        print("-" * 80)
        output_dir = args.output_dir / f"training_run_{timestamp}" / "yolo_seg"
        
        print(f"\n[{len(results)+1}/{total_combinations}] yolo-baseline")
        
        if args.dry_run:
            cmd = [
                "python train.py",
                f"--output-dir {output_dir}",
                f"--patches-dir {args.patches_dir}",
                f"--num-classes {args.num_classes}",
                f"--batch-size {args.batch_size}",
                f"--epochs {args.epochs}",
                "--include-yolo",
            ]
            print(f"Command: {' '.join(cmd)}\n")
            results["yolo-baseline"] = "dry-run"
        else:
            success = train_yolo_baseline(
                patches_dir=args.patches_dir,
                output_dir=output_dir,
                num_classes=args.num_classes,
                batch_size=args.batch_size,
                epochs=args.epochs,
                num_workers=args.num_workers,
                device=args.device,
                wandb_enabled=args.wandb,
                wandb_project=args.project,
            )
            results["yolo-baseline"] = "✅ success" if success else "❌ failed"
            if not success:
                failed.append("yolo-baseline")
                if not args.skip_errors:
                    print(f"❌ Training failed. Stopping. Use --skip-errors to continue.")
                    sys.exit(1)
    
    # Train scratch model
    if args.include_scratch and not args.segmentation_only:
        print("\n\nSCRATCH MODEL (BINARY CLASSIFIER)")
        print("-" * 80)
        output_dir = args.output_dir / f"training_run_{timestamp}" / "scratch_model"
        
        print(f"\n[{len(results)+1}/{total_combinations}] scratch-classifier")
        
        if args.dry_run:
            cmd = [
                "python train.py",
                f"--output-dir {output_dir}",
                f"--patches-dir {args.patches_dir}",
                f"--batch-size {args.batch_size}",
                f"--epochs {args.epochs}",
                "--include-scratch",
            ]
            print(f"Command: {' '.join(cmd)}\n")
            results["scratch-classifier"] = "dry-run"
        else:
            success = train_scratch_model(
                patches_dir=args.patches_dir,
                output_dir=output_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                device=args.device,
                wandb_enabled=args.wandb,
                wandb_project=args.project,
            )
            results["scratch-classifier"] = "✅ success" if success else "❌ failed"
            if not success:
                failed.append("scratch-classifier")
                if not args.skip_errors:
                    print(f"❌ Training failed. Stopping. Use --skip-errors to continue.")
                    sys.exit(1)

    # Train U-Net scratch model
    if args.include_unet_scratch and not args.segmentation_only:
        print("\n\nSCRATCH U-NET MODEL (SEGMENTATION)")
        print("-" * 80)
        output_dir = args.output_dir / f"training_run_{timestamp}" / "unet_scratch_model"
        
        print(f"\n[{len(results)+1}/{total_combinations}] scratch-segmentation")
        
        if args.dry_run:
            cmd = [
                "python train.py",
                f"--output-dir {output_dir}",
                f"--patches-dir {args.patches_dir}",
                f"--batch-size {args.batch_size}",
                f"--epochs {args.epochs}",
                "--include-unet-scratch",
            ]
            print(f"Command: {' '.join(cmd)}\n")
            results["scratch-segmentation"] = "dry-run"
        else:
            success = train_unet_scratch_model(
                patches_dir=args.patches_dir,
                output_dir=output_dir,
                num_classes=args.num_classes,
                batch_size=args.batch_size,
                epochs=args.epochs,
                num_workers=args.num_workers,
                device=args.device,
                wandb_enabled=args.wandb,
                wandb_project=args.project,
            )
            results["scratch-segmentation"] = "✅ success" if success else "❌ failed"
            if not success:
                failed.append("scratch-segmentation")
                if not args.skip_errors:
                    print(f"❌ Training failed. Stopping. Use --skip-errors to continue.")
                    sys.exit(1)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Training Summary")
    print(f"{'='*80}")
    for combo, status in results.items():
        print(f"{combo:40} {status}")
    print(f"{'='*80}\n")
    
    if failed:
        print(f"[FAILED] {len(failed)} training(s) failed:")
        for combo in failed:
            print(f"   - {combo}")
        sys.exit(1)
    else:
        print(f"[SUCCESS] All trainings completed successfully!")
        print(f"Results saved to: {args.output_dir}/training_run_{timestamp}/")


if __name__ == "__main__":
    main()
