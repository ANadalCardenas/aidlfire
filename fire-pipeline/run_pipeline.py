#!/usr/bin/env python
"""
Run the patch generation pipeline on the CEMS Wildfire dataset.

This script handles the full workflow:
1. Extract tar archives (if not already done)
2. Generate patches from all splits
3. Compute dataset statistics
"""

import argparse
import subprocess
import sys
from pathlib import Path

from patch_generator import PatchGenerator, PatchConfig


def extract_archives(dataset_dir: Path) -> bool:
    """Extract multipart tar archives if not already extracted."""
    data_dir = dataset_dir / "data"

    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        return False

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        tar_parts = list(split_dir.glob(f"{split}.tar.*.gz.part"))

        if not tar_parts:
            # Check if already extracted (look for EMSR directories)
            emsr_dirs = list(split_dir.glob("EMSR*"))
            if emsr_dirs:
                print(f"  {split}: Already extracted ({len(emsr_dirs)} activations)")
                continue
            else:
                print(f"  {split}: No tar parts found and no data extracted")
                continue

        print(f"  {split}: Extracting {len(tar_parts)} tar parts...")

        # Combine and extract: cat data/train/train.tar.* | tar -xzvf - -i
        cmd = f"cat {split_dir}/{split}.tar.*.gz.part | tar -xzvf - -i -C {split_dir}"
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            print(f"  {split}: Extraction complete")
        except subprocess.CalledProcessError as e:
            print(f"  {split}: Extraction failed: {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the CEMS Wildfire patch generation pipeline"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../wildfires-cems"),
        help="Directory containing the cloned wildfires-cems repository",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./patches"),
        help="Output directory for generated patches",
    )
    parser.add_argument(
        "--mask-type",
        choices=["DEL", "GRA"],
        default="DEL",
        help="Mask type: DEL (binary fire detection) or GRA (severity grading)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip tar extraction (use if already extracted)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process",
    )
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=0.5,
        help="Maximum cloud cover fraction (0-1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all patches, even if they already exist (default: resume)",
    )
    parser.add_argument(
        "--no-ndvi",
        action="store_true",
        help="Disable NDVI as 8th channel (output 7 channels only; default: include NDVI)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CEMS WILDFIRE PATCH GENERATION PIPELINE")
    print("=" * 60)
    print(f"Dataset directory: {args.dataset_dir.absolute()}")
    print(f"Output directory: {args.output_dir.absolute()}")
    print(f"Mask type: {args.mask_type}")
    print(f"Max cloud cover: {args.max_cloud_cover:.0%}")
    print(f"Channels: {'7 (no NDVI)' if args.no_ndvi else '8 (7 bands + NDVI)'}")
    print()

    # Check dataset exists
    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        print("Make sure to clone the dataset first:")
        print("  git lfs install")
        print("  git clone https://huggingface.co/datasets/links-ads/wildfires-cems")
        sys.exit(1)

    # Extract archives
    if not args.skip_extraction:
        print("Step 1: Extracting archives...")
        if not extract_archives(args.dataset_dir):
            print("Warning: Some extractions may have failed")
    else:
        print("Step 1: Skipping extraction (--skip-extraction)")

    # Determine the data root (where train/val/test folders are)
    data_root = args.dataset_dir / "data"
    if not data_root.exists():
        # Maybe the data is directly in the dataset dir
        data_root = args.dataset_dir

    # Check for train/val/test
    has_splits = all((data_root / s).exists() for s in ["train", "val", "test"])
    if not has_splits:
        print(f"Error: Could not find train/val/test splits in {data_root}")
        sys.exit(1)

    # Generate patches
    print("\nStep 2: Generating patches...")
    config = PatchConfig(
        max_cloud_cover=args.max_cloud_cover,
        include_ndvi=not args.no_ndvi,
    )
    generator = PatchGenerator(config)

    skip_existing = not args.force
    if skip_existing:
        print("  Resume mode: Skipping patches that already exist")
    else:
        print("  Force mode: Regenerating all patches")

    metadata = generator.process_dataset(
        data_root,
        args.output_dir,
        args.mask_type,
        args.splits,
        skip_existing=skip_existing,
    )

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    total_patches = 0
    for split, meta_list in metadata.items():
        n = len(meta_list)
        total_patches += n

        if meta_list:
            avg_cloud = sum(m.cloud_cover for m in meta_list) / n
            avg_burn = sum(m.burn_fraction for m in meta_list) / n
            print(f"  {split}: {n} patches (avg cloud: {avg_cloud:.1%}, avg burn: {avg_burn:.1%})")
        else:
            print(f"  {split}: {n} patches")

    print(f"\nTotal: {total_patches} patches")
    print(f"Output: {args.output_dir.absolute()}")

    # Estimate disk usage
    # Each patch: ~1.4MB for image + ~65KB for mask
    estimated_size_gb = total_patches * 1.5 / 1024
    print(f"Estimated size: ~{estimated_size_gb:.1f} GB")


if __name__ == "__main__":
    main()
