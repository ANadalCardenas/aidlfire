"""
Export sample images from the EO4WildFires HuggingFace dataset for use in _docs_.

Run from the repo root:
    python _docs_/scripts/export_eo4wildfires_samples.py

Requires: pip install "datasets==3.6.0" matplotlib numpy
(HuggingFace datasets 4.x is not compatible with this dataset.)
"""

from pathlib import Path

def main():
    out_dir = Path(__file__).resolve().parent.parent / "images" / "eo4wildfires"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("Missing dependency:", e)
        print('Install with: pip install "datasets==3.6.0" matplotlib numpy')
        return 1

    print("Loading EO4WildFires (validation split, may download)...")
    try:
        ds = load_dataset("AUA-Informatics-Lab/eo4wildfires", split="validation", trust_remote_code=True)
    except Exception as e:
        print("Failed to load dataset:", e)
        return 1

    ds.set_format("np")

    # Pick a few indices that likely have visible fire (e.g. 0, 16, 789 as in the notebook)
    indices = [0, 16, min(789, len(ds) - 1)]

    for i, idx in enumerate(indices):
        row = ds[idx]
        s2 = row.get("S2A")
        mask = row.get("burned_mask")

        if s2 is not None:
            # S2A: channels first, RGB = bands 3,2,1 (B04, B03, B02)
            rgb = (np.clip(s2[3:0:-1], 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(rgb)
            ax.set_title(f"EO4WildFires — Sentinel-2 RGB (sample {idx})")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / f"sample_s2_rgb_{i}.png", dpi=150, bbox_inches="tight")
            plt.close()

        if mask is not None:
            # Handle NaN: show as 0
            m = np.nan_to_num(mask, nan=0.0)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(m * 255 if m.max() <= 1 else m, cmap="hot")
            ax.set_title(f"EO4WildFires — EFFIS burned mask (sample {idx})")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / f"sample_burned_mask_{i}.png", dpi=150, bbox_inches="tight")
            plt.close()

    # One combined figure for the doc (first sample only)
    row = ds[0]
    s2 = row.get("S2A")
    mask = row.get("burned_mask")
    if s2 is not None and mask is not None:
        rgb = (np.clip(s2[3:0:-1], 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        m = np.nan_to_num(mask, nan=0.0)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("Sentinel-2 RGB (pre-fire composite)")
        axes[0].axis("off")
        axes[1].imshow(m * 255 if m.max() <= 1 else m, cmap="hot")
        axes[1].set_title("EFFIS burned area mask")
        axes[1].axis("off")
        plt.suptitle("EO4WildFires — validation sample 0")
        plt.tight_layout()
        plt.savefig(out_dir / "sample_s2_and_mask.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Copy first sample as the generic names used in the docs
    if s2 is not None:
        rgb = (np.clip(ds[0]["S2A"][3:0:-1], 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        plt.imsave(out_dir / "sample_s2_rgb.png", rgb)
    if mask is not None:
        m = np.nan_to_num(ds[0]["burned_mask"], nan=0.0)
        plt.imsave(out_dir / "sample_burned_mask.png", (m * 255 if m.max() <= 1 else m).astype(np.uint8), cmap="hot")

    print("Saved samples to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
