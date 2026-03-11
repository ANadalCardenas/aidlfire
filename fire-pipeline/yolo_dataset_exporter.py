"""
yolo_dataset_exporter.py

This file is responsible for:
  1) Loading your existing patch dataset (8-channel image tensors + pixel masks)
     via your WildfireDataModule in dataset.py.
  2) Converting it into a YOLO DETECTION dataset on disk.
  3) Calling the YOLO trainer (yolo_runner.py).

Why do we need this exporter?
-----------------------------
Ultralytics YOLO expects files on disk in a specific format. Your project stores
patches as .npy arrays and uses pixel-level masks (segmentation).

We need to adapt:

A) From segmentation masks -> detection labels:
   - detection requires bounding boxes
   - we derive bounding boxes from the segmentation mask

B) From 8-channel tensors -> image files Ultralytics can load:
   - PNG/JPG only support 3 channels
   - so we export multi-band TIFF files with 8 channels (7 bands + NDVI)

C) Ensure Ultralytics knows there are 8 input channels:
   - data.yaml must include: channels: 8

Dataset semantics (confirmed from your dataset.py):
--------------------------------------------------
Your code uses (mask > 0) to compute "fire_fraction".
That implies:
  - mask value 0 = background / no-fire
  - values 1..N-1 are foreground (fire or fire severity classes)

Therefore, when generating detection labels, we SKIP class 0 and only generate boxes
for class IDs 1..N-1.

Scaling strategy:
-----------------
We must convert float tensors to a file format. The default approach below is
per-image, per-channel min-max scaling to uint8 [0..255].

Pros: simple, robust
Cons: not globally deterministic (deployment may prefer fixed scaling)

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import shutil

import numpy as np
import torch
import tifffile

from dataset import WildfireDataModule
from yolo_runner import train_and_validate_yolo_det7, YoloDetTrainCfg


# =============================================================================
# Export configuration
# =============================================================================

@dataclass
class ExportDet7Cfg:
    """
    Configuration for exporting a YOLO detection dataset (8 channels: 7 bands + NDVI).

    channels: must be 8
    mask_to_boxes_mode:
      - "single": one box per class per image (fast, simple)
      - "components": one box per connected component per class (more correct when multiple blobs)
        Requires: scipy
    min_box_area_px: filters tiny noise boxes
    """
    channels: int = 8
    mask_to_boxes_mode: str = "components"
    min_box_area_px: int = 10


# =============================================================================
# Image export utilities
# =============================================================================

def scale_to_uint8_per_channel_minmax(x_chw: torch.Tensor) -> np.ndarray:
    """
    Convert (C,H,W) float tensor -> (C,H,W) uint8 using per-channel min/max scaling.
    """
    x = x_chw.detach().cpu().float()
    c, h, w = x.shape
    out = torch.empty((c, h, w), dtype=torch.uint8)

    for i in range(c):
        ch = x[i]
        mn = float(ch.min())
        mx = float(ch.max())
        if mx - mn < 1e-6:
            out[i] = torch.zeros_like(ch, dtype=torch.uint8)
        else:
            out[i] = ((ch - mn) / (mx - mn) * 255).clamp(0, 255).to(torch.uint8)

    return out.numpy()


def save_tiff_chw(path: Path, arr_chw_uint8: np.ndarray) -> None:
    """
    Save a multi-band TIFF in CHW layout.

    Ultralytics multispectral support commonly uses TIFF for multi-band images.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr_chw_uint8, photometric="minisblack")


# =============================================================================
# Mask -> detection labels utilities
# =============================================================================

def bbox_from_binary_mask(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Return bbox around all True pixels (xyxy inclusive).
    """
    ys, xs = np.where(binary)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def connected_components_boxes(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    One bbox per connected component (blob)
    """
    from scipy.ndimage import label, find_objects

    labeled, n = label(binary.astype(np.uint8))
    if n == 0:
        return []

    slices = find_objects(labeled)
    boxes: list[tuple[int, int, int, int]] = []
    for slc in slices:
        if slc is None:
            continue
        y0, y1 = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        boxes.append((x0, y0, x1 - 1, y1 - 1))  # inclusive xyxy
    return boxes


def yolo_xywh_norm_from_xyxy(
    x_min: int, y_min: int, x_max: int, y_max: int, w: int, h: int
) -> tuple[float, float, float, float]:
    """
    Convert pixel bbox (xyxy inclusive) to YOLO normalized xywh.
    """
    bw = (x_max - x_min + 1)
    bh = (y_max - y_min + 1)
    cx = x_min + bw / 2.0
    cy = y_min + bh / 2.0
    return (cx / w, cy / h, bw / w, bh / h)


def mask_to_yolo_det_labels(
    mask_hw: np.ndarray,
    num_classes: int,
    mode: str,
    min_box_area_px: int,
) -> list[tuple[int, float, float, float, float]]:
    """
    Convert your segmentation mask (H,W) into YOLO detection labels.

    IMPORTANT: class 0 is background and is skipped.
    Only classes 1..num_classes-1 produce boxes.
    """
    h, w = mask_hw.shape
    labels: list[tuple[int, float, float, float, float]] = []

    for cls in range(1, num_classes):
        binary = (mask_hw == cls)
        if not binary.any():
            continue

        if mode == "single":
            bbox = bbox_from_binary_mask(binary)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            if area < min_box_area_px:
                continue
            x, y, bw, bh = yolo_xywh_norm_from_xyxy(x0, y0, x1, y1, w, h)
            labels.append((cls, x, y, bw, bh))

        elif mode == "components":
            for x0, y0, x1, y1 in connected_components_boxes(binary):
                area = (x1 - x0 + 1) * (y1 - y0 + 1)
                if area < min_box_area_px:
                    continue
                x, y, bw, bh = yolo_xywh_norm_from_xyxy(x0, y0, x1, y1, w, h)
                labels.append((cls, x, y, bw, bh))
        else:
            raise ValueError(f"Unknown mask_to_boxes_mode: {mode}")

    return labels


def save_yolo_label_txt(path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    """
    Save one YOLO detection label file per image.
    Empty file is valid for images with no objects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not labels:
        path.write_text("")
        return
    lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for (c, x, y, w, h) in labels]
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# Export YOLO dataset
# =============================================================================

def export_yolo_det7_dataset(
    patches_dir: Path,
    export_root: Path,
    num_classes: int,
    export_cfg: ExportDet7Cfg,
    train_cfg: YoloDetTrainCfg,
    num_workers: int = 4,
    sen2fire_dir: Path | None = None,
) -> dict:
    """
    Exports YOLO detection dataset AND calls the YOLO runner.

    Returns:
      dict containing metrics from yolo_runner.py
    """
    export_root = Path(export_root)
    ds_dir = export_root / "yolo_det_8ch_dataset"

    # Clean old export to avoid mixing datasets
    if ds_dir.exists():
        shutil.rmtree(ds_dir)

    # Standard Ultralytics directory layout
    (ds_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labels/train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labels/val").mkdir(parents=True, exist_ok=True)

    # If using connected components, check scipy once with a clear message
    if export_cfg.mask_to_boxes_mode == "components":
        try:
            import scipy  # noqa: F401
        except Exception as e:
            raise ImportError(
                "mask_to_boxes_mode='components' requires scipy.\n"
                "Install with: pip install scipy\n"
                "Or rerun with: --mask_to_boxes_mode single"
            ) from e

    # Use your existing datamodule so export is consistent with your patch pipeline
    dm = WildfireDataModule(
        patches_root=patches_dir,
        batch_size=train_cfg.batch,
        num_workers=num_workers,
        train_augment=None,
        fire_augment=None,
        use_weighted_sampling=False,
        fire_sample_weight=1.0,
    )

    def dump(loader, split: str):
        idx = 0
        for images, masks in loader:
            # images: (B,C,H,W), masks: (B,H,W)
            for i in range(images.shape[0]):
                img7 = images[i]  # (C,H,W) float
                mask = masks[i].detach().cpu().numpy().astype(np.int32)  # (H,W) int

                if img7.shape[0] != export_cfg.channels:
                    raise ValueError(f"Expected {export_cfg.channels} channels but got {img7.shape[0]}.")

                if int(mask.max()) >= num_classes:
                    raise ValueError(f"Mask max={int(mask.max())} but num_classes={num_classes}.")

                # ---------------------------------------------------------------------
                # IMAGE EXPORT (8-channel)
                # ---------------------------------------------------------------------
                img_uint8_chw = scale_to_uint8_per_channel_minmax(img7)  # (C,H,W) uint8

                # ---------------------------------------------------------------------
                # LABEL EXPORT (mask -> detection)
                # We derive bounding boxes from your segmentation mask.
                # Background (0) is ignored; classes 1..N-1 generate boxes.
                # ---------------------------------------------------------------------
                labels = mask_to_yolo_det_labels(
                    mask_hw=mask,
                    num_classes=num_classes,
                    mode=export_cfg.mask_to_boxes_mode,
                    min_box_area_px=export_cfg.min_box_area_px,
                )

                stem = f"{split}_{idx:06d}"
                save_tiff_chw(ds_dir / f"images/{split}/{stem}.tif", img_uint8_chw)
                save_yolo_label_txt(ds_dir / f"labels/{split}/{stem}.txt", labels)
                idx += 1

    dump(dm.train_dataloader(), "train")
    dump(dm.val_dataloader(), "val")

    if sen2fire_dir is not None:
        from sen2fire_dataset import Sen2FireDataset
        from torch.utils.data import DataLoader
        print("Exporting Sen2Fire patches into YOLO dataset...")
        sen2fire_train = Sen2FireDataset(sen2fire_dir, split="train", use_s2cloudless=False)
        sen2fire_val = Sen2FireDataset(sen2fire_dir, split="val", use_s2cloudless=False)
        dump(DataLoader(sen2fire_train, batch_size=train_cfg.batch, num_workers=num_workers), "train")
        dump(DataLoader(sen2fire_val, batch_size=train_cfg.batch, num_workers=num_workers), "val")
        print(f"Sen2Fire added: {len(sen2fire_train)} train, {len(sen2fire_val)} val")

    # data.yaml for Ultralytics
    names = [f"class_{i}" for i in range(num_classes)]
    data_yaml = ds_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {ds_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"channels: {export_cfg.channels}\n"
        f"names: {names}\n"
    )

    # Now call the YOLO runner (training + validation)
    metrics = train_and_validate_yolo_det7(
        data_yaml=data_yaml,
        output_dir=export_root,
        cfg=train_cfg,
    )
    return metrics


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Export 8-channel patches to YOLO DETECT format and train YOLOv8."
    )
    parser.add_argument("--patches_dir", type=Path, required=True, help="Root dir with train/val splits of .npy patches.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where to write exported dataset + YOLO runs.")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes in mask INCLUDING background=0.")

    # Training params (passed to yolo_runner)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO detection weights checkpoint to start from.")

    # Export params
    parser.add_argument("--mask_to_boxes_mode", type=str, default="components", choices=["single", "components"])
    parser.add_argument("--min_box_area_px", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)

    # Device
    parser.add_argument("--yolo-device", type=str, default="0", help="YOLO device: '0', 'cpu', etc.")

    args = parser.parse_args()

    train_cfg = YoloDetTrainCfg(
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        device=args.yolo_device,
        model_weights=args.weights,
    )

    export_cfg = ExportDet7Cfg(
        channels=8,
        mask_to_boxes_mode=args.mask_to_boxes_mode,
        min_box_area_px=args.min_box_area_px,
    )

    metrics = export_yolo_det7_dataset(
        patches_dir=args.patches_dir,
        export_root=args.output_dir,
        num_classes=args.num_classes,
        export_cfg=export_cfg,
        train_cfg=train_cfg,
        num_workers=args.num_workers,
    )

    print("\nYOLO validation metrics:", metrics.get("val_results"))
    print("Dataset YAML:", metrics.get("data_yaml"))


if __name__ == "__main__":
    main()
