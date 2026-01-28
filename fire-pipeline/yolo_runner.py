from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import torch

from PIL import Image

from dataset import WildfireDataModule 


@dataclass
class YoloCfg:
    """
    Configuration for exporting your dataset and training YOLO.

    imgsz:  Ultralytics input resolution (square)
    batch:  Ultralytics batch size
    epochs: Training epochs for YOLO baseline
    device: Ultralytics device string ("0", "1", "cpu", etc.)
    rgb_channels:
        Which 3 channels to export as RGB from your 7-channel tensor.
        RGB channels : (0,1,2)
    """
    imgsz: int = 512
    batch: int = 16
    epochs: int = 50
    device: str = "0"
    rgb_channels: tuple[int, int, int] = (0, 1, 2)  # RGB indices


def _to_uint8_rgb(x7: torch.Tensor, rgb_channels=(0, 1, 2)) -> np.ndarray:
    """(7,H,W) float tensor -> (H,W,3) uint8 image for YOLO."""
    c0, c1, c2 = rgb_channels
    rgb = torch.stack([x7[c0], x7[c1], x7[c2]], dim=0).detach().cpu().float() 
    out = []
    for ch in rgb:
        mn = float(ch.min())
        mx = float(ch.max())
        if mx - mn < 1e-6:
            out.append(torch.zeros_like(ch, dtype=torch.uint8))
        else:
            out.append(((ch - mn) / (mx - mn) * 255).clamp(0, 255).to(torch.uint8))

    return torch.stack(out, dim=-1).numpy() 


def _save_png(path: Path, arr: np.ndarray) -> None:
    """
    Save an uint8 array as PNG using PIL (no extra heavy deps).

    arr can be:
      - (H,W,3) uint8 for RGB
      - (H,W) uint8 for masks
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def train_yolo_seg(
    patches_dir: Path,
    output_dir: Path,
    num_classes: int,
    cfg: YoloCfg = YoloCfg(),
    num_workers: int = 4,
) -> dict:
    """
    Minimal YOLOv8-seg baseline:
      1) Export RGB images + mask PNGs from your datamodule
      2) Convert masks -> YOLO-seg labels via Ultralytics
      3) Train YOLOv8n-seg
    """
    output_dir = Path(output_dir)
    ds_dir = output_dir / "yolo_dataset"

    # Fresh export dir
    if ds_dir.exists():
        shutil.rmtree(ds_dir)

    (ds_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (ds_dir / "masks/train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "masks/val").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labels/train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labels/val").mkdir(parents=True, exist_ok=True)

    dm = WildfireDataModule(
        patches_root=patches_dir,
        batch_size=cfg.batch,
        num_workers=num_workers,
        train_augment=None,
        fire_augment=None,
        use_weighted_sampling=False,
        fire_sample_weight=1.0,
    )

    # Export Torch dataloaders to yolo image + mask files
    def dump(loader, split: str):
        idx = 0
        for images, masks in loader:
            # images: (B,7,H,W), masks: (B,H,W)
            for i in range(images.shape[0]):
                img7 = images[i]
                msk = masks[i].detach().cpu().to(torch.uint8).numpy()  # (H,W)

                if int(msk.max()) >= num_classes:
                    raise ValueError(
                        f"Mask max={int(msk.max())} but num_classes={num_classes}."
                    )

                rgb = _to_uint8_rgb(img7, cfg.rgb_channels)

                stem = f"{split}_{idx:06d}"
                _save_png(ds_dir / f"images/{split}/{stem}.png", rgb)
                _save_png(ds_dir / f"masks/{split}/{stem}.png", msk)
                idx += 1

    dump(dm.train_dataloader(), "train")
    dump(dm.val_dataloader(), "val")

    # Convert mask PNGs -> YOLO-seg polygon labels
    from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
    convert_segment_masks_to_yolo_seg(
        masks_dir=str(ds_dir / "masks"),
        output_dir=str(ds_dir / "labels"),
        classes=num_classes,
    )

    # data.yaml required by Ultralytics
    names = [f"class_{i}" for i in range(num_classes)]
    (ds_dir / "data.yaml").write_text(
        f"path: {ds_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names: {names}\n"
    )

    # Train YOLOv8 segmentation
    from ultralytics import YOLO
    model = YOLO("yolov8n-seg.pt")

    train_res = model.train(
        data=str(ds_dir / "data.yaml"),
        task="segment",
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        epochs=cfg.epochs,
        device=cfg.device,
        project=str(output_dir),
        name="runs",
        exist_ok=True,
    )

    val_res = model.val(
        data=str(ds_dir / "data.yaml"),
        task="segment",
        device=cfg.device,
    )

    return {
        "data_yaml": str(ds_dir / "data.yaml"),
        "train_results": getattr(train_res, "results_dict", None),
        "val_results": getattr(val_res, "results_dict", None),
    }
