"""

This file is responsible ONLY for:
  - Training and validating Ultralytics YOLOv8 DETECTION on a prepared dataset.

It expects a standard Ultralytics dataset layout like:

  <dataset_root>/
    images/train/*.tif
    images/val/*.tif
    labels/train/*.txt
    labels/val/*.txt
    data.yaml

Where:
  - Each image is an 8-channel multi-band TIFF (CHW: 8,H,W)
  - Each label is YOLO-detection format:
      <class> <x_center> <y_center> <width> <height>
    with all coordinates normalized to [0..1].

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any



@dataclass
class YoloDetTrainCfg:
    """
    Training configuration.

    imgsz:        Ultralytics training resolution (square)
    batch:        Ultralytics batch size
    epochs:       Training epochs
    device:       Ultralytics device string ("0", "cpu", "mps", etc.)
    model_weights: which YOLO weights to start from (detection weights)
    lr0:          Initial learning rate
    weight_decay: Optimizer weight decay
    """
    imgsz: int = 512
    batch: int = 16
    epochs: int = 50
    device: str = "0"
    model_weights: str = "yolov8n.pt"
    lr0: float = 1e-2
    weight_decay: float = 5e-4


def train_and_validate_yolo_det7(
    data_yaml: Path | str,
    output_dir: Path | str,
    cfg: YoloDetTrainCfg = YoloDetTrainCfg(),
) -> dict[str, Any]:
    """
    Train + validate YOLOv8 detection using Ultralytics.

    Args:
        data_yaml: Path to the exported dataset YAML (contains channels: 8).
        output_dir: Directory where Ultralytics will write runs/ artifacts.
        cfg: training configuration.

    Returns:
        Dict with train/val metrics (when available).
    """
    from ultralytics import YOLO
    import time
    import torch
    from pathlib import Path

    data_yaml = Path(data_yaml)
    output_dir = Path(output_dir)

    # Resolve device: Ultralytics does not accept "auto" or "mps" as a string
    device = cfg.device
    if device in ("auto", "mps"):
        device = "0" if torch.cuda.is_available() else "cpu"

    # Load a detection model (NOT segmentation).
    model = YOLO(cfg.model_weights)

    # parameter count
    num_params = sum(p.numel() for p in model.model.parameters())

    # start timing
    t0 = time.perf_counter()

    use_cuda = torch.cuda.is_available() and device not in ("cpu", "mps")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # -----------------------
    # training
    # -----------------------
    train_res = model.train(
        data=str(data_yaml),
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        device=device,
        lr0=cfg.lr0,
        weight_decay=cfg.weight_decay,
        exist_ok=True,
    )

    # -----------------------
    # validation
    # -----------------------
    val_res = model.val(
        data=str(data_yaml),
        imgsz=cfg.imgsz,
        device=device,
    )

    # stop timer
    train_time_s = time.perf_counter() - t0

    # peak GPU memory
    peak_mem_mb = None
    device_name = str(device)

    if use_cuda:
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        device_name = torch.cuda.get_device_name(0)

    # locate best checkpoint
    run_dir = Path(train_res.save_dir)
    best_ckpt = run_dir / "weights" / "best.pt"

    val_dict = getattr(val_res, "results_dict", {}) or {}

    # Compute F1 from precision and recall (not reported directly by Ultralytics)
    p = val_dict.get("metrics/precision(B)", 0.0)
    r = val_dict.get("metrics/recall(B)", 0.0)
    val_dict["metrics/f1(B)"] = 2 * p * r / (p + r + 1e-9)

    return {
        "train_time_s": train_time_s,
        "best_checkpoint": str(best_ckpt),
        "best_metrics": val_dict,
        "num_params": num_params,
        "device_name": device_name,
        "peak_mem_mb": peak_mem_mb,
    }  
