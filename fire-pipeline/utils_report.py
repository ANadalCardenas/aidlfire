# utils_report.py
from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Any, Dict

ALL_CSV_COLUMNS = [
    "dataset", "model_name", "wandb_run_name",
    # Primary metrics — UPPERCASE to highlight importance
    "MAP_50_95", "VAL_F1", "FIRE_IOU",
    # Shared metrics
    "val_loss", "val_precision", "val_recall",
    # YOLO-specific
    "map_50", "f1",
    # CNN scratch-specific
    "val_accuracy", "val_auc_roc",
    # U-Net scratch-specific
    "fire_dice", "mean_iou",
    # Training metadata
    "train_time_s", "num_params", "best_epoch",
    # Launch command
    "command",
]


def append_results_csv(csv_path: Path, row: dict) -> None:
    """Append one result row to the CSV, creating it with headers if new."""
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=ALL_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in ALL_CSV_COLUMNS})


def print_run_summary(
    title: str,
    train_time_s: float,
    best_checkpoint: str,
    best_epoch: int,
    best_metrics: Dict[str, Any],
    num_params: int,
) -> None:
    print("\n" + "=" * 60)
    print(f"RUN SUMMARY: {title}")
    print("=" * 60)
    print(f"time_to_train_s: {train_time_s:.2f}")
    print(f"best_checkpoint: {best_checkpoint}")
    print(f"num_parameters: {num_params}")
    print(f"best_epoch: {best_epoch}")
    print("best_metrics:")
    for k, v in best_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {float(v):.6f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60 + "\n")