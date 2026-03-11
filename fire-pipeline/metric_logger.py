import csv
from pathlib import Path
from typing import Dict


class MetricLogger:
    """
    Logs per-epoch metrics for train / val / test splits
    in CSV format, compatible with existing train.py.
    """

    def __init__(
        self,
        output_dir: Path,
        filename: str = "metrics.csv",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.filepath = self.output_dir / filename
        self._fieldnames = None  # inferred dynamically

    def log(
        self,
        epoch: int,
        split: str,
        metrics: Dict[str, float],
    ):
        """
        Args:
            epoch: Epoch number
            split: 'train' | 'val' | 'test'
            metrics: Dict returned by metrics.compute() + loss
        """
        row = {
            "epoch": epoch,
            "split": split,
            **metrics,
        }

        # Infer CSV header on first write
        write_header = not self.filepath.exists()
        if write_header:
            self._fieldnames = list(row.keys())

        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
