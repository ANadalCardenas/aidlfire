#!/usr/bin/env python3
"""
Export W&B training runs to CSV/JSON for local analysis.

Usage:
    # Export all runs from fire-detection (uses default entity from wandb login)
    uv run python export_wandb_runs.py

    # Specify entity and project
    uv run python export_wandb_runs.py --entity myteam --project fire-detection

    # Export with per-step history (slower, larger output)
    uv run python export_wandb_runs.py --history

    # Output to specific path
    uv run python export_wandb_runs.py --output ./wandb_export.csv

Requires: wandb (uv sync --extra train)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict for CSV columns."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict) and not (v and isinstance(next(iter(v.values()), None), dict)):
            # Only flatten one level; nested dicts become JSON strings
            for k2, v2 in v.items():
                if isinstance(v2, (dict, list)):
                    out[f"{key}.{k2}"] = json.dumps(v2) if v2 is not None else ""
                else:
                    out[f"{key}.{k2}"] = v2
        elif isinstance(v, (dict, list)):
            out[key] = json.dumps(v) if v is not None else ""
        else:
            out[key] = v
    return out


def export_runs(
    entity: str | None,
    project: str,
    output_csv: Path,
    output_json: Path | None,
    include_history: bool,
    history_dir: Path | None,
) -> None:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, per_page=100, order="-created_at")

    rows = []
    for run in runs:
        run.load()
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "url": run.url,
        }
        # Summary metrics (best/final values)
        if run.summary:
            summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
            row.update(_flatten_dict(summary, "summary."))
        # Config (hyperparameters)
        if run.config:
            config_flat = _flatten_dict(dict(run.config), "config.")
            row.update(config_flat)
        rows.append(row)

        if include_history and history_dir:
            history_dir.mkdir(parents=True, exist_ok=True)
            try:
                hist = run.history(pandas=True, samples=5000)
                if hist is not None and not hist.empty:
                    hist_path = history_dir / f"{run.id}_{run.name or 'unnamed'}.csv"
                    hist_path = hist_path.with_stem(hist_path.stem[:80])
                    hist.to_csv(hist_path, index=False)
            except Exception as e:
                print(f"Warning: could not export history for {run.id}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Exported {len(df)} runs to {output_csv}")

    if output_json:
        with open(output_json, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"Exported JSON to {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export W&B runs to CSV/JSON")
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (username/team). Omit to use default from wandb login.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="fire-detection",
        help="W&B project name",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("wandb_runs_export.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also export full JSON (one row per run)",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Export per-step history for each run to separate CSVs",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("wandb_history"),
        help="Directory for history CSVs (used with --history)",
    )
    args = parser.parse_args()

    export_runs(
        entity=args.entity,
        project=args.project,
        output_csv=args.output,
        output_json=args.json,
        include_history=args.history,
        history_dir=args.history_dir if args.history else None,
    )


if __name__ == "__main__":
    main()
