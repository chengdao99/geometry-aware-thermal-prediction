"""Run MAIN on one forecast window. Inputs: Checkpoint, paired NPZ, EPW, and date/horizon. Outputs: Per-building prediction XLSX."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from data_pipeline.paths import DEFAULT_EPW, resolve_path
from data_pipeline.spreadsheets import CALENDAR_START


def main() -> None:
    """Run a single month/day window using local standard time on the non-leap calendar."""
    parser = argparse.ArgumentParser(
        description="MAIN cold-start forecasts as one XLSX per building."
    )
    parser.add_argument("--checkpoint", default="checkpoints/main/best_model.pt")
    parser.add_argument("--norm")
    parser.add_argument("--config")
    parser.add_argument("--voxel-dir", default="data_output/test_set/voxel")
    parser.add_argument("--label-dir", default="data_output/test_set/label")
    parser.add_argument(
        "--epw", default=DEFAULT_EPW, help=f"Weather file; default: {DEFAULT_EPW}"
    )
    parser.add_argument("--output-dir", default="data_output/results")
    parser.add_argument(
        "--start", default="07-18", help="MM-DD or MM-DDTHH:MM on the non-leap source calendar."
    )
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--building-id", action="append")
    args = parser.parse_args()
    try:
        start = datetime.strptime(
            f"{CALENDAR_START.year}-" + args.start,
            "%Y-%m-%dT%H:%M" if "T" in args.start else "%Y-%m-%d",
        )
    except ValueError:
        parser.error("--start must be MM-DD or MM-DDTHH:MM on a non-leap calendar.")
    if start.minute != 0 or args.hours < 1:
        parser.error("Start must be a whole hour and hours must be positive.")
    start_hour = int((start - CALENDAR_START).total_seconds() // 3600)
    checkpoint = resolve_path(args.checkpoint)
    norm = resolve_path(args.norm) if args.norm else checkpoint.parent / "norm_stats.npz"
    config = resolve_path(args.config) if args.config else checkpoint.parent / "run_config.json"
    run_config = json.loads(config.read_text(encoding="utf-8"))
    from .runner import run_inference

    results = run_inference(
        checkpoint,
        norm,
        run_config,
        resolve_path(args.voxel_dir),
        resolve_path(args.label_dir),
        resolve_path(args.epw),
        resolve_path(args.output_dir),
        start_hour,
        args.hours,
        args.device,
        args.building_id,
    )
    print(
        json.dumps(
            {
                "start_hour": start_hour,
                "forecast_hours": args.hours,
                "warmup_hours": 72,
                "buildings": results,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
