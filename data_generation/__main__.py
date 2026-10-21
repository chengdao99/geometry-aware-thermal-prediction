"""Generate a requested seed sequence. Inputs: CLI seeds, YAML, and output paths. Outputs: HBJSON, metadata, plots, and a split manifest."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import fields
import yaml
from data_pipeline.paths import resolve_path
from .parameters import BuildingParameterSet
from .runner import generate_building


def main() -> None:
    """Generate buildings serially without changing or substituting requested seeds."""
    parser = argparse.ArgumentParser(
        description="Generate seeded parameterized HBJSON buildings serially."
    )
    parser.add_argument("--start-seed", type=int, default=101)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output-dir", default="data_output/hbjson")
    parser.add_argument(
        "--plot-dir", default=None, help="Optional directory for building geometry PNG files."
    )
    parser.add_argument("--config", default="data_generation/configs/free_running.yaml")
    parser.add_argument("--manifest", default="data_output/manifest.json")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive.")
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8")) or {}
    valid = {field.name for field in fields(BuildingParameterSet)}
    if set(config) - valid:
        parser.error(f"Unknown generation parameters: {sorted(set(config) - valid)}")
    parameters = BuildingParameterSet(**config)
    manifest = resolve_path(args.manifest)
    if manifest.exists():
        raise FileExistsError(f"Manifest already exists: {manifest.name}")
    records = []
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    for seed in range(args.start_seed, args.start_seed + args.count):
        record = generate_building(
            seed,
            resolve_path(args.output_dir),
            parameters,
            resolve_path(args.plot_dir) if args.plot_dir else None,
        )
        record["split"] = (
            "train_set"
            if 101 <= seed <= 109
            else "val_set" if seed == 110 else "test_set" if 111 <= seed <= 120 else None
        )
        records.append(record)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {"schema_version": 1, "generation_parameters": config, "buildings": records},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f'Generated seed {seed}: {record["building_id"]}', flush=True)


if __name__ == "__main__":
    main()
