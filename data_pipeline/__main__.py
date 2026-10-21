"""Process a file or directory of HBJSON buildings. Inputs: CLI artifact paths and simulation settings. Outputs: Serial voxel/label NPZ and label XLSX."""

from __future__ import annotations

import argparse
import json
import logging
from .paths import DEFAULT_EPW, resolve_path


def main() -> None:
    """Run independently selected voxel or label branches with repository-relative defaults."""
    parser = argparse.ArgumentParser(
        description="Generate voxel NPZ and/or annual EnergyPlus labels serially."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--hbjson")
    source.add_argument("--hbjson-dir", default="data_output/hbjson")
    parser.add_argument("--mode", choices=["voxel", "label", "all"], default="all")
    parser.add_argument("--voxel-out-dir", default="data_output/train_set/voxel")
    parser.add_argument("--label-out-dir", default="data_output/train_set/label")
    parser.add_argument(
        "--epw", default=DEFAULT_EPW, help=f"Weather file; default: {DEFAULT_EPW}"
    )
    parser.add_argument("--energyplus-exe")
    parser.add_argument(
        "--manifest", help="Use seed-to-split metadata to route the 20-building demonstration."
    )
    parser.add_argument("--split-root", default="data_output")
    parser.add_argument("--work-dir", help="Optional temporary working directory.")
    parser.add_argument("--no-label-xlsx", action="store_true")
    args = parser.parse_args()
    if args.mode != "voxel" and not args.energyplus_exe:
        parser.error("--energyplus-exe is required for labels.")
    files = (
        [resolve_path(args.hbjson)]
        if args.hbjson
        else sorted(resolve_path(args.hbjson_dir).glob("*.hbjson"))
    )
    if not files or any(not path.is_file() for path in files):
        parser.error("No matching HBJSON input files were found.")
    splits = {}
    if args.manifest:
        records = json.loads(resolve_path(args.manifest).read_text(encoding="utf-8"))["buildings"]
        splits = {row["building_id"]: row["split"] for row in records}
        if len(splits) != len(records) or any(
            s not in {"train_set", "val_set", "test_set"} for s in splits.values()
        ):
            parser.error("Manifest has duplicate building IDs or invalid splits.")
        if set(splits) != {p.stem for p in files}:
            parser.error("Manifest and HBJSON building IDs do not match exactly.")
    from .runner import process_building

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    for hbjson in files:
        root = resolve_path(args.split_root) / splits[hbjson.stem] if splits else None
        process_building(
            hbjson,
            args.mode,
            root / "voxel" if root else resolve_path(args.voxel_out_dir),
            root / "label" if root else resolve_path(args.label_out_dir),
            resolve_path(args.epw) if args.epw else None,
            resolve_path(args.energyplus_exe) if args.energyplus_exe else None,
            not args.no_label_xlsx,
            resolve_path(args.work_dir) if args.work_dir else None,
        )
        print(f"Processed {args.mode}: {hbjson.stem}", flush=True)


if __name__ == "__main__":
    main()
