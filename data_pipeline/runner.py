"""Process HBJSON serially into voxel and EnergyPlus labels. Inputs: HBJSON, EPW, and EnergyPlus. Outputs: NPZ and optional label XLSX."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from .hbjson_utils.fix_adjacency import fix_adjacency_and_return
from .hbjson_utils.to_idf import convert_hbjson_to_idf
from .simulation.idf_simulator import (
    inject_output_variables,
    run_energyplus,
    extract_and_save_labels_npz,
)
from .voxelizer.hbjson_to_voxel import voxelize_hbjson
from .voxelizer.voxelizer_exporter import export_voxels
from .spreadsheets import export_label


def process_building(
    hbjson: Path,
    mode: str,
    voxel_dir: Path,
    label_dir: Path,
    epw: Path | None = None,
    energyplus: Path | None = None,
    label_xlsx: bool = True,
    work_dir: Path | None = None,
) -> None:
    """Run the selected branches for one building without altering its source artifacts."""
    building_id = hbjson.stem
    if mode in {"voxel", "all"}:
        target = voxel_dir / f"{building_id}_voxel.npz"
        if target.exists():
            raise FileExistsError(f"Voxel artifact already exists: {target.name}")
        points, incidence = voxelize_hbjson(hbjson, return_zone_map=True)
        if not len(points):
            raise ValueError(f"No active voxels for {building_id}")
        voxel_dir.mkdir(parents=True, exist_ok=True)
        export_voxels(
            points, f"{building_id}_voxel", str(voxel_dir), ["npz"], zone_voxel_map=incidence
        )
    if mode not in {"label", "all"}:
        return
    if epw is None or energyplus is None or not energyplus.is_file() or not epw.is_file():
        raise FileNotFoundError(
            "Label generation requires an existing EPW and EnergyPlus executable."
        )
    target = label_dir / f"{building_id}_label.npz"
    if target.exists() or (label_xlsx and target.with_suffix(".xlsx").exists()):
        raise FileExistsError(f"Label artifact already exists: {target.name}")
    if work_dir:
        work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="energyplus_", dir=work_dir) as temporary:
        temporary = Path(temporary)
        working_hbjson = temporary / hbjson.name
        shutil.copy2(hbjson, working_hbjson)
        metadata = hbjson.with_name(f"{building_id}_meta.json")
        if metadata.is_file():
            shutil.copy2(metadata, temporary / metadata.name)
        fixed, _ = fix_adjacency_and_return(working_hbjson, save_fixed=True, output_dir=temporary)
        idf = convert_hbjson_to_idf(fixed, temporary, auto_fix=False, epw_path=epw)
        inject_output_variables(idf)
        run_energyplus(energyplus, idf, epw, temporary)
        label_dir.mkdir(parents=True, exist_ok=True)
        extract_and_save_labels_npz(temporary, building_id, label_dir, fixed)
        if label_xlsx:
            export_label(target, epw)
