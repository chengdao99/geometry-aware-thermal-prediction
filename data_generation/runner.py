"""Generate seeded buildings serially. Inputs: Seed, output paths, and parameter overrides. Outputs: HBJSON, metadata, and optional geometry PNG."""

from __future__ import annotations

from pathlib import Path
from .parameters import BuildingParameterSet
from .geometric_generation.generate_footprint import generate_footprint
from .geometric_generation.retry_divide_rooms import validated_divide_rooms
from .geometric_generation.rotate_building import rotate_building
from .geometric_generation.extrude_rooms import stack_floors
from .geometric_generation.add_windows import add_windows
from .material_generation.assign_materials import assign_materials
from .material_generation.assign_program import assign_program
from .material_generation.export_hbjson import export_hbjson
from .rendering import _build_identifier, _save_plot


def generate_building(
    seed: int, output_dir: Path, parameters: BuildingParameterSet, plot_dir: Path | None = None
) -> dict:
    """Generate exactly one requested seed and return its stable artifact identities."""
    footprint = generate_footprint(shape_type=parameters.shape_type, seed=seed)
    division, retries = validated_divide_rooms(
        footprint,
        seed=seed + 500,
        **({"grid_spacing": parameters.grid_spacing} if parameters.grid_spacing else {}),
    )
    if division is None:
        raise RuntimeError(f"Room division failed for seed {seed}; no replacement seed was used.")
    if parameters.rotation_angle_deg is not None:
        division = rotate_building(division, fixed_angle_deg=parameters.rotation_angle_deg)
    else:
        division = rotate_building(division, seed=seed + 1000)
    extrusion = stack_floors(division, seed=seed + 1500, overrides=parameters)
    geometry = add_windows(division, extrusion, seed=seed + 2000, overrides=parameters)
    if extrusion is None or geometry is None:
        raise RuntimeError(f"Geometry generation failed for seed {seed}.")
    building = assign_materials(geometry, seed=seed + 2500, overrides=parameters)
    building = assign_program(building, seed=seed + 3000, overrides=parameters)
    building_id = _build_identifier(building, footprint.shape_type, seed, 1, None)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (output_dir / f"{building_id}{suffix}").exists() for suffix in [".hbjson", "_meta.json"]
    ):
        raise FileExistsError(f"Building artifacts already exist: {building_id}")
    hbjson, metadata = export_hbjson(building, output_dir, output_dir, building_id, overwrite=False)
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        _save_plot(
            division,
            extrusion,
            building,
            plot_dir / f"{building_id}.png",
            building_id,
            seed,
            retries,
        )
    return {
        "seed": seed,
        "building_id": building_id,
        "hbjson": hbjson.name,
        "metadata": metadata.name,
        "room_division_retries": retries,
    }
