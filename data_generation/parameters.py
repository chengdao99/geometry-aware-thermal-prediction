"""Generation parameter overrides.

Inputs: Optional geometry, material, and operation fields.
Outputs: A BuildingParameterSet used by the serial generator."""

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Optional, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .geometric_generation.retry_divide_rooms import RoomDivisionResult
    from .geometric_generation.extrude_rooms import ExtrusionResult
    from .material_generation.assign_materials import MaterializedBuildingModel
_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


@dataclass
class BuildingParameterSet:
    """Optional seeded-generation overrides. Unspecified fields retain their configured sampling distributions."""

    parameter_tag: Optional[str] = None
    shape_type: Optional[str] = None
    length: Optional[float] = None
    width: Optional[float] = None
    n_floors: Optional[int] = None
    floor_height: Optional[float] = None
    grid_spacing: Optional[float] = None
    rotation_angle_deg: Optional[float] = None
    floor_bc: Optional[str] = None
    ceiling_bc: Optional[str] = None
    sill_height: Optional[float] = None
    head_height: Optional[float] = None
    wwr_by_orientation: Optional[Dict[str, float]] = None
    in_shade_fixed: Optional[float] = None
    out_shade_fixed: Optional[Dict[str, float]] = None
    structural_system: Optional[str] = None
    insulation_key: Optional[str] = None
    ext_wall_structure_material: Optional[str] = None
    ext_wall_structure_thickness_m: Optional[float] = None
    ext_wall_insulation_material: Optional[str] = None
    ext_wall_insulation_thickness_m: Optional[float] = None
    glazing_system_name: Optional[str] = None
    has_inter_slab_insulation: Optional[bool] = None
    context_scenario: Optional[str] = None
    svf_wall: Optional[float] = None
    svf_roof: Optional[float] = None
    svf_height_factor: Optional[float] = None
    out_shade_material: Optional[str] = None
    scenario_type: Optional[str] = None
    ach_day: Optional[float] = None
    ach_night: Optional[float] = None
    day_start_hour: Optional[int] = None
    night_start_hour: Optional[int] = None
    equip_intensity: Optional[str] = None
    equip_peak: Optional[float] = None
    equip_shape: Optional[str] = None
    occ_intensity: Optional[str] = None
    occ_peak: Optional[float] = None
    occ_shape: Optional[str] = None
    hvac_mode: Optional[str] = None
    hvac_peak: Optional[float] = None
    hvac_shape: Optional[str] = None
