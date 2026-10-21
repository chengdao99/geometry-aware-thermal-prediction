"""Parameterized building assign materials.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)
from . import material_config as cfg
from . import material_library as lib
from .material_library import (
    LayerSpec,
    OpaqueConstructionSpec,
    AirBoundaryConstructionSpec,
    WindowConstructionSpec,
    OutShadeConstructionSpec,
    InShadeConstructionSpec,
)
from ..geometric_generation.add_windows import BuildingModel, FaceModel


@dataclass
class BuildingMaterialSpec:
    """Represent building material spec and its associated geometry or physical configuration."""

    structural_system: str
    insulation_key: str
    glazing_system_name: str
    face_constructions: Dict[str, Union[OpaqueConstructionSpec, AirBoundaryConstructionSpec]]
    aperture_constructions: Dict[str, WindowConstructionSpec]
    opaque_pool: Dict[str, OpaqueConstructionSpec]
    window_pool: Dict[str, WindowConstructionSpec]
    u_values: Dict[str, float]
    metadata: dict = field(default_factory=dict)
    out_shade_construction: OutShadeConstructionSpec = field(
        default_factory=lambda: lib.OUT_SHADE_CONSTRUCTIONS["light_metal"]
    )
    in_shade_construction: InShadeConstructionSpec = field(
        default_factory=lambda: lib.IN_SHADE_CONSTRUCTION
    )


@dataclass
class MaterializedBuildingModel:
    """Represent materialized building model and its associated geometry or physical configuration."""

    geometry: BuildingModel
    material_spec: BuildingMaterialSpec

    def summary(self) -> str:
        """Compute summary for the current building or model state."""
        ms = self.material_spec
        ctx = ms.metadata.get("context", {})
        bld = ms.metadata.get("building", {})
        u = ms.u_values
        lines = [
            "MaterializedBuildingModel Summary",
            f"  Structural system  : {ms.structural_system}",
            f"  Insulation         : {ms.insulation_key}",
            f"  Glazing system     : {ms.glazing_system_name}",
            f"  Building           : {bld.get('n_floors', '—')}-floor [{bld.get('building_type', '—')}]  h={bld.get('floor_height', '—')} m",
            f"  Context scenario   : {ctx.get('scenario', '—')}  (SVF wall={ctx.get('svf_wall', '—')}, roof={ctx.get('svf_roof', '—')})",
            f"  U-values [W/m²K]   :  ext_wall={u.get('ext_wall', '—'):.3f}  roof={u.get('roof', '—'):.3f}  floor={u.get('floor', '—'):.3f}  inter_slab={u.get('inter_floor', '—')}  window={u.get('window', '—'):.2f}",
            f"  Opaque constructs  : {len(ms.opaque_pool)}",
            f"  Window constructs  : {len(ms.window_pool)}",
            f"  Faces assigned     : {len(ms.face_constructions)}",
            f"  Apertures assigned : {len(ms.aperture_constructions)}",
        ]
        return "\n".join(lines)


def _sample_discrete(rng: random.Random, spec: dict) -> float:
    return rng.choices(spec["values"], weights=spec["weights"])[0]


def _base_system(structural_system: str) -> str:
    if structural_system.startswith("Concrete"):
        return "Concrete"
    return structural_system


def _make_layer(mat: lib.EnergyMaterialSpec, t: float) -> LayerSpec:
    return LayerSpec(material=mat, thickness=t)


def _build_ext_wall_construction(
    structural_system: str, insulation_key: str, rng: random.Random, overrides=None
) -> OpaqueConstructionSpec:
    wall_system = (
        overrides.ext_wall_structure_material
        if overrides and overrides.ext_wall_structure_material
        else structural_system
    )
    wall_insulation = (
        overrides.ext_wall_insulation_material
        if overrides and overrides.ext_wall_insulation_material
        else insulation_key
    )
    struct_mat = lib.STRUCTURAL_WALL_MATERIALS[wall_system]
    insul_mat = lib.INSULATION_MATERIALS[wall_insulation]
    base = _base_system(wall_system)
    struct_t = (
        float(overrides.ext_wall_structure_thickness_m)
        if overrides and overrides.ext_wall_structure_thickness_m is not None
        else _sample_discrete(rng, cfg.WALL_STRUCTURAL_T_DISCRETE[base])
    )
    insul_t = (
        float(overrides.ext_wall_insulation_thickness_m)
        if overrides and overrides.ext_wall_insulation_thickness_m is not None
        else _sample_discrete(rng, cfg.WALL_INSULATION_T_DISCRETE[wall_insulation])
    )
    if struct_t <= 0 or insul_t <= 0:
        raise ValueError("Exterior-wall layer thicknesses must be positive")
    return OpaqueConstructionSpec(
        layers=[_make_layer(insul_mat, insul_t), _make_layer(struct_mat, struct_t)]
    )


def _build_outdoor_roof_construction(
    structural_system: str, insulation_key: str, rng: random.Random
) -> OpaqueConstructionSpec:
    roof_mat = lib.ROOF_MEMBRANE
    insul_mat = lib.INSULATION_MATERIALS[insulation_key]
    struct_mat = lib.STRUCTURAL_ROOF_MATERIALS[structural_system]
    base = _base_system(structural_system)
    struct_t = _sample_discrete(rng, cfg.ROOF_STRUCTURAL_T_DISCRETE[base])
    insul_t = _sample_discrete(rng, cfg.ROOF_INSULATION_T_DISCRETE[insulation_key])
    return OpaqueConstructionSpec(
        layers=[
            _make_layer(roof_mat, 0.01),
            _make_layer(insul_mat, insul_t),
            _make_layer(struct_mat, struct_t),
        ]
    )


def _build_adiabatic_ceiling_construction(
    structural_system: str, rng: random.Random
) -> OpaqueConstructionSpec:
    struct_mat = lib.STRUCTURAL_ROOF_MATERIALS[structural_system]
    base = _base_system(structural_system)
    struct_t = _sample_discrete(rng, cfg.ROOF_STRUCTURAL_T_DISCRETE[base])
    return OpaqueConstructionSpec(layers=[_make_layer(struct_mat, struct_t)])


def _build_ground_floor_construction(
    structural_system: str, insulation_key: str, rng: random.Random
) -> OpaqueConstructionSpec:
    struct_mat = lib.STRUCTURAL_FLOOR_MATERIALS[structural_system]
    insul_mat = lib.INSULATION_MATERIALS[insulation_key]
    base = _base_system(structural_system)
    struct_t = _sample_discrete(rng, cfg.FLOOR_STRUCTURAL_T_DISCRETE[base])
    if rng.random() < cfg.FLOOR_INSULATION_PROB_GROUND:
        insul_t = _sample_discrete(rng, cfg.FLOOR_INSULATION_T_DISCRETE[insulation_key])
        return OpaqueConstructionSpec(
            layers=[_make_layer(insul_mat, insul_t), _make_layer(struct_mat, struct_t)]
        )
    return OpaqueConstructionSpec(layers=[_make_layer(struct_mat, struct_t)])


def _build_exposed_floor_construction(
    structural_system: str, insulation_key: str, rng: random.Random
) -> OpaqueConstructionSpec:
    struct_mat = lib.STRUCTURAL_FLOOR_MATERIALS[structural_system]
    insul_mat = lib.INSULATION_MATERIALS[insulation_key]
    base = _base_system(structural_system)
    struct_t = _sample_discrete(rng, cfg.FLOOR_STRUCTURAL_T_DISCRETE[base])
    insul_t = _sample_discrete(rng, cfg.FLOOR_INSULATION_T_DISCRETE[insulation_key])
    return OpaqueConstructionSpec(
        layers=[_make_layer(insul_mat, insul_t), _make_layer(struct_mat, struct_t)]
    )


def _build_adiabatic_floor_construction(
    structural_system: str, rng: random.Random
) -> OpaqueConstructionSpec:
    struct_mat = lib.STRUCTURAL_FLOOR_MATERIALS[structural_system]
    base = _base_system(structural_system)
    struct_t = _sample_discrete(rng, cfg.FLOOR_STRUCTURAL_T_DISCRETE[base])
    return OpaqueConstructionSpec(layers=[_make_layer(struct_mat, struct_t)])


def _build_inter_floor_construction(
    structural_system: str, insulation_key: str, rng: random.Random, add_insulation: bool
) -> OpaqueConstructionSpec:
    struct_mat = lib.STRUCTURAL_FLOOR_MATERIALS[structural_system]
    base = _base_system(structural_system)
    struct_t = _sample_discrete(rng, cfg.FLOOR_STRUCTURAL_T_DISCRETE[base])
    if add_insulation:
        insul_mat = lib.INSULATION_MATERIALS[insulation_key]
        insul_t = _sample_discrete(rng, cfg.INTER_SLAB_INSULATION_T_DISCRETE[insulation_key])
        return OpaqueConstructionSpec(
            layers=[_make_layer(insul_mat, insul_t), _make_layer(struct_mat, struct_t)]
        )
    return OpaqueConstructionSpec(layers=[_make_layer(struct_mat, struct_t)])


def _build_interior_wall_construction(
    structural_system: str, rng: random.Random
) -> OpaqueConstructionSpec:
    mat_key = lib.INTERIOR_WALL_MAP[structural_system]
    mat = lib.INTERIOR_WALL_MATERIALS[mat_key]
    struct_t = _sample_discrete(rng, cfg.INTERIOR_WALL_T_DISCRETE[mat_key])
    return OpaqueConstructionSpec(layers=[_make_layer(mat, struct_t)])


_FACE_WALL = "Wall"
_FACE_FLOOR = "Floor"
_FACE_ROOF = "RoofCeiling"
_BC_OUTDOORS = "Outdoors"
_BC_SURFACE = "Surface"
_BC_GROUND = "Ground"
_BC_ADIABAT = "Adiabatic"


def _dispatch_face_construction(
    face: FaceModel, construction_pool: dict
) -> Union[OpaqueConstructionSpec, AirBoundaryConstructionSpec]:
    ft = face.face_type
    bc = face.bc_type
    if ft == _FACE_WALL:
        return (
            construction_pool["interior_wall"]
            if bc == _BC_SURFACE
            else construction_pool["exterior_wall"]
        )
    if ft == _FACE_ROOF:
        if bc == _BC_OUTDOORS:
            return construction_pool["outdoor_roof"]
        if bc == _BC_ADIABAT:
            return construction_pool["adiabatic_ceiling"]
        if bc == _BC_SURFACE:
            return construction_pool["inter_floor_slab"]
        logger.warning(
            f"[dispatch] RoofCeiling with unexpected BC={bc!r} — fallback to adiabatic_ceiling"
        )
        return construction_pool["adiabatic_ceiling"]
    if ft == _FACE_FLOOR:
        if bc == _BC_GROUND:
            return construction_pool["ground_floor"]
        if bc == _BC_OUTDOORS:
            return construction_pool["exposed_floor"]
        if bc == _BC_ADIABAT:
            return construction_pool["adiabatic_floor"]
        if bc == _BC_SURFACE:
            return construction_pool["inter_floor_slab"]
        logger.warning(f"[dispatch] Floor with unexpected BC={bc!r} — fallback to adiabatic_floor")
        return construction_pool["adiabatic_floor"]
    logger.warning(f"[dispatch] Unknown face_type={ft!r} — fallback to interior_wall")
    return construction_pool["interior_wall"]


def _compute_u_values(pool: dict, glazing_system: WindowConstructionSpec) -> dict[str, float]:
    result: dict[str, float] = {}
    if "exterior_wall" in pool:
        result["ext_wall"] = pool["exterior_wall"].u_value(cfg.R_SI_WALL, cfg.R_SE_WALL)
    if "outdoor_roof" in pool:
        result["roof"] = pool["outdoor_roof"].u_value(cfg.R_SI_ROOF, cfg.R_SE_ROOF)
    if "ground_floor" in pool:
        result["floor"] = pool["ground_floor"].u_value(cfg.R_SI_FLOOR, cfg.R_SE_FLOOR)
    if "inter_floor_slab" in pool:
        slab = pool["inter_floor_slab"]
        if hasattr(slab, "u_value"):
            result["inter_floor"] = round(slab.u_value(cfg.R_SI_FLOOR, cfg.R_SE_FLOOR), 4)
    result["window"] = glazing_system.u_value
    return result


def _check_u_values(u_values: dict) -> list[str]:
    warnings = []
    bounds = {
        "ext_wall": cfg.U_VALUE_WALL_RANGE,
        "roof": cfg.U_VALUE_ROOF_RANGE,
        "floor": cfg.U_VALUE_FLOOR_RANGE,
        "window": cfg.U_VALUE_WINDOW_RANGE,
    }
    labels = {"ext_wall": "Exterior wall", "roof": "Roof", "floor": "Floor", "window": "Window"}
    for key, (lo, hi) in bounds.items():
        u = u_values.get(key)
        if u is not None and (not lo <= u <= hi):
            warnings.append(f"{labels[key]} U={u:.3f} W/m²K outside reference range [{lo}, {hi}]")
    return warnings


def assign_materials(
    building_model: BuildingModel, seed: Optional[int] = None, overrides=None
) -> MaterializedBuildingModel:
    """Assign sampled constructions and material properties to the building geometry."""
    rng = random.Random(seed)
    total_area = building_model.total_floor_area
    if overrides and overrides.structural_system:
        structural_system = overrides.structural_system
    else:
        structural_system = lib.sample_structural_system(rng, total_area)
    if overrides and overrides.insulation_key:
        insulation_key = overrides.insulation_key
    else:
        insulation_key = lib.sample_insulation_key(rng)
    if overrides and overrides.glazing_system_name:
        glazing_system = lib.GLAZING_SYSTEMS[overrides.glazing_system_name]
    else:
        glazing_system = lib.sample_glazing_system(rng)
    if overrides and overrides.context_scenario:
        context_scenario = overrides.context_scenario
    else:
        context_scenario = lib.sample_context_scenario(rng)
    if overrides and overrides.svf_wall is not None and (overrides.svf_roof is not None):
        svf_wall, svf_roof = (overrides.svf_wall, overrides.svf_roof)
    else:
        svf_wall, svf_roof = lib.sample_svf(rng, context_scenario)
    if overrides and overrides.has_inter_slab_insulation is not None:
        has_inter_slab_insulation = overrides.has_inter_slab_insulation
    else:
        has_inter_slab_insulation = rng.random() < cfg.INTER_SLAB_INSULATION_PROB
    if overrides and overrides.out_shade_material:
        out_shade_const = lib.OUT_SHADE_CONSTRUCTIONS[overrides.out_shade_material]
    else:
        out_shade_const = lib.sample_out_shade_construction(rng)
    in_shade_const = lib.IN_SHADE_CONSTRUCTION
    logger.info(
        f"[assign_materials] System={structural_system} | Insul={insulation_key} | Glazing={glazing_system.name} | Context={context_scenario} OutShade={out_shade_const.name} (refl={out_shade_const.solar_reflectance}) | InShade={in_shade_const.name}(SVF wall={svf_wall}, roof={svf_roof}) | InterSlab insul={('YES' if has_inter_slab_insulation else 'NO')} |InteriorWall={cfg.INTERIOR_WALL_MODE}"
    )
    if cfg.INTERIOR_WALL_MODE == "real":
        _interior_wall = _build_interior_wall_construction(structural_system, rng)
    else:
        _interior_wall = lib.make_air_boundary_construction()
    construction_pool: dict = {
        "exterior_wall": _build_ext_wall_construction(
            structural_system, insulation_key, rng, overrides
        ),
        "interior_wall": _interior_wall,
        "outdoor_roof": _build_outdoor_roof_construction(structural_system, insulation_key, rng),
        "adiabatic_ceiling": _build_adiabatic_ceiling_construction(structural_system, rng),
        "ground_floor": _build_ground_floor_construction(structural_system, insulation_key, rng),
        "exposed_floor": _build_exposed_floor_construction(structural_system, insulation_key, rng),
        "adiabatic_floor": _build_adiabatic_floor_construction(structural_system, rng),
        "inter_floor_slab": _build_inter_floor_construction(
            structural_system, insulation_key, rng, add_insulation=has_inter_slab_insulation
        ),
    }
    face_constructions: dict = {}
    aperture_constructions: dict = {}
    shaded_window_cache = {}
    shaded_window_cache = {}
    for room in building_model.rooms:
        for face in room.faces:
            face_constructions[face.identifier] = _dispatch_face_construction(
                face, construction_pool
            )
            for apt in face.apertures:
                if apt.identifier.endswith("_shaded"):
                    base_name = glazing_system.name
                    if base_name not in shaded_window_cache:
                        shade_glazing = lib.make_equivalent_shade_glazing(in_shade_const)
                        air_gap = lib.GAP_AIR_12MM
                        new_layers = list(glazing_system.layers) + [air_gap, shade_glazing]
                        effective_shgc = glazing_system.shgc * (
                            1 - in_shade_const.solar_transmittance
                        )
                        shaded_const = WindowConstructionSpec(
                            name=f"{glazing_system.name}_shaded",
                            layers=new_layers,
                            u_value=glazing_system.u_value,
                            shgc=round(effective_shgc, 3),
                        )
                        shaded_window_cache[base_name] = shaded_const
                    aperture_constructions[apt.identifier] = shaded_window_cache[base_name]
                else:
                    aperture_constructions[apt.identifier] = glazing_system
    window_pool = {}
    for wconst in aperture_constructions.values():
        window_pool.setdefault(wconst.identifier, wconst)
    opaque_pool: dict = {}
    for const in face_constructions.values():
        if isinstance(const, OpaqueConstructionSpec):
            opaque_pool.setdefault(const.identifier, const)
    window_pool: dict = {}
    for wconst in aperture_constructions.values():
        window_pool.setdefault(wconst.identifier, wconst)
    u_values = _compute_u_values(construction_pool, glazing_system)
    warnings = _check_u_values(u_values)
    if warnings:
        for w in warnings:
            logger.warning(f"[assign_materials] WARNING: {w}")
    logger.info(
        f"[assign_materials] U_wall={u_values.get('ext_wall', '—'):.3f}  U_roof={u_values.get('roof', '—'):.3f}  U_inter={u_values.get('inter_floor', '—')}  U_win={u_values.get('window', '—'):.2f}  W/m²K"
    )

    def _layers_info(const: OpaqueConstructionSpec) -> list[dict]:
        return [
            {
                "material": l.material.name,
                "thickness_m": l.thickness,
                "conductivity": l.material.conductivity,
                "density": l.material.density,
                "specific_heat": l.material.specific_heat,
            }
            for l in const.layers
        ]

    constructions_meta = {}
    for name, const in construction_pool.items():
        if not isinstance(const, OpaqueConstructionSpec):
            continue
        if const.identifier in opaque_pool:
            constructions_meta[name] = {
                "identifier": const.identifier,
                "total_thickness_m": const.total_thickness,
                "has_insulation": has_inter_slab_insulation if name == "inter_floor_slab" else None,
                "layers": _layers_info(const),
            }
    metadata = {
        "building": {
            "n_floors": building_model.n_floors,
            "building_type": building_model.building_type,
            "floor_height": building_model.floor_height,
        },
        "structural_system": structural_system,
        "insulation_key": insulation_key,
        "glazing_system_name": glazing_system.name,
        "has_inter_slab_insulation": has_inter_slab_insulation,
        "interior_wall_mode": cfg.INTERIOR_WALL_MODE,
        "context": {"scenario": context_scenario, "svf_wall": svf_wall, "svf_roof": svf_roof},
        "u_values": u_values,
        "warnings": warnings,
        "constructions": constructions_meta,
        "glazing": {
            "identifier": glazing_system.identifier,
            "u_value": glazing_system.u_value,
            "shgc": glazing_system.shgc,
            "n_panes": sum(
                (1 for lyr in glazing_system.layers if isinstance(lyr, lib.GlazingSpec))
            ),
        },
    }
    material_spec = BuildingMaterialSpec(
        structural_system=structural_system,
        insulation_key=insulation_key,
        glazing_system_name=glazing_system.name,
        face_constructions=face_constructions,
        aperture_constructions=aperture_constructions,
        opaque_pool=opaque_pool,
        window_pool=window_pool,
        u_values=u_values,
        metadata=metadata,
        out_shade_construction=out_shade_const,
        in_shade_construction=in_shade_const,
    )
    return MaterializedBuildingModel(geometry=building_model, material_spec=material_spec)
