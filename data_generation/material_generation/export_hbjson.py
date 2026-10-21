"""Parameterized building export hbjson.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

from __future__ import annotations
import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)
from .assign_materials import MaterializedBuildingModel
from ..geometric_generation.add_windows import FaceModel, RoomModel, ApertureGeom, OutShadingGeom
from .material_library import (
    LayerSpec,
    OpaqueConstructionSpec,
    WindowConstructionSpec,
    GlazingSpec,
    GasSpec,
    AirBoundaryConstructionSpec,
    OutShadeConstructionSpec,
    InShadeConstructionSpec,
    ShadeMaterialSpec,
)
from . import material_config as cfg

Vec3 = Tuple[float, float, float]
_SPATIAL_LINK_VERSION = 1


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normalize(v: Vec3) -> Vec3:
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if mag < 1e-10:
        return (0.0, 0.0, 1.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _compute_plane(vertices: List[Vec3]) -> dict:
    o = vertices[0]
    x = _normalize(_sub(vertices[1], vertices[0]))
    e2 = _sub(vertices[2], vertices[0])
    n = _normalize(_cross(_sub(vertices[1], vertices[0]), e2))
    return {"type": "Plane", "o": list(o), "x": list(x), "n": list(n)}


def _face3d(vertices: List[Vec3]) -> dict:
    return {
        "type": "Face3D",
        "boundary": [list(v) for v in vertices],
        "plane": _compute_plane(vertices),
    }


def _polygon_area_3d(vertices: List[Vec3]) -> float:
    n = len(vertices)
    if n < 3:
        return 0.0
    nx = ny = nz = 0.0
    for i in range(n):
        v0, v1 = (vertices[i], vertices[(i + 1) % n])
        nx += v0[1] * v1[2] - v0[2] * v1[1]
        ny += v0[2] * v1[0] - v0[0] * v1[2]
        nz += v0[0] * v1[1] - v0[1] * v1[0]
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def _room_infil_fpa(room: RoomModel, ach: float) -> float:
    z_min = float("inf")
    z_max = -float("inf")
    floor_area = 0.0
    ext_area = 0.0
    for face in room.faces:
        for v in face.vertices:
            z_min = min(z_min, v[2])
            z_max = max(z_max, v[2])
        area = _polygon_area_3d(face.vertices)
        if face.bc_type == "Outdoors":
            ext_area += area
        if face.face_type == "Floor":
            floor_area += area
    floor_h = max(z_max - z_min, 0.01)
    volume = floor_area * floor_h if floor_area > 1e-06 else 0.0
    if volume < 1e-06:
        logger.warning(f"[_room_infil_fpa] {room.identifier}: degenerate volume, using fpa=0.0003")
        return 0.0003
    if ext_area >= 1e-06:
        return ach / 3600.0 * (volume / ext_area)
    elif floor_area >= 1e-06:
        logger.info(
            f"[_room_infil_fpa] {room.identifier}: interior room, encoding ACH via floor_area"
        )
        return ach / 3600.0 * (volume / floor_area)
    else:
        logger.warning(f"[_room_infil_fpa] {room.identifier}: no valid reference area, fpa=0.0003")
        return 0.0003


def _vertex_key(v: Vec3, decimals: int = 3) -> Tuple:
    return (round(v[0], decimals), round(v[1], decimals), round(v[2], decimals))


def _face_vertex_set(vertices: List[Vec3]) -> frozenset:
    return frozenset((_vertex_key(v) for v in vertices))


def _adj_face_to_room_id(adj_face_id: str) -> str:
    return re.sub("_f\\d{2}$", "", adj_face_id)


_SCHEDULE_TYPE_LIMIT = {
    "type": "ScheduleTypeLimit",
    "identifier": "Fractional",
    "lower_limit": 0.0,
    "upper_limit": 1.0,
    "numeric_type": "Continuous",
    "unit_type": "Dimensionless",
}
_ALWAYS_ON_SCHEDULE_ID = "AlwaysOn"
_ALWAYS_ON_DAY_ID = "AlwaysOn_Day"
_ALWAYS_ON_SCHEDULE = {
    "type": "ScheduleRulesetAbridged",
    "identifier": _ALWAYS_ON_SCHEDULE_ID,
    "schedule_type_limit": "Fractional",
    "default_day_schedule": _ALWAYS_ON_DAY_ID,
    "day_schedules": [
        {
            "type": "ScheduleDay",
            "identifier": _ALWAYS_ON_DAY_ID,
            "interpolate": False,
            "values": [1.0],
            "times": [[0, 0]],
        }
    ],
}
_PROGRAM_TYPE_ID = "ProgramType_Unoccupied"
_SCHEDULE_TYPE_LIMIT_ANY_NUMBER = {
    "type": "ScheduleTypeLimit",
    "identifier": "AnyNumber",
    "numeric_type": "Continuous",
    "unit_type": "Dimensionless",
}
_SCHEDULE_TYPE_LIMIT_ACTIVITY_LEVEL = {
    "type": "ScheduleTypeLimit",
    "identifier": "ActivityLevel",
    "lower_limit": 0.0,
    "numeric_type": "Continuous",
    "unit_type": "ActivityLevel",
}
_ACTIVITY_LEVEL_SCHEDULE_ID = "ActivityLevel_120W"
_ACTIVITY_LEVEL_DAY_ID = "ActivityLevel_120W_Day"
_ACTIVITY_LEVEL_SCHEDULE = {
    "type": "ScheduleRulesetAbridged",
    "identifier": _ACTIVITY_LEVEL_SCHEDULE_ID,
    "schedule_type_limit": "ActivityLevel",
    "default_day_schedule": _ACTIVITY_LEVEL_DAY_ID,
    "day_schedules": [
        {
            "type": "ScheduleDay",
            "identifier": _ACTIVITY_LEVEL_DAY_ID,
            "interpolate": False,
            "values": [120.0],
            "times": [[0, 0]],
        }
    ],
}
_IN_SHADE_WIN_MAT_ID = "WinShadeMat__white_roller_blind"


def _make_in_shade_win_mat(spec: InShadeConstructionSpec) -> dict:
    solar_abs = round(1.0 - spec.solar_transmittance - spec.solar_reflectance, 4)
    return {
        "type": "EnergyWindowMaterialShade",
        "identifier": spec.identifier.replace("ShadeConst", "WinShadeMat"),
        "solar_transmittance": spec.solar_transmittance,
        "solar_reflectance": spec.solar_reflectance,
        "visible_transmittance": getattr(spec, "visible_transmittance", 0.35),
        "visible_reflectance": getattr(spec, "visible_reflectance", 0.55),
        "infrared_hemispherical_transmittance": 0.0,
        "infrared_hemispherical_emissivity": 0.9,
        "thickness": 0.005,
        "conductivity": 0.05,
        "distance_to_glass": 0.05,
        "top_opening_multiplier": 0.5,
        "bottom_opening_multiplier": 0.5,
        "left_side_opening_multiplier": 0.5,
        "right_side_opening_multiplier": 0.5,
        "airflow_permeability": 0.0,
    }


@dataclass
class _FaceRef:
    """Represent face ref and its associated geometry or physical configuration."""

    face_id: str
    room_id: str
    vset: frozenset


def _serialize_energy_material(layer: LayerSpec) -> dict:
    m = layer.material
    return {
        "type": "EnergyMaterial",
        "identifier": layer.identifier,
        "display_name": layer.identifier,
        "roughness": m.roughness,
        "thickness": round(layer.thickness, 4),
        "conductivity": m.conductivity,
        "density": float(m.density),
        "specific_heat": float(m.specific_heat),
        "thermal_absorptance": m.thermal_absorptance,
        "solar_absorptance": m.solar_absorptance,
        "visible_absorptance": m.visible_absorptance,
    }


def _serialize_glazing(g: GlazingSpec) -> dict:
    return {
        "type": "EnergyWindowMaterialGlazing",
        "identifier": g.name,
        "thickness": g.thickness,
        "solar_transmittance": g.solar_transmittance,
        "solar_reflectance": g.solar_reflectance,
        "solar_reflectance_back": g.solar_reflectance_back,
        "visible_transmittance": g.visible_transmittance,
        "visible_reflectance": round(g.visible_reflectance, 4),
        "visible_reflectance_back": round(g.visible_reflectance_back, 4),
        "infrared_transmittance": g.infrared_transmittance,
        "emissivity": g.emissivity,
        "emissivity_back": g.emissivity_back,
        "conductivity": g.conductivity,
        "dirt_correction": 1.0,
        "solar_diffusing": False,
        "density": g.density,
        "specific_heat": g.specific_heat,
    }


def _serialize_gas(g: GasSpec) -> dict:
    return {
        "type": "EnergyWindowMaterialGas",
        "identifier": g.name,
        "gas_type": g.gas_type,
        "thickness": g.thickness,
        "conductivity": g.conductivity,
        "density": g.density,
        "specific_heat": g.specific_heat,
    }


def _serialize_shade_material(mat: ShadeMaterialSpec) -> dict:
    return {
        "type": "EnergyWindowMaterialShade",
        "identifier": mat.identifier,
        "solar_transmittance": mat.solar_transmittance,
        "solar_reflectance": mat.solar_reflectance,
        "visible_transmittance": mat.visible_transmittance,
        "visible_reflectance": mat.visible_reflectance,
        "infrared_hemispherical_transmittance": 0.0,
        "infrared_hemispherical_emissivity": mat.emissivity,
        "thickness": mat.thickness,
        "conductivity": mat.conductivity,
        "distance_to_glass": mat.distance_to_glass,
        "top_opening_multiplier": mat.top_opening_multiplier,
        "bottom_opening_multiplier": mat.bottom_opening_multiplier,
        "left_side_opening_multiplier": mat.left_side_opening_multiplier,
        "right_side_opening_multiplier": mat.right_side_opening_multiplier,
        "airflow_permeability": mat.airflow_permeability,
    }


def _collect_all_materials(mat_spec) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    opaque_mats: Dict[str, dict] = {}
    window_mats: Dict[str, dict] = {}
    for const in mat_spec.opaque_pool.values():
        if isinstance(const, AirBoundaryConstructionSpec):
            continue
        for layer in const.layers:
            if layer.identifier not in opaque_mats:
                opaque_mats[layer.identifier] = _serialize_energy_material(layer)
    for wconst in mat_spec.window_pool.values():
        for lyr in wconst.layers:
            if isinstance(lyr, GlazingSpec) and lyr.name not in window_mats:
                window_mats[lyr.name] = _serialize_glazing(lyr)
            elif isinstance(lyr, GasSpec) and lyr.name not in window_mats:
                window_mats[lyr.name] = _serialize_gas(lyr)
            elif isinstance(lyr, ShadeMaterialSpec) and lyr.identifier not in window_mats:
                window_mats[lyr.identifier] = _serialize_shade_material(lyr)
    return (opaque_mats, window_mats)


def _serialize_opaque_construction(const: OpaqueConstructionSpec) -> dict:
    return {
        "type": "OpaqueConstructionAbridged",
        "identifier": const.identifier,
        "materials": [layer.identifier for layer in const.layers],
    }


def _serialize_air_boundary_construction(const: AirBoundaryConstructionSpec) -> dict:
    return {
        "type": "AirBoundaryConstructionAbridged",
        "identifier": const.identifier,
        "air_mixing_per_area": cfg.AIR_MIX,
        "air_mixing_schedule": _ALWAYS_ON_SCHEDULE_ID,
    }


def _serialize_window_construction(wconst: WindowConstructionSpec) -> dict:
    return {
        "type": "WindowConstructionAbridged",
        "identifier": wconst.identifier,
        "materials": [lyr.identifier for lyr in wconst.layers],
    }


def _make_outdoors_bc(svf: float) -> dict:
    return {
        "type": "Outdoors",
        "sun_exposure": True,
        "wind_exposure": True,
        "view_factor": round(float(svf), 4),
    }


_BC_ADIABATIC_DICT = {"type": "Adiabatic"}
_BC_GROUND_DICT = {"type": "Ground"}


def _serialize_bc(face: FaceModel, svf_wall: float, svf_roof: float) -> dict:
    bc = face.bc_type
    ft = face.face_type
    if bc == "Outdoors":
        svf = svf_roof if ft == "RoofCeiling" else svf_wall
        return _make_outdoors_bc(svf)
    if bc == "Adiabatic":
        return _BC_ADIABATIC_DICT
    if bc == "Ground":
        return _BC_GROUND_DICT
    if bc == "Surface":
        if face.adjacent_to:
            adj_room_id = _adj_face_to_room_id(face.adjacent_to)
            return {
                "type": "Surface",
                "boundary_condition_objects": [face.adjacent_to, adj_room_id],
            }
        else:
            logger.warning(
                f"[_serialize_bc] Face {face.identifier} has bc=Surface but missing adjacent_to — using self-reference as fallback."
            )
            return {
                "type": "Surface",
                "boundary_condition_objects": [
                    face.identifier,
                    _adj_face_to_room_id(face.identifier),
                ],
            }
    logger.warning(
        f"[_serialize_bc] Unknown bc={bc!r} on face {face.identifier} — fallback to Outdoors."
    )
    return _make_outdoors_bc(svf_wall)


def _serialize_aperture(apt: ApertureGeom, construction_id: str, svf_wall: float) -> dict:
    d = {
        "type": "Aperture",
        "identifier": apt.identifier,
        "display_name": apt.identifier,
        "is_operable": False,
        "boundary_condition": _make_outdoors_bc(svf_wall),
        "geometry": _face3d(apt.vertices),
        "user_data": {
            "spatial_link_version": _SPATIAL_LINK_VERSION,
            "owner_zone_id": apt.owner_zone_id,
            "host_face_id": apt.host_face_id,
            "aperture_group_id": apt.aperture_group_id,
            "relation_type": "zone_aperture",
        },
        "properties": {
            "type": "AperturePropertiesAbridged",
            "radiance": {"type": "ApertureRadiancePropertiesAbridged"},
            "energy": {"type": "ApertureEnergyPropertiesAbridged", "construction": construction_id},
        },
    }
    return d


def _serialize_out_shade_construction(spec: OutShadeConstructionSpec) -> dict:
    return {
        "type": "ShadeConstruction",
        "identifier": spec.identifier,
        "solar_reflectance": spec.solar_reflectance,
        "visible_reflectance": spec.visible_reflectance,
        "is_specular": spec.is_specular,
    }


def _serialize_out_shade(shade: OutShadingGeom, construction_id: str) -> dict:
    return {
        "type": "Shade",
        "identifier": shade.identifier,
        "display_name": shade.identifier,
        "geometry": _face3d(shade.vertices),
        "user_data": {
            "spatial_link_version": _SPATIAL_LINK_VERSION,
            "owner_zone_id": shade.owner_zone_id,
            "host_face_id": shade.host_face_id,
            "host_aperture_group_id": shade.host_aperture_group_id,
            "relation_type": "attached_exterior_shade",
        },
        "properties": {
            "type": "ShadePropertiesAbridged",
            "radiance": {"type": "ShadeRadiancePropertiesAbridged"},
            "energy": {"type": "ShadeEnergyPropertiesAbridged", "construction": construction_id},
        },
    }


def _serialize_face(
    face: FaceModel,
    construction_id: str,
    mat_spec,
    svf_wall: float,
    svf_roof: float,
    out_shade_const_id: str,
) -> Tuple[dict, List[dict]]:
    apertures_out = []
    for apt in face.apertures:
        wconst = mat_spec.aperture_constructions.get(apt.identifier)
        const_id = wconst.identifier if wconst else "WindowConst__double_low_e_high_gain"
        apertures_out.append(_serialize_aperture(apt, const_id, svf_wall))
    out_shade_dicts = [_serialize_out_shade(sh, out_shade_const_id) for sh in face.out_shading]
    is_air_boundary = construction_id.startswith("AirBoundaryConst")
    actual_face_type = "AirBoundary" if is_air_boundary else face.face_type
    face_user_data = {
        "spatial_link_version": _SPATIAL_LINK_VERSION,
        "owner_zone_id": face.owner_zone_id,
        "relation_type": "zone_boundary",
    }
    if face.adjacent_to:
        face_user_data.update(
            {
                "adjacent_face_id": face.adjacent_to,
                "adjacent_zone_id": _adj_face_to_room_id(face.adjacent_to),
            }
        )
    face_dict = {
        "type": "Face",
        "identifier": face.identifier,
        "display_name": face.identifier,
        "face_type": actual_face_type,
        "boundary_condition": _serialize_bc(face, svf_wall, svf_roof),
        "geometry": _face3d(face.vertices),
        "apertures": apertures_out,
        "user_data": face_user_data,
        "properties": {
            "type": "FacePropertiesAbridged",
            "radiance": {"type": "FaceRadiancePropertiesAbridged"},
            "energy": {"type": "FaceEnergyPropertiesAbridged", "construction": construction_id},
        },
    }
    return (face_dict, out_shade_dicts)


def _make_room_schedule(
    ach_day: float, ach_night: float, day_start: int, night_start: int, room_id: str
) -> tuple[str, dict]:
    sched_id = f"ACH_DayNight_{room_id}"
    day_sched_id = f"{sched_id}_Day"
    if ach_day > 1e-06:
        night_frac = ach_night / ach_day
        night_frac = max(0.0, min(1.0, night_frac))
    else:
        night_frac = 0.0
    day_frac = 1.0
    sched_dict = {
        "type": "ScheduleRulesetAbridged",
        "identifier": sched_id,
        "schedule_type_limit": "Fractional",
        "default_day_schedule": day_sched_id,
        "day_schedules": [
            {
                "type": "ScheduleDay",
                "identifier": day_sched_id,
                "interpolate": False,
                "values": [night_frac, day_frac, night_frac],
                "times": [[0, 0], [day_start, 0], [night_start, 0]],
            }
        ],
    }
    return (sched_id, sched_dict)


def _make_feature_schedule_24h(
    values_24h: list[float], room_id: str, feature: str
) -> tuple[str, dict]:
    sched_id = f"{feature}_Sched_{room_id}"
    day_sched_id = f"{feature}_Sched_{room_id}_Day"
    times = [[h, 0] for h in range(24)]
    values = [round(float(v), 6) for v in values_24h]
    sched_dict = {
        "type": "ScheduleRulesetAbridged",
        "identifier": sched_id,
        "schedule_type_limit": "Fractional",
        "default_day_schedule": day_sched_id,
        "day_schedules": [
            {
                "type": "ScheduleDay",
                "identifier": day_sched_id,
                "interpolate": False,
                "values": values,
                "times": times,
            }
        ],
    }
    return (sched_id, sched_dict)


def _make_electric_equipment(room_id: str, watts_m2: float, sched_id: str) -> dict:
    return {
        "type": "ElectricEquipmentAbridged",
        "identifier": f"Equip_{room_id}",
        "watts_per_area": round(watts_m2, 4),
        "schedule": sched_id,
        "radiant_fraction": cfg.EQUIP_FRACTION_RADIANT,
        "latent_fraction": cfg.EQUIP_FRACTION_LATENT,
        "lost_fraction": 0.0,
    }


def _make_people(room_id: str, people_per_area: float, occ_sched_id: str) -> dict:
    return {
        "type": "PeopleAbridged",
        "identifier": f"People_{room_id}",
        "people_per_area": round(people_per_area, 6),
        "occupancy_schedule": occ_sched_id,
        "activity_schedule": _ACTIVITY_LEVEL_SCHEDULE_ID,
        "radiant_fraction": cfg.OCC_FRACTION_RADIANT,
    }


def _serialize_room(
    room: RoomModel,
    mat_spec,
    svf_wall: float,
    svf_roof: float,
    ach_day: float,
    ach_night: float,
    schedule_id: str,
    out_shade_const_id: str,
    equip_watts_m2: float = 0.0,
    equip_sched_id: Optional[str] = None,
    occ_people_m2: float = 0.0,
    occ_sched_id: Optional[str] = None,
) -> dict:
    faces_out = []
    out_shades = []
    for face in room.faces:
        const = mat_spec.face_constructions.get(face.identifier)
        if const is None:
            logger.error(
                f"Missing construction for face {face.identifier} in room {room.identifier}"
            )
            continue
        const_id = const.identifier
        face_dict, face_out_shades = _serialize_face(
            face, const_id, mat_spec, svf_wall, svf_roof, out_shade_const_id=out_shade_const_id
        )
        faces_out.append(face_dict)
        out_shades.extend(face_out_shades)
    fpa = _room_infil_fpa(room, ach_day)
    infiltration_dict = {
        "type": "InfiltrationAbridged",
        "identifier": f"Infil_{room.identifier}",
        "flow_per_exterior_area": round(fpa, 8),
        "schedule": schedule_id,
    }
    energy_props: dict = {
        "type": "RoomEnergyPropertiesAbridged",
        "program_type": _PROGRAM_TYPE_ID,
        "infiltration": infiltration_dict,
    }
    if equip_watts_m2 > 1e-06 and equip_sched_id is not None:
        energy_props["electric_equipment"] = _make_electric_equipment(
            room.identifier, equip_watts_m2, equip_sched_id
        )
    if occ_people_m2 > 1e-06 and occ_sched_id is not None:
        energy_props["people"] = _make_people(room.identifier, occ_people_m2, occ_sched_id)
    return {
        "type": "Room",
        "identifier": room.identifier,
        "display_name": room.identifier,
        "faces": faces_out,
        "outdoor_shades": out_shades,
        "user_data": {
            "spatial_link_version": _SPATIAL_LINK_VERSION,
            "zone_id": room.identifier,
            "relation_type": "thermal_zone",
        },
        "properties": {
            "type": "RoomPropertiesAbridged",
            "doe2": {"type": "RoomDoe2Properties"},
            "radiance": {"type": "RoomRadiancePropertiesAbridged"},
            "energy": energy_props,
        },
    }


def _make_program_type() -> dict:
    return {"type": "ProgramTypeAbridged", "identifier": _PROGRAM_TYPE_ID}


def _build_energy_properties(
    opaque_mats: Dict[str, dict],
    window_mats: Dict[str, dict],
    opaque_consts: Dict[str, OpaqueConstructionSpec],
    window_consts: Dict[str, WindowConstructionSpec],
    all_schedules: List[dict],
    out_shade_construction: OutShadeConstructionSpec,
) -> dict:
    all_materials = list(opaque_mats.values()) + list(window_mats.values())
    all_constructions = []
    for c in opaque_consts.values():
        if isinstance(c, AirBoundaryConstructionSpec):
            all_constructions.append(_serialize_air_boundary_construction(c))
        else:
            all_constructions.append(_serialize_opaque_construction(c))
    all_constructions += [_serialize_window_construction(w) for w in window_consts.values()]
    all_constructions.append(_serialize_out_shade_construction(out_shade_construction))
    return {
        "type": "ModelEnergyProperties",
        "materials": all_materials,
        "constructions": all_constructions,
        "construction_sets": [],
        "program_types": [_make_program_type()],
        "hvacs": [],
        "shws": [],
        "schedules": all_schedules,
        "schedule_type_limits": [
            _SCHEDULE_TYPE_LIMIT,
            _SCHEDULE_TYPE_LIMIT_ANY_NUMBER,
            _SCHEDULE_TYPE_LIMIT_ACTIVITY_LEVEL,
        ],
        "ventilation_simulation_control": {
            "type": "VentilationSimulationControl",
            "vent_control_type": "SingleZone",
            "building_type": "LowRise",
            "reference_temperature": 20.0,
            "reference_pressure": 101325.0,
            "reference_humidity_ratio": 0.0,
            "long_axis_angle": 0.0,
            "aspect_ratio": 1.0,
        },
    }


def _short_hash(text: str, length: int = 8) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:length]


def _build_hbjson(mat_model: MaterializedBuildingModel, identifier: Optional[str]) -> dict:
    building = mat_model.geometry
    mat_spec = mat_model.material_spec
    out_shade_const_id = mat_spec.out_shade_construction.identifier
    if identifier is None:
        bld_meta = mat_spec.metadata.get("building", {})
        uid = _short_hash(
            f"{building.shape_type}_{bld_meta.get('n_floors', 1)}F_{building.total_floor_area:.1f}"
            + str(uuid.uuid4())
        )
        identifier = f"building_{uid}"
    context = mat_spec.metadata.get("context", {})
    svf_wall = float(context.get("svf_wall", 1.0))
    svf_roof = float(context.get("svf_roof", 1.0))
    program = mat_spec.metadata.get("program", {})
    svf_per_floor = {
        int(entry["floor_index"]): float(entry["svf_wall"])
        for entry in program.get("svf_per_floor", [])
    }
    room_ach_day_list = program.get("room_infiltration_ach_day", [])
    room_ach_night_list = program.get("room_infiltration_ach_night", [])
    day_start = int(program.get("day_start_hour", 8))
    night_start = int(program.get("night_start_hour", 22))
    if not room_ach_day_list or len(room_ach_day_list) != len(building.rooms):
        logger.warning(
            "[_build_hbjson] Missing or mismatched room_ach_day list. Using a single infiltration schedule."
        )
        fallback_ach_day = program.get("ach_day", 0.5)
        fallback_ach_night = program.get("ach_night", 0.2)
        room_ach_day_list = [fallback_ach_day] * len(building.rooms)
        room_ach_night_list = [fallback_ach_night] * len(building.rooms)
    scenario_type = program.get("scenario_type", "passive")
    is_occupied = scenario_type.startswith("occupied")
    room_equip_peak_list = program.get("room_equip_peak", [])
    room_equip_sched_list = program.get("room_equip_schedule", [])
    room_occ_peak_list = program.get("room_occ_peak", [])
    room_occ_sched_list = program.get("room_occ_schedule", [])
    all_schedules = [_ALWAYS_ON_SCHEDULE]
    if is_occupied:
        all_schedules.append(_ACTIVITY_LEVEL_SCHEDULE)
    rooms_out = []
    for idx, room in enumerate(building.rooms):
        ach_day = room_ach_day_list[idx]
        ach_night = room_ach_night_list[idx]
        sched_id, sched_dict = _make_room_schedule(
            ach_day, ach_night, day_start, night_start, room.identifier
        )
        all_schedules.append(sched_dict)
        equip_watts_m2 = 0.0
        equip_sched_id = None
        if is_occupied and room_equip_peak_list:
            equip_watts_m2 = room_equip_peak_list[idx]
            if equip_watts_m2 > 1e-06 and room_equip_sched_list:
                raw = room_equip_sched_list[idx]
                norm = [v / equip_watts_m2 for v in raw]
                equip_sched_id, equip_sched_dict = _make_feature_schedule_24h(
                    norm, room.identifier, "Equip"
                )
                all_schedules.append(equip_sched_dict)
        occ_people_m2 = 0.0
        occ_sched_id = None
        if is_occupied and room_occ_peak_list:
            occ_people_m2 = room_occ_peak_list[idx]
            if occ_people_m2 > 1e-06 and room_occ_sched_list:
                raw = room_occ_sched_list[idx]
                norm = [v / occ_people_m2 for v in raw]
                occ_sched_id, occ_sched_dict = _make_feature_schedule_24h(
                    norm, room.identifier, "Occ"
                )
                all_schedules.append(occ_sched_dict)
        room_svf_wall = svf_per_floor.get(room.floor_index, svf_wall)
        room_dict = _serialize_room(
            room,
            mat_spec,
            svf_wall=room_svf_wall,
            svf_roof=svf_roof,
            ach_day=ach_day,
            ach_night=ach_night,
            schedule_id=sched_id,
            out_shade_const_id=out_shade_const_id,
            equip_watts_m2=equip_watts_m2,
            equip_sched_id=equip_sched_id,
            occ_people_m2=occ_people_m2,
            occ_sched_id=occ_sched_id,
        )
        rooms_out.append(room_dict)
    opaque_mats, window_mats = _collect_all_materials(mat_spec)
    energy_props = _build_energy_properties(
        opaque_mats,
        window_mats,
        mat_spec.opaque_pool,
        mat_spec.window_pool,
        all_schedules=all_schedules,
        out_shade_construction=mat_spec.out_shade_construction,
    )
    return {
        "type": "Model",
        "version": "1.58.3",
        "identifier": identifier,
        "display_name": identifier,
        "units": "Meters",
        "tolerance": 0.01,
        "angle_tolerance": 1.0,
        "rooms": rooms_out,
        "properties": {
            "type": "ModelProperties",
            "energy": energy_props,
            "doe2": {"type": "ModelDoe2Properties"},
            "radiance": {"type": "ModelRadianceProperties", "modifier_sets": [], "modifiers": []},
        },
    }


def export_hbjson(
    materialized_model: MaterializedBuildingModel,
    output_dir: str | Path = ".",
    meta_dir: str | Path = ".",
    identifier: Optional[str] = None,
    pretty: bool = False,
    overwrite: bool = True,
) -> Tuple[Path, Path]:
    """Write the building HBJSON and its physical and operational metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    building = materialized_model.geometry
    mat_spec = materialized_model.material_spec
    bld_meta = mat_spec.metadata.get("building", {})
    if identifier is None:
        uid = _short_hash(
            f"{building.shape_type}_{bld_meta.get('n_floors', 1)}F_{building.total_floor_area:.1f}"
            + str(uuid.uuid4())
        )
        identifier = f"building_{uid}"
    hbjson_path = output_dir / f"{identifier}.hbjson"
    meta_path = meta_dir / f"{identifier}_meta.json"
    if not overwrite and (hbjson_path.exists() or meta_path.exists()):
        logger.info(f"[export_hbjson] File exists and overwrite=False, skipping: {identifier}")
        return (hbjson_path, meta_path)
    hbjson_dict = _build_hbjson(materialized_model, identifier)
    with open(hbjson_path, "w", encoding="utf-8") as f:
        json.dump(hbjson_dict, f, indent=2 if pretty else None, ensure_ascii=False)
    metadata_out = {
        "identifier": identifier,
        "geometry": {
            "shape_type": building.shape_type,
            "n_floors": bld_meta.get("n_floors", 1),
            "building_type": bld_meta.get("building_type", "real"),
            "floor_height_m": building.floor_height,
            "floor_bc": building.floor_bc,
            "ceiling_bc": building.ceiling_bc,
            "total_floor_area_m2": building.total_floor_area,
            "n_rooms": building.n_rooms,
            "sill_height_m": building.sill_height,
            "head_height_m": building.head_height,
            "wwr_by_orientation": building.wwr_by_orientation,
            "total_window_area_m2": building.total_window_area,
            "out_shade_depth_by_orientation": getattr(
                building, "out_shade_depth_by_orientation", {}
            ),
            "out_shade_gap_m": getattr(building, "out_shade_gap", 0.1),
        },
        "shading": {
            "out_shade_name": mat_spec.out_shade_construction.name,
            "out_shade_construction_id": mat_spec.out_shade_construction.identifier,
            "out_shade_solar_reflectance": mat_spec.out_shade_construction.solar_reflectance,
            "out_shade_solar_absorptance": mat_spec.out_shade_construction.solar_absorptance,
            "in_shade_construction_id": mat_spec.in_shade_construction.identifier,
            "in_shade_solar_reflectance": mat_spec.in_shade_construction.solar_reflectance,
            "in_shade_solar_transmittance": mat_spec.in_shade_construction.solar_transmittance,
            "in_shade_solar_absorptance": mat_spec.in_shade_construction.solar_absorptance,
        },
        "material": mat_spec.metadata,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, indent=2, ensure_ascii=False)
    hbjson_kb = hbjson_path.stat().st_size / 1024
    prg = mat_spec.metadata.get("program", {})
    context = mat_spec.metadata.get("context", {})
    n_floors = bld_meta.get("n_floors", 1)
    btype = bld_meta.get("building_type", "real")
    room_ach_day = prg.get("room_infiltration_ach_day", [])
    if room_ach_day:
        ach_day_summary = f"min={min(room_ach_day):.2f} avg={sum(room_ach_day) / len(room_ach_day):.2f} max={max(room_ach_day):.2f}"
    else:
        ach_day_summary = "—"
    logger.info(
        f"[export_hbjson] → {hbjson_path.name} ({hbjson_kb:.1f} KB) | floors={n_floors} [{btype}] | rooms={building.n_rooms} | windows={building.total_windows} | SVF wall(f0)={context.get('svf_wall', '—')} roof={context.get('svf_roof', '—')} | ACH_day={ach_day_summary}"
    )
    if mat_spec.metadata.get("warnings"):
        for w in mat_spec.metadata["warnings"]:
            logger.warning(f"[export_hbjson] WARNING: {w}")
    return (hbjson_path, meta_path)
