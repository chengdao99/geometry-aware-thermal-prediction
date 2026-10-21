"""Intersect HBJSON material slabs with a metric voxel grid. Inputs: HBJSON and grid settings. Outputs: Physical voxel points and Zone-Voxel CSR."""

import json
import logging
import time
from pathlib import Path
import numpy as np
import shapely.vectorized
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from ladybug_geometry.geometry3d.pointvector import Point3D
from ladybug_geometry.geometry3d.face import Face3D
from typing import Tuple
from .. import config

logger = logging.getLogger(__name__)
_MAX_CANDIDATES = config.MAX_CANDIDATES
_PROPS_LEN: int = 11
ZONE_VOXEL_SCHEMA_VERSION = 1
ZONE_ROLE_OWNER_SURFACE = 1
ZONE_ROLE_ADJACENT_SURFACE = 2
ZONE_ROLE_APERTURE = 3
ZONE_ROLE_ADJACENT_APERTURE = 4
ZONE_ROLE_OUT_SHADE = 5
ZONE_ROLE_IN_SHADE = 6


def _clean_identifier(value) -> str:
    return str(value).strip() if value is not None else ""


def _surface_zone_links(
    entity: dict, fallback_owner: str, owner_role: int, adjacent_role: int
) -> list[tuple[str, int]]:
    user_data = entity.get("user_data") or {}
    owner = _clean_identifier(user_data.get("owner_zone_id")) or fallback_owner
    adjacent = _clean_identifier(user_data.get("adjacent_zone_id"))
    bc = entity.get("boundary_condition") or {}
    if not adjacent and bc.get("type") == "Surface":
        bc_objects = bc.get("boundary_condition_objects") or []
        if isinstance(bc_objects, list) and len(bc_objects) >= 2:
            adjacent = _clean_identifier(bc_objects[1])
    links: list[tuple[str, int]] = []
    seen: set[str] = set()
    for zone_id, role in ((owner, owner_role), (adjacent, adjacent_role)):
        key = zone_id.upper()
        if zone_id and key not in seen:
            seen.add(key)
            links.append((zone_id, role))
    return links


def _shade_zone_links(shade: dict, fallback_owner: str, role: int) -> list[tuple[str, int]]:
    user_data = shade.get("user_data") or {}
    owner = _clean_identifier(user_data.get("owner_zone_id")) or fallback_owner
    return [(owner, role)] if owner else []


class PhysicalEngine:
    """Resolve HBJSON material layers into equivalent thermal and optical surface properties."""

    def __init__(self, hbjson_data):
        """Initialize and validate the supplied model or physical parameters."""
        props = hbjson_data.get("properties", {})
        energy_props = props.get("energy", {})
        if not energy_props:
            logger.warning("HBJSON contains NO Energy Properties. All physics will use defaults.")
            self.materials = {}
            self.constructions = {}
        else:
            self.materials = {m["identifier"]: m for m in energy_props.get("materials", [])}
            self.constructions = {c["identifier"]: c for c in energy_props.get("constructions", [])}

    def _get_safe_value(self, mat_dict, key, mat_id, default=0.0):
        value = mat_dict.get(key)
        if value is not None:
            return value
        mat_type = mat_dict.get("type", "")
        if key == "conductivity":
            if "Gas" in mat_type or "Gap" in mat_id or "Air" in mat_id:
                logger.info(
                    f"Material '{mat_id}' missing '{key}'. Treated as AIR: {config.AIR_CONDUCTIVITY}"
                )
                return config.AIR_CONDUCTIVITY
            logger.info(
                f"Material '{mat_id}' missing '{key}'. Treated as OPAQUE default: {config.DEFAULT_OPAQUE_CONDUCTIVITY}"
            )
            return config.DEFAULT_OPAQUE_CONDUCTIVITY
        elif key == "density":
            if "Gas" in mat_type or "Gap" in mat_id or "Air" in mat_id:
                logger.info(
                    f"Material '{mat_id}' missing '{key}'. Treated as AIR density: {config.AIR_DENSITY}"
                )
                return config.AIR_DENSITY
            elif "Glazing" in mat_type or "Glass" in mat_id:
                logger.info(
                    f"Material '{mat_id}' missing '{key}'. Treated as GLASS density: {config.GLASS_DENSITY}"
                )
                return config.GLASS_DENSITY
            logger.info(
                f"Material '{mat_id}' missing '{key}'. Treated as OPAQUE density: {config.DEFAULT_OPAQUE_DENSITY}"
            )
            return config.DEFAULT_OPAQUE_DENSITY
        elif key == "specific_heat":
            if "Gas" in mat_type or "Gap" in mat_id or "Air" in mat_id:
                logger.info(
                    f"Material '{mat_id}' missing '{key}'. Treated as AIR specific_heat: {config.AIR_SPECIFIC_HEAT}"
                )
                return config.AIR_SPECIFIC_HEAT
            elif "Glazing" in mat_type or "Glass" in mat_id:
                logger.info(
                    f"Material '{mat_id}' missing '{key}'. Treated as GLASS specific_heat: {config.GLASS_SPECIFIC_HEAT}"
                )
                return config.GLASS_SPECIFIC_HEAT
            logger.info(
                f"Material '{mat_id}' missing '{key}'. Treated as OPAQUE specific_heat: {config.DEFAULT_OPAQUE_SPECIFIC_HEAT}"
            )
            return config.DEFAULT_OPAQUE_SPECIFIC_HEAT
        logger.warning(f"Material '{mat_id}' missing '{key}'. Using absolute fallback: {default}")
        return default

    def get_equivalent_properties(self, construction_id, face_type="Wall", bc_type="Outdoors"):
        """Return thickness (m), conductivity (W/m/K), U-value (W/m2/K), intrinsic volumetric heat capacity (J/m3/K), absorptance, transmittance, and emissivity for the construction."""
        construction = self.constructions.get(construction_id)
        if not construction:
            logger.warning(
                f"Construction ID '{construction_id}' not found in library! Using generic defaults."
            )
            return (
                config.MIN_THICKNESS,
                config.DEFAULT_OPAQUE_CONDUCTIVITY,
                2.0,
                500000.0,
                config.DEFAULT_OPAQUE_SOLAR_ABS,
                config.DEFAULT_OPAQUE_TRANSMITTANCE,
                config.DEFAULT_OPAQUE_EMISSIVITY,
            )
        if construction.get("type") == "ShadeConstruction":
            const_id = construction.get("identifier", "")
            if "InShade" in const_id or const_id.startswith("InShadeConst"):
                thickness = config.INSHADE_THICKNESS
                conductivity = config.INSHADE_CONDUCTIVITY
                density = config.INSHADE_DENSITY
                specific_heat = config.INSHADE_SPECIFIC_HEAT
                cp_rho = density * specific_heat
                refl = construction.get("solar_reflectance", config.INSHADE_SOLAR_REFLECTANCE)
                trans = construction.get("solar_transmittance", config.INSHADE_SOLAR_TRANSMITTANCE)
                solar_abs = round(1.0 - refl - trans, 4)
                emiss = construction.get(
                    "infrared_hemispherical_emissivity", config.INSHADE_EMISSIVITY
                )
                u_value = config.INSHADE_U_VALUE
                return (thickness, conductivity, u_value, cp_rho, solar_abs, trans, emiss)
            else:
                refl = construction.get("solar_reflectance", config.DEFAULT_SHADE_REFLECTANCE)
                solar_abs = round(1.0 - refl, 3)
                return (
                    config.MIN_THICKNESS,
                    config.DEFAULT_OPAQUE_CONDUCTIVITY,
                    5.0,
                    100000.0,
                    solar_abs,
                    config.DEFAULT_OPAQUE_TRANSMITTANCE,
                    config.DEFAULT_SHADE_EMISSIVITY,
                )
        if construction.get("type") == "WindowConstructionShadeAbridged":
            base_id = construction.get("window_construction")
            if isinstance(base_id, dict):
                base_id = base_id.get("identifier")
            base = self.constructions.get(base_id) if base_id else None
            if base:
                construction = base
                logger.debug(
                    f"  [PhysicalEngine] '{construction_id}' → resolved to base '{base_id}'"
                )
            else:
                logger.warning(
                    f"  [PhysicalEngine] '{construction_id}': base construction '{base_id}' not found. Using glazing defaults."
                )
                return (
                    config.MIN_THICKNESS,
                    config.DEFAULT_OPAQUE_CONDUCTIVITY,
                    2.0,
                    500000.0,
                    config.DEFAULT_OPAQUE_SOLAR_ABS,
                    config.DEFAULT_GLAZING_TRANSMITTANCE,
                    config.DEFAULT_GLAZING_EMISSIVITY,
                )
        layers = construction.get("materials", [])
        d_total, r_total, total_heat_capacity = (0, 0, 0)
        solar_abs = config.DEFAULT_OPAQUE_SOLAR_ABS
        transmittance = config.DEFAULT_OPAQUE_TRANSMITTANCE
        emissivity = config.DEFAULT_OPAQUE_EMISSIVITY
        for i, mat_id in enumerate(layers):
            mat = self.materials.get(mat_id)
            if not mat:
                continue
            if mat.get("type") == "EnergyMaterialNoMass":
                d = config.MIN_THICKNESS
                r = mat.get("r_value", 0.1)
                l_val = d / r if r > 0 else 0.5
                rho, cp = (100.0, 800.0)
            else:
                d = self._get_safe_value(mat, "thickness", mat_id, default=config.MIN_THICKNESS)
                l_val = self._get_safe_value(mat, "conductivity", mat_id)
                rho = self._get_safe_value(mat, "density", mat_id)
                cp = self._get_safe_value(mat, "specific_heat", mat_id)
            d_total += d
            if l_val > 0:
                r_total += d / l_val
            total_heat_capacity += rho * cp * d
            if mat.get("type") == "EnergyWindowMaterialGlazing":
                t_pane = mat.get("solar_transmittance", config.DEFAULT_GLAZING_TRANSMITTANCE)
                r_pane = mat.get("solar_reflectance", config.DEFAULT_GLAZING_SOLAR_ABS)
                if transmittance == config.DEFAULT_OPAQUE_TRANSMITTANCE:
                    transmittance = round(t_pane, 4)
                else:
                    transmittance = round(transmittance * t_pane, 4)
                if i == 0:
                    solar_abs = round(max(0.0, 1.0 - t_pane - r_pane), 4)
                    emissivity = mat.get("emissivity", config.DEFAULT_GLAZING_EMISSIVITY)
        if d_total <= 0.001:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if d_total == 0:
            d_total = config.MIN_THICKNESS
            r_total = 0.1
        lambda_eq = d_total / r_total
        cp_rho_eq = total_heat_capacity / d_total
        r_si = config.R_SI.get(face_type, 0.13)
        if bc_type == "Adiabatic":
            logger.info(
                f"Construction '{construction_id}' has Adiabatic BC. lambda_eq and u_value forced to 0. cp_rho_eq={cp_rho_eq:.1f} retained."
            )
            return (d_total, 0.0, 0.0, cp_rho_eq, solar_abs, transmittance, emissivity)
        elif bc_type == "Outdoors":
            r_se = config.R_SE_OUTDOORS
        elif bc_type == "Ground":
            r_se = config.R_SE_GROUND
        elif bc_type == "Surface":
            r_se = r_si
        else:
            r_se = config.R_SE_OUTDOORS
        u_value = 1.0 / (r_si + r_total + r_se)
        return (d_total, lambda_eq, u_value, cp_rho_eq, solar_abs, transmittance, emissivity)


def load_hbjson(file_path, return_metadata: bool = False):
    """Read HBJSON surfaces, physical properties, and coordinate transformation metadata."""
    logger.info(f"Start loading: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    units = data.get("units", "Meters")
    geo_scale = 0.001 if units.lower() == "millimeters" else 1.0
    rooms = data.get("rooms", [])
    zone_ids = []
    for room in rooms:
        user_data = room.get("user_data") or {}
        zone_ids.append(_clean_identifier(user_data.get("zone_id")) or room["identifier"])
    top_shades = []
    seen_top_shades: set[str] = set()
    for key in ("orphaned_shades", "shades"):
        for shade in data.get(key, []):
            shade_id = _clean_identifier(shade.get("identifier"))
            if shade_id not in seen_top_shades:
                seen_top_shades.add(shade_id)
                top_shades.append(shade)
    all_points = []
    for room in rooms:
        for face in room.get("faces", []):
            all_points.extend(face["geometry"]["boundary"])
            for aperture in face.get("apertures", []):
                for shade in aperture.get("indoor_shades", []):
                    all_points.extend(shade["geometry"]["boundary"])
        for shade in room.get("outdoor_shades", []):
            all_points.extend(shade["geometry"]["boundary"])
    for shade in top_shades:
        all_points.extend(shade["geometry"]["boundary"])
    if not all_points:
        logger.error("No geometry points found in HBJSON.")
        empty_metadata = {
            "zone_ids": zone_ids,
            "coordinate_scale": geo_scale,
            "coordinate_offset_m": np.zeros(3, dtype=np.float32),
        }
        return ([], empty_metadata) if return_metadata else []
    min_x = min((pt[0] for pt in all_points))
    min_y = min((pt[1] for pt in all_points))
    min_z = min((pt[2] for pt in all_points))
    logger.info(f"Global Origin Shift applied: [{-min_x}, {-min_y}, {-min_z}]")
    engine = PhysicalEngine(data)
    face_registry = []
    source_metadata = {
        "zone_ids": zone_ids,
        "coordinate_scale": geo_scale,
        "coordinate_offset_m": np.array(
            [-min_x * geo_scale, -min_y * geo_scale, -min_z * geo_scale], dtype=np.float32
        ),
    }

    def transform(pts):
        """Compute transform for the current building or model state. Inputs: pts."""
        return [
            [(p[0] - min_x) * geo_scale, (p[1] - min_y) * geo_scale, (p[2] - min_z) * geo_scale]
            for p in pts
        ]

    for room, room_zone_id in zip(rooms, zone_ids):
        for face in room.get("faces", []):
            bc_dict = face.get("boundary_condition", {})
            bc_type = bc_dict.get("type", "Outdoors")
            bc_code = config.BC_MAP.get(bc_type, 0)
            face_zone_links = _surface_zone_links(
                face, room_zone_id, ZONE_ROLE_OWNER_SURFACE, ZONE_ROLE_ADJACENT_SURFACE
            )
            svf = float(bc_dict.get("view_factor", 1.0)) if bc_type == "Outdoors" else 0.0
            f_type = face.get("face_type")
            if bc_type == "Surface":
                if f_type in ("Floor", "RoofCeiling"):
                    f_type = "InterFloorSlab"
                else:
                    f_type = "InteriorWall"
            constr_id = face["properties"]["energy"]["construction"]
            phys_props = engine.get_equivalent_properties(
                constr_id, face_type=f_type, bc_type=bc_type
            )
            face_apertures = []
            for ap in face.get("apertures", []):
                ap_bc_dict = ap.get("boundary_condition", {})
                ap_bc_type = ap_bc_dict.get("type", bc_type)
                aperture_zone_links = _surface_zone_links(
                    ap, room_zone_id, ZONE_ROLE_APERTURE, ZONE_ROLE_ADJACENT_APERTURE
                )
                ap_bc_code = config.BC_MAP.get(ap_bc_type, 0)
                ap_svf = float(ap_bc_dict.get("view_factor", svf))
                ap_constr = ap["properties"]["energy"]["construction"]
                ap_phys = engine.get_equivalent_properties(
                    ap_constr, face_type="Aperture", bc_type=ap_bc_type
                )
                constr_id = (
                    ap_constr if isinstance(ap_constr, str) else ap_constr.get("identifier", "")
                )
                is_shaded = constr_id.endswith("_shaded")
                face_apertures.append(
                    {
                        "id": ap["identifier"],
                        "boundary": transform(ap["geometry"]["boundary"]),
                        "props": (*ap_phys, ap_bc_code, ap_svf),
                        "is_shaded": (
                            ap_constr.endswith("_shaded") if isinstance(ap_constr, str) else False
                        ),
                        "zone_links": aperture_zone_links,
                    }
                )
            face_registry.append(
                {
                    "id": face["identifier"],
                    "type": f_type,
                    "boundary": transform(face["geometry"]["boundary"]),
                    "props": (*phys_props, bc_code, svf),
                    "apertures": face_apertures,
                    "zone_links": face_zone_links,
                }
            )
        for face in room.get("faces", []):
            for ap in face.get("apertures", []):
                for in_shade in ap.get("indoor_shades", []):
                    shade_constr = (
                        in_shade.get("properties", {}).get("energy", {}).get("construction")
                    )
                    s_phys = engine.get_equivalent_properties(shade_constr)
                    s_scaled = transform(in_shade["geometry"]["boundary"])
                    face_registry.append(
                        {
                            "id": in_shade["identifier"],
                            "type": "InShade",
                            "boundary": s_scaled,
                            "props": (*s_phys, 0, 0.0),
                            "apertures": [],
                            "zone_links": _shade_zone_links(
                                in_shade, room_zone_id, ZONE_ROLE_IN_SHADE
                            ),
                        }
                    )
        for shade in room.get("outdoor_shades", []):
            shade_constr = shade.get("properties", {}).get("energy", {}).get("construction")
            s_phys = engine.get_equivalent_properties(shade_constr)
            s_scaled = transform(shade["geometry"]["boundary"])
            face_registry.append(
                {
                    "id": shade["identifier"],
                    "type": "OutShade",
                    "boundary": s_scaled,
                    "props": (*s_phys, 0, 0.0),
                    "apertures": [],
                    "zone_links": _shade_zone_links(shade, room_zone_id, ZONE_ROLE_OUT_SHADE),
                }
            )
    for shade in top_shades:
        shade_constr = shade.get("properties", {}).get("energy", {}).get("construction")
        s_phys = engine.get_equivalent_properties(shade_constr)
        s_scaled = transform(shade["geometry"]["boundary"])
        face_registry.append(
            {
                "id": shade["identifier"],
                "type": "OutShade",
                "boundary": s_scaled,
                "props": (*s_phys, 0, 0.0),
                "apertures": [],
                "zone_links": _shade_zone_links(shade, "", ZONE_ROLE_OUT_SHADE),
            }
        )
    logger.info(f"Successfully processed {len(face_registry)} main faces/shades.")
    return (face_registry, source_metadata) if return_metadata else face_registry


def _build_zone_voxel_map(
    zone_ids: "list[str] | None",
    winner_fidx: np.ndarray,
    record_zone_links: list[list[tuple[str, int]]],
    voxel_size_m: float,
    source_metadata: "dict | None" = None,
) -> dict[str, np.ndarray]:
    """Build a CSR incidence table from zone IDs to global voxel rows, retaining surface roles and the source coordinate transform."""
    if zone_ids is None:
        resolved_zone_ids: list[str] = []
        seen: set[str] = set()
        for links in record_zone_links:
            for zone_id, _ in links:
                key = zone_id.upper()
                if zone_id and key not in seen:
                    seen.add(key)
                    resolved_zone_ids.append(zone_id)
    else:
        resolved_zone_ids = [_clean_identifier(z) for z in zone_ids]
    zone_lookup = {zone_id.upper(): i for i, zone_id in enumerate(resolved_zone_ids)}
    per_zone_voxels: list[list[int]] = [[] for _ in resolved_zone_ids]
    per_zone_roles: list[list[int]] = [[] for _ in resolved_zone_ids]
    voxel_zone_counts = np.zeros(len(winner_fidx), dtype=np.uint16)
    unknown_zone_ids: set[str] = set()
    for voxel_index, record_index in enumerate(winner_fidx):
        for zone_id, role in record_zone_links[int(record_index)]:
            zone_index = zone_lookup.get(zone_id.upper())
            if zone_index is None:
                unknown_zone_ids.add(zone_id)
                continue
            per_zone_voxels[zone_index].append(voxel_index)
            per_zone_roles[zone_index].append(role)
            voxel_zone_counts[voxel_index] += 1
    if unknown_zone_ids:
        logger.warning(
            "[ZoneVoxel] Ignored links to unknown Zone IDs: %s", sorted(unknown_zone_ids)
        )
    zone_ptr = np.zeros(len(resolved_zone_ids) + 1, dtype=np.int64)
    for zone_index, voxel_indices in enumerate(per_zone_voxels):
        zone_ptr[zone_index + 1] = zone_ptr[zone_index] + len(voxel_indices)
    if zone_ptr[-1] > 0:
        zone_voxel_indices = np.concatenate(
            [np.asarray(v, dtype=np.int64) for v in per_zone_voxels]
        )
        zone_voxel_roles = np.concatenate([np.asarray(v, dtype=np.uint8) for v in per_zone_roles])
    else:
        zone_voxel_indices = np.empty(0, dtype=np.int64)
        zone_voxel_roles = np.empty(0, dtype=np.uint8)
    source_metadata = source_metadata or {}
    return {
        "zone_voxel_schema_version": np.asarray(ZONE_VOXEL_SCHEMA_VERSION, dtype=np.int16),
        "zone_ids": np.asarray(resolved_zone_ids, dtype=str),
        "zone_ptr": zone_ptr,
        "zone_voxel_indices": zone_voxel_indices,
        "zone_voxel_roles": zone_voxel_roles,
        "voxel_zone_counts": voxel_zone_counts,
        "voxel_size_m": np.asarray(voxel_size_m, dtype=np.float32),
        "coordinate_scale": np.asarray(
            source_metadata.get("coordinate_scale", 1.0), dtype=np.float32
        ),
        "coordinate_offset_m": np.asarray(
            source_metadata.get("coordinate_offset_m", np.zeros(3)), dtype=np.float32
        ),
    }


def voxelize_hbjson(
    hbjson_path: "str | Path",
    parent_size: "float | None" = None,
    min_fill: "float | None" = None,
    return_zone_map: bool = False,
):
    """Read HBJSON and return float32 physical voxel points (V,15), optionally with Zone-Voxel CSR metadata. Millimeter coordinates are converted to meters and shifted using the stored origin offset."""
    face_registry, source_metadata = load_hbjson(hbjson_path, return_metadata=True)
    return voxelize_from_faces(
        face_registry,
        parent_size,
        min_fill,
        zone_ids=source_metadata["zone_ids"],
        source_metadata=source_metadata,
        return_zone_map=return_zone_map,
    )


def voxelize_from_faces(
    face_registry: list,
    parent_size: "float | None" = None,
    min_fill: "float | None" = None,
    *,
    zone_ids: "list[str] | None" = None,
    source_metadata: "dict | None" = None,
    return_zone_map: bool = False,
):
    """Intersect material slabs with the parent grid, activate them by thickness-normalized support, and resolve overlapping surfaces. Heat capacity remains intrinsic; geometric fill is quality metadata, not a model channel."""
    L = float(parent_size if parent_size is not None else config.VOXEL_PARENT_SIZE)
    support_override = float(min_fill) if min_fill is not None else None
    t_total = time.time()
    logger.info(
        f"[HbjsonToVoxel] Start face extrution  parent_size={L} m  support_override={support_override}  faces={len(face_registry)}"
    )
    face_records: list[np.ndarray] = []
    face_weights: list[float] = []
    ijk_chunks: list[np.ndarray] = []
    record_zone_links: list[list[tuple[str, int]]] = []
    fill_chunks: list[np.ndarray] = []
    fidx_chunks: list[np.ndarray] = []
    face_id_counter = 1000
    total_raw_hits = 0
    total_discarded_by_fill = 0
    t1 = time.time()
    for entry in face_registry:
        n_ap = len(entry.get("apertures", []))
        hits, disc = _process_face_entry(
            entry,
            face_id_counter,
            L,
            support_override,
            face_records,
            face_weights,
            record_zone_links,
            ijk_chunks,
            fill_chunks,
            fidx_chunks,
        )
        face_id_counter += 1 + n_ap
        total_raw_hits += hits
        total_discarded_by_fill += disc
    logger.info(
        f"  Pass 1 finish: {total_raw_hits:,} raw hits  (discarded by support threshold: {total_discarded_by_fill:,})  {len(face_records)} valid faces  ({time.time() - t1:.2f} s)"
    )
    if not ijk_chunks:
        logger.warning("[HbjsonToVoxel] Fail in voxelization, please check face_registry。")
        empty_voxels = np.empty((0, 15), dtype=np.float32)
        if return_zone_map:
            return (
                empty_voxels,
                _build_zone_voxel_map(
                    zone_ids, np.empty(0, dtype=np.int32), record_zone_links, L, source_metadata
                ),
            )
        return empty_voxels
    t2 = time.time()
    all_ijk = np.vstack(ijk_chunks).astype(np.int32)
    all_fills = np.concatenate(fill_chunks).astype(np.float32)
    all_fidx = np.concatenate(fidx_chunks).astype(np.int32)
    N = len(all_ijk)
    wt_arr = np.array(face_weights, dtype=np.float32)
    face_thickness = np.array(
        [max(float(props[0]), np.finfo(np.float32).eps) for props in face_records], dtype=np.float32
    )
    thickness_scale = L / np.minimum(face_thickness, L)
    score_weights = wt_arr * thickness_scale
    all_scores = all_fills * score_weights[all_fidx]
    g_min = all_ijk.min(axis=0).astype(np.int64)
    g_dims = all_ijk.max(axis=0).astype(np.int64) - g_min + 1
    ijk64 = all_ijk.astype(np.int64) - g_min
    flat = ijk64[:, 0] * g_dims[1] * g_dims[2] + ijk64[:, 1] * g_dims[2] + ijk64[:, 2]
    order = np.lexsort((all_scores, flat))
    flat_s = flat[order]
    last = np.empty(N, dtype=bool)
    last[-1] = True
    last[:-1] = flat_s[:-1] != flat_s[1:]
    win_pos = order[last]
    winner_ijk = all_ijk[win_pos]
    winner_fills = all_fills[win_pos]
    winner_fidx = all_fidx[win_pos]
    V = len(winner_ijk)
    logger.info(
        f"  Pass 2 finish: {N:,} total → {V:,} single parent voxel  （Deduplication rate: {100 * (1 - V / max(N, 1)):.1f}%）  ({time.time() - t2:.2f} s)"
    )
    t3 = time.time()
    face_props_mat = np.vstack(face_records).astype(np.float32)
    props_mat = face_props_mat[winner_fidx]
    centers = (winner_ijk.astype(np.float32) + 0.5) * L
    final_vox = np.column_stack([centers, props_mat, winner_fills])
    logger.info(f"  Pass 3 finish: totally {V:,} voxels,  ({time.time() - t3:.2f} s)")
    logger.info(
        f"[HbjsonToVoxel] Finish  Total use: {time.time() - t_total:.2f} s  Output shape: {final_vox.shape}"
    )
    final_vox = final_vox.astype(np.float32)
    if return_zone_map:
        zone_voxel_map = _build_zone_voxel_map(
            zone_ids, winner_fidx, record_zone_links, L, source_metadata
        )
        shared_voxels = int(np.count_nonzero(zone_voxel_map["voxel_zone_counts"] > 1))
        logger.info(
            f"  Zone–Voxel CSR: zones={len(zone_voxel_map['zone_ids'])}  edges={len(zone_voxel_map['zone_voxel_indices']):,}  shared_voxels={shared_voxels:,}"
        )
        return (final_vox, zone_voxel_map)
    return final_vox


def _process_face_entry(
    entry: dict,
    face_id_base: int,
    L: float,
    support_override: "float | None",
    face_records: list,
    face_weights: list,
    record_zone_links: list,
    ijk_chunks: list,
    fill_chunks: list,
    fidx_chunks: list,
) -> Tuple[int, int]:
    total_hits = 0
    total_discarded = 0
    try:
        host_pts = [Point3D(*p) for p in entry["boundary"]]
        face3d = Face3D(host_pts)
    except Exception as exc:
        logger.warning(f"[HbjsonToVoxel] Skip face {entry['id']}: fail construction ({exc})")
        return (0, 0)
    plane = face3d.plane
    origin = np.array([plane.o.x, plane.o.y, plane.o.z], dtype=np.float64)
    vx = np.array([plane.x.x, plane.x.y, plane.x.z], dtype=np.float64)
    vy = np.array([plane.y.x, plane.y.y, plane.y.z], dtype=np.float64)
    normal = np.array([face3d.normal.x, face3d.normal.y, face3d.normal.z], dtype=np.float64)

    def _to_2d(boundary_3d: list) -> list:
        pts = np.array(boundary_3d, dtype=np.float64) - origin
        return list(zip(pts @ vx, pts @ vy))

    exterior_2d = _to_2d(entry["boundary"])
    aperture_entries = []
    aperture_keepouts = []
    for ap in entry.get("apertures", []):
        ap_2d = _to_2d(ap["boundary"])
        try:
            ap_keepout = ShapelyPolygon(ap_2d)
            if not ap_keepout.is_valid:
                ap_keepout = ap_keepout.buffer(0)
            ap_d = float(ap["props"][0])
            if ap_d > 0.001 and (not ap_keepout.is_empty):
                ap_keepout = ap_keepout.buffer(ap_d / 2.0, join_style=2)
            if not ap_keepout.is_empty:
                aperture_keepouts.append(ap_keepout)
        except Exception as exc:
            logger.warning(
                f"[HbjsonToVoxel] Fail to build aperture keepout on {entry['id']}: {exc}"
            )
        aperture_entries.append(
            {
                "boundary_2d": ap_2d,
                "props": ap["props"],
                "is_shaded": ap.get("is_shaded", False),
                "zone_links": ap.get("zone_links", []),
            }
        )
    wall_d = float(entry["props"][0])
    host_tag = config.SEMANTIC_TAGS.get(entry["type"], 1)
    host_wt = config.VOXEL_TAG_WEIGHTS.get(host_tag, 1)
    host_min_support = (
        support_override
        if support_override is not None
        else config.TAG_MIN_SUPPORT_RATE.get(host_tag, 0.15)
    )
    if wall_d > 0.001:
        try:
            wall_poly = ShapelyPolygon(exterior_2d)
            if not wall_poly.is_valid:
                wall_poly = wall_poly.buffer(0)
            if not wall_poly.is_empty:
                wall_poly_buf = wall_poly.buffer(wall_d / 2.0, join_style=2)
                if aperture_keepouts:
                    wall_poly_buf = wall_poly_buf.difference(unary_union(aperture_keepouts))
                props_arr = np.array(
                    [*entry["props"], float(host_tag), float(face_id_base)], dtype=np.float32
                )
                rec_idx = len(face_records)
                face_records.append(props_arr)
                face_weights.append(float(host_wt))
                record_zone_links.append(entry.get("zone_links", []))
                hits, disc = _extrude_slab(
                    wall_poly_buf,
                    origin,
                    normal,
                    vx,
                    vy,
                    wall_d,
                    rec_idx,
                    L,
                    host_min_support,
                    ijk_chunks,
                    fill_chunks,
                    fidx_chunks,
                )
                total_hits += hits
                total_discarded += disc
        except Exception as exc:
            logger.warning(f"[HbjsonToVoxel] Fail in main face {entry['id']}:{exc}")
    for i, ap_entry in enumerate(aperture_entries):
        ap_d = float(ap_entry["props"][0])
        if ap_d <= 0.001:
            continue
        try:
            ap_poly = ShapelyPolygon(ap_entry["boundary_2d"])
            if ap_poly.is_empty:
                continue
            ap_poly_buf = ap_poly.buffer(ap_d / 2.0, join_style=2)
            if ap_entry.get("is_shaded", False):
                ap_tag = config.SEMANTIC_TAGS.get("InShade", 8)
                ap_wt = config.VOXEL_TAG_WEIGHTS.get(ap_tag, 4)
            else:
                ap_tag = config.SEMANTIC_TAGS.get("Aperture", 2)
                ap_wt = config.VOXEL_TAG_WEIGHTS.get(ap_tag, 4)
            ap_props = np.array(
                [*ap_entry["props"], float(ap_tag), float(face_id_base + i + 1)], dtype=np.float32
            )
            rec_idx = len(face_records)
            face_records.append(ap_props)
            face_weights.append(float(ap_wt))
            record_zone_links.append(ap_entry.get("zone_links", []))
            aperture_min_support = (
                support_override
                if support_override is not None
                else config.TAG_MIN_SUPPORT_RATE.get(ap_tag, 0.05)
            )
            hits, disc = _extrude_slab(
                ap_poly_buf,
                origin,
                normal,
                vx,
                vy,
                ap_d,
                rec_idx,
                L,
                aperture_min_support,
                ijk_chunks,
                fill_chunks,
                fidx_chunks,
            )
            total_hits += hits
            total_discarded += disc
        except Exception as exc:
            logger.warning(f"[HbjsonToVoxel] Fail in Aperture {entry['id']} the {i}th:{exc}")
    return (total_hits, total_discarded)


def _extrude_slab(
    poly_2d: ShapelyPolygon,
    origin: np.ndarray,
    normal: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    thickness: float,
    rec_idx: int,
    L: float,
    min_support: float,
    ijk_chunks: list,
    fill_chunks: list,
    fidx_chunks: list,
) -> Tuple[int, int]:
    """Intersect a face slab with candidate parent voxels. Activation compares normal overlap / min(material thickness, voxel size) with the requested support threshold."""
    half_d = thickness / 2.0
    minx, miny, maxx, maxy = poly_2d.bounds
    c2d = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]], dtype=np.float64)
    c3d = origin + c2d[:, 0:1] * vx + c2d[:, 1:2] * vy
    all_corners = np.vstack([c3d + half_d * normal, c3d - half_d * normal])
    w_min = all_corners.min(axis=0) - L
    w_max = all_corners.max(axis=0) + L
    i_rng = np.arange(int(np.floor(w_min[0] / L)), int(np.ceil(w_max[0] / L)) + 1, dtype=np.int32)
    j_rng = np.arange(int(np.floor(w_min[1] / L)), int(np.ceil(w_max[1] / L)) + 1, dtype=np.int32)
    k_rng = np.arange(int(np.floor(w_min[2] / L)), int(np.ceil(w_max[2] / L)) + 1, dtype=np.int32)
    n_cand = len(i_rng) * len(j_rng) * len(k_rng)
    if n_cand > _MAX_CANDIDATES:
        logger.warning(
            f"[HbjsonToVoxel] Panel AABB too large: ({n_cand:,} candidate voxels)，Skip. Check the geometry."
        )
        return (0, 0)
    n_cand = len(i_rng) * len(j_rng) * len(k_rng)
    II, JJ, KK = np.meshgrid(i_rng, j_rng, k_rng, indexing="ij")
    shape_flat = II.size
    centers = np.empty((shape_flat, 3), dtype=np.float64)
    centers[:, 0] = (II.ravel() + 0.5) * L
    centers[:, 1] = (JJ.ravel() + 0.5) * L
    centers[:, 2] = (KK.ravel() + 0.5) * L
    ijk_raw = np.stack([II.ravel(), JJ.ravel(), KK.ravel()], axis=1).astype(np.int32)
    delta = centers - origin
    dist_n = delta @ normal
    half_L = L / 2.0
    overlap = np.clip(
        np.minimum(half_d, dist_n + half_L) - np.maximum(-half_d, dist_n - half_L), 0.0, None
    )
    physical_fill = overlap / L
    support_denominator = max(min(thickness, L), np.finfo(np.float64).eps)
    support_ratio = overlap / support_denominator
    mask_n = support_ratio >= min_support
    kept_by_fill = np.count_nonzero(mask_n)
    discarded_by_fill = n_cand - kept_by_fill
    if not np.any(mask_n):
        return (0, discarded_by_fill)
    delta_f = delta[mask_n]
    fill_f = physical_fill[mask_n].astype(np.float32)
    ijk_f = ijk_raw[mask_n]
    proj_u = (delta_f @ vx).astype(np.float64)
    proj_v = (delta_f @ vy).astype(np.float64)
    inside = shapely.vectorized.contains(poly_2d, proj_u, proj_v)
    if not np.any(inside):
        return (0, discarded_by_fill)
    final_ijk = ijk_f[inside]
    final_fills = fill_f[inside]
    H = len(final_ijk)
    ijk_chunks.append(final_ijk)
    fill_chunks.append(final_fills)
    fidx_chunks.append(np.full(H, rec_idx, dtype=np.int32))
    return (H, discarded_by_fill)
