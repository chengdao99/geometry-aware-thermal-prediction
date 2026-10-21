"""Parameterized building add windows.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)
from . import geometric_config as cfg
from .divide_rooms import RoomDivisionResult
from .extrude_rooms import ExtrusionResult


@dataclass
class ApertureGeom:
    """Represent aperture geom and its associated geometry or physical configuration."""

    identifier: str
    vertices: List[Tuple[float, float, float]]
    orientation: str
    area: float
    wwr: float
    coverage: Optional[float] = None
    owner_zone_id: str = ""
    host_face_id: str = ""
    aperture_group_id: str = ""


@dataclass
class OutShadingGeom:
    """Represent out shading geom and its associated geometry or physical configuration."""

    identifier: str
    vertices: List[Tuple[float, float, float]]
    depth: float
    overhang: float
    gap: float
    owner_zone_id: str = ""
    host_face_id: str = ""
    host_aperture_group_id: str = ""


@dataclass
class FaceModel:
    """Represent face model and its associated geometry or physical configuration."""

    identifier: str
    owner_zone_id: str
    vertices: List[Tuple[float, float, float]]
    face_type: str
    bc_type: str
    is_exterior_wall: bool
    orientation: Optional[str]
    apertures: List[ApertureGeom] = field(default_factory=list)
    out_shading: List[OutShadingGeom] = field(default_factory=list)
    adjacent_to: Optional[str] = None

    @property
    def net_wall_area(self) -> float:
        """Compute net wall area for the current building or model state."""
        verts = self.vertices
        n = len(verts)
        cross = [0.0, 0.0, 0.0]
        for i in range(n):
            v0 = verts[i]
            v1 = verts[(i + 1) % n]
            cross[0] += v0[1] * v1[2] - v0[2] * v1[1]
            cross[1] += v0[2] * v1[0] - v0[0] * v1[2]
            cross[2] += v0[0] * v1[1] - v0[1] * v1[0]
        gross = 0.5 * math.sqrt(sum((c * c for c in cross)))
        return round(gross - sum((a.area for a in self.apertures)), 4)


@dataclass
class RoomModel:
    """Represent room model and its associated geometry or physical configuration."""

    identifier: str
    centroid_2d: Tuple[float, float]
    zone_type: str
    floor_area: float
    floor_index: int = 0
    faces: List[FaceModel] = field(default_factory=list)

    @property
    def exterior_faces(self) -> List[FaceModel]:
        """Compute exterior faces for the current building or model state."""
        return [f for f in self.faces if f.is_exterior_wall]

    @property
    def n_windows(self) -> int:
        """Compute n windows for the current building or model state."""
        return sum((len(f.apertures) for f in self.faces))

    @property
    def total_window_area(self) -> float:
        """Compute total window area for the current building or model state."""
        return round(sum((a.area for f in self.faces for a in f.apertures)), 3)


@dataclass
class BuildingModel:
    """Represent building model and its associated geometry or physical configuration."""

    shape_type: str
    n_floors: int
    building_type: str
    floor_height: float
    grid_spacing: float
    floor_bc: str
    ceiling_bc: str
    total_floor_area: float
    sill_height: float
    head_height: float
    wwr_by_orientation: dict
    out_shade_depth_by_orientation: dict = field(default_factory=dict)
    out_shade_gap: float = 0.1
    rotation_angle_deg: float = 0.0
    rooms: List[RoomModel] = field(default_factory=list)

    @property
    def n_rooms(self) -> int:
        """Compute n rooms for the current building or model state."""
        return len(self.rooms)

    @property
    def total_windows(self) -> int:
        """Compute total windows for the current building or model state."""
        return sum((r.n_windows for r in self.rooms))

    @property
    def total_window_area(self) -> float:
        """Compute total window area for the current building or model state."""
        return round(sum((r.total_window_area for r in self.rooms)), 3)

    def summary(self) -> str:
        """Compute summary for the current building or model state."""
        lines = [
            f"BuildingModel Summary",
            f"  Shape:          {self.shape_type}",
            f"  Building type:  {self.building_type} ({self.n_floors} floors)",
            f"  Floor height:   {self.floor_height:.2f} m",
            f"  Room Grid:      {self.grid_spacing:.2f} m",
            f"  Floor BC:       {self.floor_bc}",
            f"  Ceiling BC:     {self.ceiling_bc}",
            f"  Total area:     {self.total_floor_area:.1f} m²",
            f"  Rooms:          {self.n_rooms}",
            f"  Sill / Head:    {self.sill_height:.2f} m / {self.head_height:.2f} m",
            f"  WWR by orient:  {self.wwr_by_orientation}",
            f"  OutShade depth: {self.out_shade_depth_by_orientation}  gap={self.out_shade_gap:.2f} m",
            f"  Rotation:       {self.rotation_angle_deg:.2f}°",
            f"  Total windows:  {self.total_windows}  (area = {self.total_window_area:.1f} m²)",
        ]
        for r in self.rooms:
            lines.append(
                f"    [{r.identifier}] zone={r.zone_type:10s}  area={r.floor_area:6.1f} m²  ext_walls={len(r.exterior_faces)}  windows={r.n_windows}  win_area={r.total_window_area:.1f} m²"
            )
        return "\n".join(lines)


def _wall_outward_normal_2d(verts: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    x0, y0, _ = verts[0]
    x1, y1, _ = verts[1]
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-09:
        return (0.0, 1.0)
    nx = -dy / length
    ny = dx / length
    return (nx, ny)


def _normal_to_orientation(nx: float, ny: float) -> str:
    angle_deg = math.degrees(math.atan2(ny, nx))
    if -45.0 <= angle_deg < 45.0:
        return "E"
    elif 45.0 <= angle_deg < 135.0:
        return "N"
    elif angle_deg >= 135.0 or angle_deg < -135.0:
        return "W"
    else:
        return "S"


def _compute_aperture_vertices(
    wall_verts: List[Tuple[float, float, float]], sill_height: float, head_height: float, wwr: float
) -> Optional[Tuple[List[Tuple[float, float, float]], float]]:
    BL = wall_verts[0]
    BR = wall_verts[1]
    wall_width = math.hypot(BR[0] - BL[0], BR[1] - BL[1])
    wall_height = wall_verts[2][2] - wall_verts[0][2]
    actual_head = min(head_height, wall_height - 0.05)
    if sill_height >= actual_head:
        return None
    win_height = actual_head - sill_height
    if win_height < 0.2:
        return None
    if wall_width < cfg.MIN_WALL_WIDTH_FOR_WINDOW:
        return None
    wall_area = wall_width * wall_height
    target_area = wwr * wall_area
    target_width = target_area / win_height
    max_width = wall_width - 2.0 * cfg.WINDOW_WALL_MARGIN
    if max_width < 0.3:
        return None
    actual_width = min(target_width, max_width)
    actual_area = actual_width * win_height
    actual_wwr = actual_area / wall_area
    u_center = wall_width / 2.0
    u_start = u_center - actual_width / 2.0
    u_end = u_center + actual_width / 2.0
    z_base = wall_verts[0][2]
    ux = (BR[0] - BL[0]) / wall_width
    uy = (BR[1] - BL[1]) / wall_width

    def wall_pt(u: float, v: float) -> Tuple[float, float, float]:
        """Compute wall pt for the current building or model state. Inputs: u, v."""
        return (BL[0] + u * ux, BL[1] + u * uy, z_base + v)

    aperture_verts = [
        wall_pt(u_start, sill_height),
        wall_pt(u_end, sill_height),
        wall_pt(u_end, actual_head),
        wall_pt(u_start, actual_head),
    ]
    return (aperture_verts, round(actual_wwr, 4))


def _compute_exterior_shade_vertices(
    aperture_verts: List[Tuple[float, float, float]],
    wall_verts: List[Tuple[float, float, float]],
    depth: float,
    overhang: float,
    gap: float,
) -> List[Tuple[float, float, float]]:
    TL = aperture_verts[3]
    TR = aperture_verts[2]
    nx, ny = _wall_outward_normal_2d(wall_verts)
    dx = TR[0] - TL[0]
    dy = TR[1] - TL[1]
    width = math.hypot(dx, dy)
    if width < 1e-06:
        width = 1.0
    ux = dx / width
    uy = dy / width
    left_ext = (-overhang * ux, -overhang * uy)
    right_ext = (overhang * ux, overhang * uy)
    z_base = TL[2] + gap
    pt_bl = (TL[0] + left_ext[0] + nx * depth, TL[1] + left_ext[1] + ny * depth, z_base)
    pt_br = (TR[0] + right_ext[0] + nx * depth, TR[1] + right_ext[1] + ny * depth, z_base)
    pt_tr = (TR[0] + right_ext[0], TR[1] + right_ext[1], z_base)
    pt_tl = (TL[0] + left_ext[0], TL[1] + left_ext[1], z_base)
    return [pt_bl, pt_br, pt_tr, pt_tl]


def _split_aperture_vertices(
    aperture_verts: List[Tuple[float, float, float]], coverage: float
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    BL, BR, TR, TL = aperture_verts
    z_bottom = BL[2]
    z_top = TR[2]
    total_h = z_top - z_bottom
    split_z = z_top - total_h * coverage
    lower_BL = (BL[0], BL[1], z_bottom)
    lower_BR = (BR[0], BR[1], z_bottom)
    lower_TR = (BR[0], BR[1], split_z)
    lower_TL = (BL[0], BL[1], split_z)
    upper_BL = (BL[0], BL[1], split_z)
    upper_BR = (BR[0], BR[1], split_z)
    upper_TR = (BR[0], BR[1], z_top)
    upper_TL = (BL[0], BL[1], z_top)
    return ([lower_BL, lower_BR, lower_TR, lower_TL], [upper_BL, upper_BR, upper_TR, upper_TL])


def add_windows(
    division_result: RoomDivisionResult,
    extrusion_result: ExtrusionResult,
    seed: int | None = None,
    overrides=None,
) -> BuildingModel:
    """Attach orientation-dependent windows and shading to building surfaces."""
    if division_result is None or extrusion_result is None:
        missing = []
        if division_result is None:
            missing.append("division_result")
        if extrusion_result is None:
            missing.append("extrusion_result")
        logger.error(
            f"[add_windows] Received None for {', '.join(missing)} from upstream — skipping window generation."
        )
        return None
    rng = random.Random(seed)
    rng_style = random.Random(int(seed * 1.2))
    max_allowable_head = round(extrusion_result.floor_height - 0.15, 3)
    overridden_sill = None
    overridden_head = None
    overridden_wwr = None
    if overrides:
        if overrides.sill_height is not None:
            overridden_sill = round(max(0.15, overrides.sill_height), 3)
        if overrides.head_height is not None:
            overridden_head = round(min(overrides.head_height, max_allowable_head), 3)
        if overrides.wwr_by_orientation is not None:
            overridden_wwr = overrides.wwr_by_orientation
    style_choices = list(cfg.WINDOW_STYLE_PROBS.keys())
    style_weights = list(cfg.WINDOW_STYLE_PROBS.values())
    window_style = rng_style.choices(style_choices, weights=style_weights, k=1)[0]
    logger.info(f"[add_windows] Selected building window style: {window_style}")
    if window_style == "extreme_large":
        sill_height = (
            overridden_sill
            if overridden_sill is not None
            else round(max(0.15, rng.uniform(*cfg.SILL_EXTREME_LARGE_RANGE)), 3)
        )
        if overridden_head is not None:
            head_height = overridden_head
        else:
            head_raw = rng.uniform(*cfg.HEAD_EXTREME_LARGE_RANGE)
            head_height = round(min(head_raw, max_allowable_head), 3)
        if head_height <= sill_height + 0.5:
            head_height = max_allowable_head
            sill_height = round(max(0.15, head_height - 1.0), 3)
        if overridden_wwr is not None:
            wwr_map = overridden_wwr
        else:
            large_wwr = round(rng.uniform(*cfg.WWR_EXTREME_LARGE_RANGE), 3)
            wwr_map = {"N": large_wwr, "S": large_wwr, "E": large_wwr, "W": large_wwr}
    elif window_style == "extreme_small":
        sill_height = (
            overridden_sill
            if overridden_sill is not None
            else round(max(0.15, rng.uniform(*cfg.SILL_EXTREME_SMALL_RANGE)), 3)
        )
        if overridden_head is not None:
            head_height = overridden_head
        else:
            head_raw = rng.uniform(*cfg.HEAD_EXTREME_SMALL_RANGE)
            head_height = round(max(sill_height + 0.3, min(head_raw, max_allowable_head)), 3)
        if sill_height >= head_height - 0.3:
            head_height = max_allowable_head
            sill_height = round(max(0.15, head_height - 0.4), 3)
        if overridden_wwr is not None:
            wwr_map = overridden_wwr
        else:
            small_wwr = round(rng.uniform(*cfg.WWR_EXTREME_SMALL_RANGE), 3)
            wwr_map = {"N": small_wwr, "S": small_wwr, "E": small_wwr, "W": small_wwr}
    else:
        if overridden_sill is not None:
            sill_height = overridden_sill
        else:
            sill_height = round(max(0.15, rng.uniform(*cfg.SILL_HEIGHT_RANGE)), 3)
        if overridden_head is not None:
            head_height = overridden_head
        else:
            head_raw = rng.uniform(*cfg.HEAD_HEIGHT_RANGE)
            head_height = round(max(sill_height + 0.5, min(head_raw, max_allowable_head)), 3)
        if overridden_wwr is not None:
            wwr_map = overridden_wwr
        else:
            wwr_map = {
                "N": round(rng.uniform(*cfg.WWR_NORTH_RANGE), 3),
                "S": round(rng.uniform(*cfg.WWR_SOUTH_RANGE), 3),
                "E": round(rng.uniform(*cfg.WWR_EAST_RANGE), 3),
                "W": round(rng.uniform(*cfg.WWR_WEST_RANGE), 3),
            }
    if sill_height >= head_height - 0.1:
        logger.warning(
            "[add_windows] Overrides or generated bounds caused an overlap. Enforcing safe emergency bounds."
        )
        head_height = max_allowable_head
        sill_height = round(max(0.15, head_height - 0.5), 3)
    if overrides and overrides.out_shade_fixed is not None:
        exterior_depth_map = {
            orient: max(0.0, float(overrides.out_shade_fixed.get(orient, 0.0)))
            for orient in ("N", "S", "E", "W")
        }
        exterior_has_map = {orient: depth > 1e-09 for orient, depth in exterior_depth_map.items()}
    else:
        exterior_depth_map = {
            "N": round(rng.uniform(*cfg.EXTERIOR_SHADE_DEPTH_NORTH_RANGE), 3),
            "S": round(rng.uniform(*cfg.EXTERIOR_SHADE_DEPTH_SOUTH_RANGE), 3),
            "E": round(rng.uniform(*cfg.EXTERIOR_SHADE_DEPTH_EAST_RANGE), 3),
            "W": round(rng.uniform(*cfg.EXTERIOR_SHADE_DEPTH_WEST_RANGE), 3),
        }
        exterior_has_map = {
            "N": rng.random() < cfg.EXTERIOR_SHADE_PROB_BY_ORIENT["N"],
            "S": rng.random() < cfg.EXTERIOR_SHADE_PROB_BY_ORIENT["S"],
            "E": rng.random() < cfg.EXTERIOR_SHADE_PROB_BY_ORIENT["E"],
            "W": rng.random() < cfg.EXTERIOR_SHADE_PROB_BY_ORIENT["W"],
        }
    in_shade_fixed = None
    if overrides and overrides.in_shade_fixed is not None:
        in_shade_fixed = overrides.in_shade_fixed
    use_fixed_shade = in_shade_fixed is not None
    if use_fixed_shade:
        fixed_coverage = max(0.0, min(1.0, in_shade_fixed))
        generate_shade = fixed_coverage > 0.0
    else:
        generate_shade = True
    logger.info(
        f"[add_windows] Sill: {sill_height:.2f} m | Head: {head_height:.2f} m | WWR → N:{wwr_map['N']:.2f} S:{wwr_map['S']:.2f} E:{wwr_map['E']:.2f} W:{wwr_map['W']:.2f}"
    )
    room_models: List[RoomModel] = []
    for room_3d in extrusion_result.rooms_3d:
        room_id = room_3d.identifier
        centroid_2d = room_3d.centroid_2d
        zone_type = room_3d.zone_type
        face_models: List[FaceModel] = []
        for face in room_3d.faces:
            if not face.is_exterior_wall:
                face_models.append(
                    FaceModel(
                        identifier=face.identifier,
                        owner_zone_id=room_id,
                        vertices=face.vertices,
                        face_type=face.face_type,
                        bc_type=face.bc_type,
                        is_exterior_wall=False,
                        orientation=None,
                        apertures=[],
                        out_shading=[],
                        adjacent_to=face.adjacent_to,
                    )
                )
                continue
            nx, ny = _wall_outward_normal_2d(face.vertices)
            orientation = _normal_to_orientation(nx, ny)
            wwr = wwr_map[orientation]
            apertures: List[ApertureGeom] = []
            out_shades: List[OutShadingGeom] = []
            apt_result = _compute_aperture_vertices(
                wall_verts=face.vertices, sill_height=sill_height, head_height=head_height, wwr=wwr
            )
            if apt_result is not None:
                apt_verts, actual_wwr = apt_result
                apt_id = f"{face.identifier}_apt00"
                win_w = math.hypot(
                    apt_verts[1][0] - apt_verts[0][0], apt_verts[1][1] - apt_verts[0][1]
                )
                win_h = apt_verts[2][2] - apt_verts[0][2]
                apt_area = round(win_w * win_h, 4)
                if exterior_has_map[orientation]:
                    depth = exterior_depth_map[orientation]
                    shade_verts = _compute_exterior_shade_vertices(
                        aperture_verts=apt_verts,
                        wall_verts=face.vertices,
                        depth=depth,
                        overhang=cfg.EXTERIOR_SHADE_OVERHANG,
                        gap=cfg.EXTERIOR_SHADE_GAP,
                    )
                    out_shades.append(
                        OutShadingGeom(
                            identifier=f"{apt_id}_ext_shade",
                            vertices=shade_verts,
                            depth=depth,
                            overhang=cfg.EXTERIOR_SHADE_OVERHANG,
                            gap=cfg.EXTERIOR_SHADE_GAP,
                        )
                    )
                if use_fixed_shade:
                    if generate_shade and fixed_coverage > 0.0:
                        lower_verts, upper_verts = _split_aperture_vertices(
                            apt_verts, fixed_coverage
                        )
                        win_width = math.hypot(
                            apt_verts[1][0] - apt_verts[0][0], apt_verts[1][1] - apt_verts[0][1]
                        )
                        lower_height = (1.0 - fixed_coverage) * win_h
                        upper_height = fixed_coverage * win_h
                        lower_apt = ApertureGeom(
                            identifier=f"{apt_id}_clear",
                            vertices=lower_verts,
                            orientation=orientation,
                            area=round(win_width * lower_height, 4),
                            wwr=actual_wwr,
                            coverage=0.0,
                        )
                        upper_apt = ApertureGeom(
                            identifier=f"{apt_id}_shaded",
                            vertices=upper_verts,
                            orientation=orientation,
                            area=round(win_width * upper_height, 4),
                            wwr=actual_wwr,
                            coverage=fixed_coverage,
                        )
                        apertures.extend([lower_apt, upper_apt])
                    else:
                        apertures.append(
                            ApertureGeom(
                                identifier=apt_id,
                                vertices=apt_verts,
                                orientation=orientation,
                                area=apt_area,
                                wwr=actual_wwr,
                                coverage=0.0,
                            )
                        )
                elif rng.random() < cfg.INTERIOR_SHADE_PROBABILITY:
                    coverage = round(rng.uniform(*cfg.INTERIOR_SHADE_COVERAGE_RANGE), 3)
                    if coverage > 0.0:
                        lower_verts, upper_verts = _split_aperture_vertices(apt_verts, coverage)
                        win_width = math.hypot(
                            apt_verts[1][0] - apt_verts[0][0], apt_verts[1][1] - apt_verts[0][1]
                        )
                        lower_height = (1.0 - coverage) * win_h
                        upper_height = coverage * win_h
                        lower_apt = ApertureGeom(
                            identifier=f"{apt_id}_clear",
                            vertices=lower_verts,
                            orientation=orientation,
                            area=round(win_width * lower_height, 4),
                            wwr=actual_wwr,
                            coverage=0.0,
                        )
                        upper_apt = ApertureGeom(
                            identifier=f"{apt_id}_shaded",
                            vertices=upper_verts,
                            orientation=orientation,
                            area=round(win_width * upper_height, 4),
                            wwr=actual_wwr,
                            coverage=coverage,
                        )
                        apertures.extend([lower_apt, upper_apt])
                    else:
                        apertures.append(
                            ApertureGeom(
                                identifier=apt_id,
                                vertices=apt_verts,
                                orientation=orientation,
                                area=apt_area,
                                wwr=actual_wwr,
                                coverage=0.0,
                            )
                        )
                else:
                    apertures.append(
                        ApertureGeom(
                            identifier=apt_id,
                            vertices=apt_verts,
                            orientation=orientation,
                            area=apt_area,
                            wwr=actual_wwr,
                            coverage=0.0,
                        )
                    )
            for apt in apertures:
                apt.owner_zone_id = room_id
                apt.host_face_id = face.identifier
                apt.aperture_group_id = apt_id
            for shade in out_shades:
                shade.owner_zone_id = room_id
                shade.host_face_id = face.identifier
                shade.host_aperture_group_id = apt_id
            face_models.append(
                FaceModel(
                    identifier=face.identifier,
                    owner_zone_id=room_id,
                    vertices=face.vertices,
                    face_type=face.face_type,
                    bc_type=face.bc_type,
                    is_exterior_wall=True,
                    orientation=orientation,
                    apertures=apertures,
                    out_shading=out_shades,
                    adjacent_to=face.adjacent_to,
                )
            )
        room_models.append(
            RoomModel(
                identifier=room_id,
                centroid_2d=centroid_2d,
                zone_type=zone_type,
                floor_area=room_3d.floor_area,
                floor_index=room_3d.floor_index,
                faces=face_models,
            )
        )
    total_area = round(sum((r.floor_area for r in room_models)), 3)
    model = BuildingModel(
        shape_type=division_result.footprint.geom_type,
        n_floors=extrusion_result.n_floors,
        building_type=extrusion_result.building_type,
        floor_height=extrusion_result.floor_height,
        grid_spacing=division_result.grid_spacing,
        floor_bc=extrusion_result.floor_bc,
        ceiling_bc=extrusion_result.ceiling_bc,
        total_floor_area=total_area,
        sill_height=sill_height,
        head_height=head_height,
        wwr_by_orientation=wwr_map,
        rotation_angle_deg=getattr(division_result, "rotation_angle_deg", 0.0),
        rooms=room_models,
        out_shade_depth_by_orientation={
            orient: exterior_depth_map[orient] if exterior_has_map[orient] else 0.0
            for orient in ("N", "S", "E", "W")
        },
        out_shade_gap=cfg.EXTERIOR_SHADE_GAP,
    )
    logger.info(
        f"[add_windows] Rooms: {model.n_rooms} | Total windows: {model.total_windows} | Total window area: {model.total_window_area:.1f} m²"
    )
    return model
