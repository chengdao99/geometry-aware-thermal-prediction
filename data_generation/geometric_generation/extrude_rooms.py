"""Parameterized building extrude rooms.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from shapely.geometry import LineString, Point
import logging

logger = logging.getLogger(__name__)
from . import geometric_config as cfg
from .divide_rooms import RoomDivisionResult

FACE_TYPE_FLOOR = "Floor"
FACE_TYPE_WALL = "Wall"
FACE_TYPE_ROOF = "RoofCeiling"
BC_OUTDOORS = "Outdoors"
BC_GROUND = "Ground"
BC_ADIABATIC = "Adiabatic"
BC_SURFACE = "Surface"


@dataclass
class Face3D:
    """Represent face3 d and its associated geometry or physical configuration."""

    identifier: str
    vertices: List[Tuple[float, float, float]]
    face_type: str
    bc_type: str
    is_exterior_wall: bool = False
    adjacent_to: Optional[str] = None


@dataclass
class Room3D:
    """Represent room3 d and its associated geometry or physical configuration."""

    identifier: str
    faces: List[Face3D]
    zone_type: str
    floor_area: float
    floor_index: int = 0
    z_bottom: float = 0.0
    centroid_2d: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    @property
    def n_faces(self) -> int:
        """Compute n faces for the current building or model state."""
        return len(self.faces)

    @property
    def exterior_walls(self) -> List[Face3D]:
        """Compute exterior walls for the current building or model state."""
        return [f for f in self.faces if f.is_exterior_wall]

    @property
    def interior_walls(self) -> List[Face3D]:
        """Compute interior walls for the current building or model state."""
        return [f for f in self.faces if f.face_type == FACE_TYPE_WALL and (not f.is_exterior_wall)]


@dataclass
class ExtrusionResult:
    """Represent extrusion result and its associated geometry or physical configuration."""

    rooms_3d: List[Room3D]
    n_floors: int
    floor_height: float
    floor_bc: str
    ceiling_bc: str
    building_type: str

    @property
    def n_rooms(self) -> int:
        """Compute n rooms for the current building or model state."""
        return len(self.rooms_3d)

    @property
    def n_rooms_per_floor(self) -> int:
        """Compute n rooms per floor for the current building or model state."""
        return self.n_rooms // self.n_floors if self.n_floors > 0 else 0

    @property
    def total_height(self) -> float:
        """Compute total height for the current building or model state."""
        return self.n_floors * self.floor_height

    @property
    def total_exterior_walls(self) -> int:
        """Compute total exterior walls for the current building or model state."""
        return sum((len(r.exterior_walls) for r in self.rooms_3d))

    @property
    def total_interior_walls(self) -> int:
        """Compute total interior walls for the current building or model state."""
        return sum((len(r.interior_walls) for r in self.rooms_3d))

    def rooms_on_floor(self, floor_index: int) -> List[Room3D]:
        """Compute rooms on floor for the current building or model state. Inputs: floor_index."""
        return [r for r in self.rooms_3d if r.floor_index == floor_index]


def _sample_bc_from_probs(probs: dict[str, float], rng: random.Random) -> str:
    bc_types = list(probs.keys())
    weights = [probs[k] for k in bc_types]
    return rng.choices(bc_types, weights=weights, k=1)[0]


def _resolve_floor_bcs(
    floor_index: int, n_floors: int, building_type: str, floor_bc: str, ceiling_bc: str
) -> tuple[str, str]:
    is_bottom = floor_index == 0
    is_top = floor_index == n_floors - 1
    is_single = n_floors == 1
    if building_type == "tall_mid":
        this_floor_bc = BC_ADIABATIC if is_bottom else BC_SURFACE
        this_ceiling_bc = BC_ADIABATIC if is_top else BC_SURFACE
    elif is_single:
        this_floor_bc = floor_bc
        this_ceiling_bc = ceiling_bc
    elif is_bottom:
        this_floor_bc = floor_bc
        this_ceiling_bc = BC_SURFACE
    elif is_top:
        this_floor_bc = BC_SURFACE
        this_ceiling_bc = ceiling_bc
    else:
        this_floor_bc = BC_SURFACE
        this_ceiling_bc = BC_SURFACE
    return (this_floor_bc, this_ceiling_bc)


def _build_all_boundary_lines(footprint) -> list[LineString]:
    lines = [LineString(list(footprint.exterior.coords))]
    for interior in footprint.interiors:
        lines.append(LineString(list(interior.coords)))
    return lines


def _is_exterior_wall(
    p0: tuple[float, float], p1: tuple[float, float], boundary_lines: list[LineString], tol: float
) -> bool:
    midpoint = Point((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    return any((line.distance(midpoint) <= tol for line in boundary_lines))


def _extrude_one_room(
    base_room_id: str,
    room_polygon,
    zone_type: str,
    floor_height: float,
    floor_bc: str,
    ceiling_bc: str,
    exterior_line: list[LineString],
    tol: float,
    floor_index: int = 0,
    z_base: float = 0.0,
) -> Room3D:
    z_top = z_base + floor_height
    cx = round(room_polygon.centroid.x, 3)
    cy = round(room_polygon.centroid.y, 3)
    cz_mid = z_base + floor_height / 2.0
    room_id = f"room_{base_room_id}_f{floor_index}"
    ring_coords = list(room_polygon.exterior.coords)[:-1]
    floor_verts = [(x, y, z_base) for x, y in ring_coords]
    floor_face = Face3D(
        identifier=f"{room_id}_f00",
        vertices=floor_verts,
        face_type=FACE_TYPE_FLOOR,
        bc_type=floor_bc,
    )
    ceiling_verts = [(x, y, z_top) for x, y in reversed(ring_coords)]
    ceiling_face = Face3D(
        identifier=f"{room_id}_f01",
        vertices=ceiling_verts,
        face_type=FACE_TYPE_ROOF,
        bc_type=ceiling_bc,
    )
    wall_faces: list[Face3D] = []
    n = len(ring_coords)
    for i in range(n):
        p0 = ring_coords[i]
        p1 = ring_coords[(i + 1) % n]
        is_ext = _is_exterior_wall(p0, p1, exterior_line, tol)
        wall_verts = [
            (p0[0], p0[1], z_base),
            (p1[0], p1[1], z_base),
            (p1[0], p1[1], z_top),
            (p0[0], p0[1], z_top),
        ]
        wall_faces.append(
            Face3D(
                identifier=f"{room_id}_f{i + 2:02d}",
                vertices=wall_verts,
                face_type=FACE_TYPE_WALL,
                bc_type=BC_OUTDOORS if is_ext else BC_SURFACE,
                is_exterior_wall=is_ext,
            )
        )
    all_faces = [floor_face, ceiling_face] + wall_faces
    return Room3D(
        identifier=room_id,
        faces=all_faces,
        zone_type=zone_type,
        floor_area=round(room_polygon.area, 3),
        floor_index=floor_index,
        z_bottom=z_base,
        centroid_2d=(cx, cy),
    )


def _solve_adjacency_3d(rooms_3d: list[Room3D]) -> None:
    vset_to_faces: dict = {}
    for room in rooms_3d:
        for face in room.faces:
            if face.bc_type != BC_SURFACE:
                continue
            vset = frozenset(
                ((round(v[0], 3), round(v[1], 3), round(v[2], 3)) for v in face.vertices)
            )
            vset_to_faces.setdefault(vset, []).append(face)
    for vset, faces in vset_to_faces.items():
        if len(faces) == 2:
            faces[0].adjacent_to = faces[1].identifier
            faces[1].adjacent_to = faces[0].identifier
        else:
            logger.warning(
                f"[solve_adjacency] Mismatch: {len(faces)} faces share the same vertex set — expected exactly 2."
            )


def stack_floors(
    division_result: RoomDivisionResult | None, seed: int | None = None, overrides=None
) -> ExtrusionResult | None:
    """Extrude room partitions into floors with consistent inter-room boundary conditions."""
    if division_result is None:
        logger.error("[stack_floors] Received None from upstream — skipping extrusion.")
        return None
    rng = random.Random(seed)
    footprint_area = division_result.footprint.area
    if overrides and overrides.n_floors is not None:
        n_floors = overrides.n_floors
        building_type = "real"
    else:
        if footprint_area <= cfg.LOW_RISE_AREA_THRESHOLD:
            storey_weights_dict = cfg.STOREY_COUNT_WEIGHTS_LOW_RISE
            logger.info(
                f"[stack_floors] Low-rise building (area={footprint_area:.1f} m² ≤ {cfg.LOW_RISE_AREA_THRESHOLD}) → using low-rise floor weights (1–3 stories only)"
            )
        else:
            storey_weights_dict = cfg.STOREY_COUNT_WEIGHTS_NORMAL
            logger.info(
                f"[stack_floors] Normal building (area={footprint_area:.1f} m² > {cfg.LOW_RISE_AREA_THRESHOLD}) → using normal floor weights"
            )
        storey_choices = list(storey_weights_dict.keys())
        storey_weights = list(storey_weights_dict.values())
        storey_choice = rng.choices(storey_choices, weights=storey_weights, k=1)[0]
        if storey_choice == "tall_mid":
            n_floors = cfg.TALL_BUILDING_SIM_FLOORS
            building_type = "tall_mid"
        else:
            n_floors = int(storey_choice)
            building_type = "real"
    if overrides and overrides.floor_height is not None:
        floor_height = overrides.floor_height
    else:
        lo, hi = cfg.FLOOR_HEIGHT_RANGE
        floor_height = rng.randint(round(lo * 10), round(hi * 10)) / 10.0
    if overrides and overrides.floor_bc:
        floor_bc = overrides.floor_bc
    elif building_type == "tall_mid":
        floor_bc = BC_ADIABATIC
    else:
        floor_bc = _sample_bc_from_probs(cfg.FLOOR_BC_PROBS, rng)
    if overrides and overrides.ceiling_bc:
        ceiling_bc = overrides.ceiling_bc
    elif building_type == "tall_mid":
        ceiling_bc = BC_ADIABATIC
    else:
        ceiling_bc = _sample_bc_from_probs(cfg.CEILING_BC_PROBS, rng)
    logger.info(
        f"[stack_floors] Type: {building_type} | Floors: {n_floors} | Height: {floor_height:.1f} m | Floor BC: {floor_bc} | Ceiling BC: {ceiling_bc}"
    )
    boundary_lines = _build_all_boundary_lines(division_result.footprint)
    tol = cfg.EXTERIOR_WALL_TOLERANCE
    all_rooms: list[Room3D] = []
    for floor_index in range(n_floors):
        z_base = floor_index * floor_height
        this_floor_bc, this_ceiling_bc = _resolve_floor_bcs(
            floor_index=floor_index,
            n_floors=n_floors,
            building_type=building_type,
            floor_bc=floor_bc,
            ceiling_bc=ceiling_bc,
        )
        for room_poly, zone_type, base_room_id in zip(
            division_result.rooms, division_result.zone_types, division_result.room_ids
        ):
            room_3d = _extrude_one_room(
                base_room_id=base_room_id,
                room_polygon=room_poly,
                zone_type=zone_type,
                floor_height=floor_height,
                floor_bc=this_floor_bc,
                ceiling_bc=this_ceiling_bc,
                exterior_line=boundary_lines,
                tol=tol,
                floor_index=floor_index,
                z_base=z_base,
            )
            all_rooms.append(room_3d)
        logger.info(
            f"[stack_floors]   Floor {floor_index} | z={z_base:.1f}–{z_base + floor_height:.1f} m | Floor BC: {this_floor_bc} | Ceiling BC: {this_ceiling_bc}"
        )
    _solve_adjacency_3d(all_rooms)
    total_ext = sum((len(r.exterior_walls) for r in all_rooms))
    total_int = sum((len(r.interior_walls) for r in all_rooms))
    logger.info(
        f"[stack_floors] Total rooms: {len(all_rooms)} | Exterior walls: {total_ext} | Interior walls: {total_int}"
    )
    return ExtrusionResult(
        rooms_3d=all_rooms,
        n_floors=n_floors,
        floor_height=floor_height,
        floor_bc=floor_bc,
        ceiling_bc=ceiling_bc,
        building_type=building_type,
    )
