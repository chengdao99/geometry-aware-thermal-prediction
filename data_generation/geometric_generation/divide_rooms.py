"""Parameterized building divide rooms.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Tuple
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon, GeometryCollection
from shapely.ops import unary_union, voronoi_diagram, polygonize
from shapely.validation import make_valid
import logging

logger = logging.getLogger(__name__)
from . import geometric_config as cfg
from .generate_footprint import FootprintResult


@dataclass
class RoomDivisionResult:
    """Represent room division result and its associated geometry or physical configuration."""

    footprint: Polygon
    rooms: List[Polygon]
    zone_types: List[str]
    room_ids: List[str]
    seeds: List[Tuple]
    shape_type: str = ""
    grid_spacing: float = 0.0
    rotation_angle_deg: float = 0.0

    @property
    def n_rooms(self) -> int:
        """Compute n rooms for the current building or model state."""
        return len(self.rooms)

    @property
    def areas(self) -> List[float]:
        """Compute areas for the current building or model state."""
        return [round(r.area, 3) for r in self.rooms]


def _generate_grid_seeds(
    footprint: Polygon, grid_spacing: float, grid_offset: tuple[float, float] = (0.0, 0.0)
) -> List[Tuple[float, float]]:
    min_x, min_y, max_x, max_y = footprint.bounds
    dx, dy = grid_offset
    xs = []
    x = min_x + grid_spacing / 2.0 + dx
    while x < max_x:
        xs.append(x)
        x += grid_spacing
    ys = []
    y = min_y + grid_spacing / 2.0 + dy
    while y < max_y:
        ys.append(y)
        y += grid_spacing
    seeds = [(x, y) for x in xs for y in ys if footprint.contains(Point(x, y))]
    return seeds


def _voronoi_and_clip(seeds: List[Tuple[float, float]], footprint: Polygon) -> List[Polygon]:
    multi_pt = MultiPoint(seeds)
    envelope = footprint.buffer(50.0).envelope
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regions = voronoi_diagram(multi_pt, envelope=envelope)
    rooms: List[Polygon] = []
    for region in regions.geoms:
        clipped = region.intersection(footprint)
        clipped = make_valid(clipped)
        if clipped.is_empty:
            continue
        if isinstance(clipped, Polygon):
            if clipped.area > 1e-06:
                rooms.append(clipped)
        elif isinstance(clipped, (MultiPolygon, GeometryCollection)):
            for geom in clipped.geoms:
                if isinstance(geom, Polygon) and geom.area > 1e-06:
                    rooms.append(geom)
    return rooms


def _find_best_merge_target(small_room: Polygon, candidates: List[Polygon]) -> int | None:
    best_idx = None
    best_length = 0.0
    small_boundary = small_room.boundary
    for idx, candidate in enumerate(candidates):
        shared = small_boundary.intersection(candidate.boundary)
        shared_len = shared.length if not shared.is_empty else 0.0
        if shared_len > best_length:
            best_length = shared_len
            best_idx = idx
    return best_idx


def _merge_small_rooms(rooms: List[Polygon], min_area: float) -> List[Polygon]:
    rooms = list(rooms)
    for _iteration in range(cfg.MAX_MERGE_ITERATIONS):
        small_rooms = sorted(
            [(i, r) for i, r in enumerate(rooms) if r.area < min_area], key=lambda x: x[1].area
        )
        if not small_rooms:
            break
        small_idx, small_room = small_rooms[0]
        candidates = [r for i, r in enumerate(rooms) if i != small_idx]
        merge_idx_in_candidates = _find_best_merge_target(small_room, candidates)
        if merge_idx_in_candidates is None:
            logger.warning(
                f"  [merge] Discarded isolated fragment (area={small_room.area:.2f} m², no neighbours found)."
            )
            rooms.pop(small_idx)
            continue
        target_room = candidates[merge_idx_in_candidates]
        merged = make_valid(unary_union([small_room, target_room]))
        target_idx_in_rooms = rooms.index(target_room)
        remove_indices = sorted([small_idx, target_idx_in_rooms], reverse=True)
        for idx in remove_indices:
            rooms.pop(idx)
        if isinstance(merged, MultiPolygon):
            merged = max(merged.geoms, key=lambda g: g.area)
        if isinstance(merged, Polygon) and (not merged.is_empty):
            rooms.append(merged)
    return rooms


def _clean_topology(rooms: List[Polygon], tol: float = 0.01) -> List[Polygon]:
    if not rooms:
        return []
    occupied_geometry = make_valid(unary_union(rooms))
    lines = unary_union([room.boundary for room in rooms])
    simplified_lines = lines.simplify(tol, preserve_topology=True)
    new_polygons = list(polygonize(simplified_lines))
    cleaned_rooms = [
        polygon
        for polygon in new_polygons
        if not polygon.is_empty
        and polygon.area > 1e-06
        and occupied_geometry.covers(polygon.representative_point())
    ]
    if len(cleaned_rooms) != len(rooms):
        raise RuntimeError(
            f"[clean_topology] Polygonization changed room count: {len(rooms)} -> {len(cleaned_rooms)}"
        )
    overlap_tolerance = max(1e-08, tol * tol)
    for i, room_a in enumerate(cleaned_rooms):
        for room_b in cleaned_rooms[i + 1 :]:
            overlap_area = room_a.intersection(room_b).area
            if overlap_area > overlap_tolerance:
                raise RuntimeError(
                    f"[clean_topology] Rebuilt rooms overlap by {overlap_area:.6g} m²"
                )
    return cleaned_rooms


def _label_zone_types(rooms: List[Polygon], footprint: Polygon, tol: float = 0.05) -> List[str]:
    exterior_boundary = footprint.exterior
    zone_types = []
    for room in rooms:
        shared = room.boundary.intersection(exterior_boundary)
        if shared.is_empty or shared.length < tol:
            zone_types.append("core")
        else:
            zone_types.append("perimeter")
    return zone_types


def build_room_identifiers(room_polygons: List[Polygon]) -> List[str]:
    """Compute build room identifiers for the current building or model state. Inputs: room_polygons."""
    used: dict[str, int] = {}
    identifiers: List[str] = []
    for poly in room_polygons:
        cx = poly.centroid.x
        cy = poly.centroid.y
        base = f"{int(cx)}_{int(cy)}"
        if base not in used:
            used[base] = 1
            identifiers.append(base)
        else:
            used[base] += 1
            identifiers.append(f"{base}_v{used[base]}")
    return identifiers


def divide_rooms(
    footprint_result: FootprintResult,
    grid_spacing: float | None = None,
    grid_offset: tuple[float, float] | None = None,
    seed: int | None = None,
) -> RoomDivisionResult:
    """Partition a footprint into perimeter and core rooms using seeded geometry."""
    import random as _random

    rng = _random.Random(seed)
    footprint = footprint_result.polygon
    grid_spacing = grid_spacing if grid_spacing is not None else cfg.GRID_SPACING
    min_area = cfg.MIN_ROOM_AREA
    _offset = grid_offset if grid_offset is not None else (0.0, 0.0)
    interior_seeds = _generate_grid_seeds(footprint, grid_spacing, _offset)
    if len(interior_seeds) < cfg.MIN_SEED_COUNT:
        centroid = footprint.centroid
        for _ in range(cfg.MIN_SEED_COUNT - len(interior_seeds)):
            min_x, min_y, max_x, max_y = footprint.bounds
            for _attempt in range(500):
                rx = rng.uniform(min_x, max_x)
                ry = rng.uniform(min_y, max_y)
                if footprint.contains(Point(rx, ry)):
                    interior_seeds.append((rx, ry))
                    break
    if not interior_seeds:
        raise RuntimeError(
            f"[divide_rooms] No interior seed points could be generated for '{footprint_result.shape_type}' with grid_spacing={grid_spacing} m. Consider reducing GRID_SPACING in config_hyperparams.py."
        )
    all_seeds = interior_seeds
    logger.info(
        f"[divide_rooms] Shape: {footprint_result.shape_type:15s} | Interior seeds: {len(interior_seeds):3d} | Grid spacing: {grid_spacing} m"
    )
    rooms = _voronoi_and_clip(all_seeds, footprint)
    if not rooms:
        raise RuntimeError(
            "[divide_rooms] Voronoi clipping produced no valid room polygons. This is likely a geometry issue — check the footprint validity."
        )
    n_before = len(rooms)
    rooms = _merge_small_rooms(rooms, min_area)
    rooms = _clean_topology(rooms, tol=0.01)
    n_after = len(rooms)
    rooms.sort(key=lambda r: r.area, reverse=True)
    logger.info(
        f"[divide_rooms] Rooms before merge: {n_before:3d} | After merge: {n_after:3d} | Min area threshold: {min_area} m²"
    )
    if cfg.LABEL_ZONE_TYPES:
        zone_types = _label_zone_types(rooms, footprint)
        n_peri = zone_types.count("perimeter")
        n_core = zone_types.count("core")
        logger.info(f"[divide_rooms] Zone types — perimeter: {n_peri} | core: {n_core}")
    else:
        zone_types = ["unknown"] * len(rooms)
    room_ids = build_room_identifiers(rooms)
    return RoomDivisionResult(
        footprint=footprint,
        rooms=rooms,
        zone_types=zone_types,
        room_ids=room_ids,
        seeds=interior_seeds,
        shape_type=footprint_result.shape_type,
        grid_spacing=grid_spacing,
    )
