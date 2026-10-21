"""Parameterized building retry divide rooms.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

from __future__ import annotations
import math
import random
from shapely.geometry import LineString, Point, Polygon
import shapely.errors
import logging

logger = logging.getLogger(__name__)
from . import geometric_config as cfg
from .generate_footprint import FootprintResult
from .divide_rooms import RoomDivisionResult, divide_rooms


def _all_boundary_lines(footprint: Polygon) -> list[LineString]:
    lines = [LineString(list(footprint.exterior.coords))]
    for interior in footprint.interiors:
        lines.append(LineString(list(interior.coords)))
    return lines


def _edge_midpoint_on_boundary(
    p0: tuple[float, float], p1: tuple[float, float], lines: list[LineString], tol: float
) -> bool:
    mid = Point((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    return any((line.distance(mid) <= tol for line in lines))


def _all_edges_long_enough(
    division_result: RoomDivisionResult, min_ext_length: float, min_int_length: float, tol: float
) -> bool:
    boundary_lines = _all_boundary_lines(division_result.footprint)
    for room in division_result.rooms:
        coords = list(room.exterior.coords)
        for i in range(len(coords) - 1):
            p0, p1 = (coords[i], coords[i + 1])
            length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            is_ext = _edge_midpoint_on_boundary(p0, p1, boundary_lines, tol)
            if is_ext:
                if length < min_ext_length:
                    return False
            elif length < min_int_length:
                return False
    return True


def _all_rooms_small_enough(division_result: RoomDivisionResult, max_area: float) -> bool:
    return all((room.area <= max_area for room in division_result.rooms))


def _all_shared_edges_consistent(division_result: RoomDivisionResult, tol: float = 0.012) -> bool:
    rooms = division_result.rooms
    n = len(rooms)
    for i in range(n):
        for j in range(i + 1, n):
            shared = rooms[i].boundary.intersection(rooms[j].boundary)
            if shared.is_empty or shared.length < tol:
                continue
            len_i = rooms[i].boundary.intersection(shared.buffer(tol)).length
            len_j = rooms[j].boundary.intersection(shared.buffer(tol)).length
            if abs(len_i - len_j) > tol:
                return False
    return True


def find_short_edges(
    division_result: RoomDivisionResult,
    min_ext_length: float | None = None,
    min_int_length: float | None = None,
    tol: float | None = None,
) -> list[dict]:
    """Compute find short edges for the current building or model state. Inputs: division_result, min_ext_length, min_int_length, tol."""
    min_ext = min_ext_length if min_ext_length is not None else cfg.MIN_EXTERIOR_EDGE_LENGTH
    min_int = min_int_length if min_int_length is not None else cfg.MIN_INTERIOR_EDGE_LENGTH
    tol_dist = tol if tol is not None else cfg.EXTERIOR_WALL_TOLERANCE
    boundary_lines = _all_boundary_lines(division_result.footprint)
    offenders: list[dict] = []
    for room_idx, room in enumerate(division_result.rooms):
        coords = list(room.exterior.coords)
        for i in range(len(coords) - 1):
            p0, p1 = (coords[i], coords[i + 1])
            length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            is_ext = _edge_midpoint_on_boundary(p0, p1, boundary_lines, tol_dist)
            limit = min_ext if is_ext else min_int
            if length < limit:
                offenders.append(
                    dict(
                        room_index=room_idx,
                        edge_index=i,
                        is_exterior=is_ext,
                        length=round(length, 4),
                        p0=p0,
                        p1=p1,
                    )
                )
    return offenders


def _run_with_retry(
    footprint_result: FootprintResult,
    base_spacing: float,
    max_tries: int,
    min_ext_len: float,
    min_int_len: float,
    max_area: float,
    tol: float,
    seed: int | None,
) -> tuple[RoomDivisionResult | None, RoomDivisionResult | None, int]:
    rng = random.Random(seed)
    last_attempt: RoomDivisionResult | None = None
    for attempt in range(max_tries):
        if attempt == 0:
            spacing = base_spacing
            offset = (0.0, 0.0)
        else:
            spacing = base_spacing * rng.uniform(
                cfg.DIVIDE_RETRY_SPACING_MIN_FACTOR, cfg.DIVIDE_RETRY_SPACING_MAX_FACTOR
            )
            offset = (rng.uniform(0.0, spacing), rng.uniform(0.0, spacing))
        child_seed = None if seed is None else seed + attempt
        try:
            division = divide_rooms(
                footprint_result, grid_spacing=spacing, grid_offset=offset, seed=child_seed
            )
        except (RuntimeError, shapely.errors.GEOSException) as exc:
            logger.warning(
                f"[retry_divide_rooms] Attempt {attempt + 1}/{max_tries} — divide_rooms Exception: {exc}"
            )
            continue
        last_attempt = division
        edge_ok = _all_edges_long_enough(division, min_ext_len, min_int_len, tol)
        area_ok = _all_rooms_small_enough(division, max_area)
        adj_ok = _all_shared_edges_consistent(division)
        if edge_ok and area_ok and adj_ok:
            n_peri = division.zone_types.count("perimeter")
            n_core = division.zone_types.count("core")
            logger.info(
                f"[retry_divide_rooms] Rooms: {division.n_rooms:3d} | Grid: {spacing:.2f} m | Perimeter: {n_peri} | Core: {n_core} | Attempts: {attempt + 1}/{max_tries}"
            )
            return (division, division, attempt + 1)
        reasons = []
        if not edge_ok:
            reasons.append("exterior edge too short")
        if not area_ok:
            max_found = max((r.area for r in division.rooms))
            reasons.append(f"room too large ({max_found:.0f} m²)")
        logger.warning(
            f"[retry_divide_rooms] ✗ Attempt {attempt + 1}/{max_tries} — {', '.join(reasons)} (spacing={spacing:.2f} m). Retrying…"
        )
    logger.warning(
        f"[retry_divide_rooms] WARNING — all {max_tries} attempts failed for footprint '{footprint_result.shape_type}' (area={footprint_result.area:.1f} m²). Returning None; caller should log and skip this sample."
    )
    return (None, last_attempt, max_tries)


def validated_divide_rooms(
    footprint_result: FootprintResult,
    grid_spacing: float | None = None,
    seed: int | None = None,
    *,
    max_retries: int | None = None,
    min_ext_length: float | None = None,
    min_int_length: float | None = None,
    max_room_area: float | None = None,
) -> tuple[RoomDivisionResult | None, int]:
    """Retry room division deterministically while enforcing geometric constraints."""
    base_spacing = grid_spacing if grid_spacing is not None else cfg.GRID_SPACING
    max_tries = max_retries if max_retries is not None else cfg.MAX_DIVIDE_RETRIES
    min_ext = min_ext_length if min_ext_length is not None else cfg.MIN_EXTERIOR_EDGE_LENGTH
    min_int = min_int_length if min_int_length is not None else cfg.MIN_INTERIOR_EDGE_LENGTH
    max_area = max_room_area if max_room_area is not None else cfg.MAX_ROOM_AREA
    tol = cfg.EXTERIOR_WALL_TOLERANCE
    accepted, _, n_tries = _run_with_retry(
        footprint_result, base_spacing, max_tries, min_ext, min_int, max_area, tol, seed
    )
    return (accepted, n_tries)
