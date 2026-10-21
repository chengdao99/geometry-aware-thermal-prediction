"""Parameterized building rotate building.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

from __future__ import annotations
import math
import random
from typing import List, Tuple
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Polygon
from . import geometric_config as cfg
from .divide_rooms import RoomDivisionResult
import logging

logger = logging.getLogger(__name__)


def _rotate_polygon(poly: Polygon, angle_deg: float) -> Polygon:
    return shapely_rotate(poly, angle_deg, origin=(0, 0), use_radians=False)


def _rotate_seeds(seeds: List[Tuple[float, float]], angle_deg: float) -> List[Tuple[float, float]]:
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in seeds]


def rotate_building(
    division_result: RoomDivisionResult | None,
    seed: int | None = None,
    mode: str | None = None,
    probability: float | None = None,
    fixed_angle_deg: float | None = None,
) -> RoomDivisionResult | None:
    """Rotate a room partition while preserving its topology and room identities."""
    if division_result is None:
        logger.error("[rotate_building] Received None from upstream — skipping rotation.")
        return None
    rng = random.Random(seed)
    prob = probability if probability is not None else cfg.ROTATION_PROBABILITY
    mode = mode if mode is not None else cfg.ROTATION_MODE
    if fixed_angle_deg is not None:
        angle_deg = float(fixed_angle_deg)
        logger.info(
            f"[rotate_building] Fixed angle: {angle_deg:.2f}° | Rooms: {division_result.n_rooms}"
        )
    elif rng.random() > prob:
        logger.info(f"[rotate_building] Skipped (p={prob:.2f}) | rotation_angle_deg = 0.0°")
        return RoomDivisionResult(
            footprint=division_result.footprint,
            rooms=division_result.rooms,
            zone_types=division_result.zone_types,
            room_ids=division_result.room_ids,
            seeds=division_result.seeds,
            shape_type=division_result.shape_type,
            grid_spacing=division_result.grid_spacing,
            rotation_angle_deg=0.0,
        )
    elif mode == "quarter":
        angle_deg = rng.uniform(0.0, 90.0)
    else:
        angle_deg = rng.uniform(0.0, 360.0)
    rotated_footprint = _rotate_polygon(division_result.footprint, angle_deg)
    rotated_rooms = [_rotate_polygon(room, angle_deg) for room in division_result.rooms]
    rotated_seeds = _rotate_seeds(division_result.seeds, angle_deg)
    rotated_ids = list(division_result.room_ids)
    area_before = division_result.footprint.area
    area_after = rotated_footprint.area
    area_delta = abs(area_after - area_before)
    if area_delta > 1e-06:
        logger.warning(
            f"[rotate_building] WARNING: footprint area changed by {area_delta:.2e} m² (before={area_before:.4f}, after={area_after:.4f}) — check for degenerate geometry."
        )
    logger.info(
        f"[rotate_building] Mode: {mode:8s} | Angle: {angle_deg:7.2f}° | Rooms: {len(rotated_rooms)} | Area: {area_before:.2f} m²  →  {area_after:.2f} m²"
    )
    return RoomDivisionResult(
        footprint=rotated_footprint,
        rooms=rotated_rooms,
        zone_types=division_result.zone_types,
        room_ids=rotated_ids,
        seeds=rotated_seeds,
        shape_type=division_result.shape_type,
        grid_spacing=division_result.grid_spacing,
        rotation_angle_deg=round(angle_deg, 4),
    )
