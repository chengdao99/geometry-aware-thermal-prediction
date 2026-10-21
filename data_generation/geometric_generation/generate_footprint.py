"""Parameterized building generate footprint.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Callable
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.validation import make_valid
import logging

logger = logging.getLogger(__name__)
from . import geometric_config as cfg


@dataclass
class FootprintResult:
    """Represent footprint result and its associated geometry or physical configuration."""

    shape_type: str
    polygon: Polygon
    area: float
    bbox_w: float
    bbox_d: float


def _make_rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def _sample_base_dimensions(rng: random.Random, area_inflation: float = 1.0) -> tuple[float, float]:
    A_min, A_max = cfg.AREA_RANGE
    W_min, W_max = cfg.WIDTH_RANGE
    D_min, D_max = cfg.DEPTH_RANGE
    for _ in range(200):
        A_bbox = rng.uniform(A_min * area_inflation, A_max * area_inflation)
        r = rng.uniform(1.0, cfg.ASPECT_RATIO_MAX)
        if rng.random() < 0.5:
            r = 1.0 / r
        D = (A_bbox / r) ** 0.5
        W = D * r
        W = max(W_min, min(W_max, W))
        D = max(D_min, min(D_max, D))
        if max(W / D, D / W) <= cfg.ASPECT_RATIO_MAX:
            return (W, D)
    return (20.0, 20.0)


def _validate(poly: Polygon) -> bool:
    if poly is None or poly.is_empty:
        return False
    if not poly.is_valid:
        return False
    if isinstance(poly, MultiPolygon):
        return False
    A_min, A_max = cfg.AREA_RANGE
    if not A_min <= poly.area <= A_max:
        return False
    coords = list(poly.exterior.coords)
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        if (dx * dx + dy * dy) ** 0.5 < cfg.MIN_EDGE_LENGTH:
            return False
    return True


def _safe_difference(base: Polygon, cut: Polygon) -> Polygon | None:
    try:
        result = base.difference(cut)
        result = make_valid(result)
        if not isinstance(result, Polygon):
            return None
        return result
    except Exception:
        return None


def _bbox_from_polygon(poly: Polygon) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = poly.bounds
    return (max_x - min_x, max_y - min_y)


def _gen_rectangle(rng: random.Random) -> FootprintResult | None:
    W, D = _sample_base_dimensions(rng, area_inflation=1.0)
    poly = box(0.0, 0.0, W, D)
    if not _validate(poly):
        return None
    return FootprintResult(
        shape_type="rectangle",
        polygon=poly,
        area=round(poly.area, 3),
        bbox_w=round(W, 3),
        bbox_d=round(D, 3),
    )


def _gen_quadrilateral(rng: random.Random) -> FootprintResult | None:
    W, D = _sample_base_dimensions(rng, area_inflation=1.0)
    perturb = cfg.QUAD_PERTURB_FRACTION * min(W, D)

    def jitter() -> float:
        """Compute jitter for the current building or model state."""
        return rng.uniform(-perturb, perturb)

    base_corners = [(0.0, 0.0), (W, 0.0), (W, D), (0.0, D)]
    displaced = [(x + jitter(), y + jitter()) for x, y in base_corners]
    poly = Polygon(displaced)
    if not poly.is_valid:
        poly = make_valid(poly)
    excess_area = poly.difference(poly.convex_hull).area
    if excess_area > 0.01:
        return None
    if not _validate(poly):
        return None
    bbox_w, bbox_d = _bbox_from_polygon(poly)
    return FootprintResult(
        shape_type="quadrilateral",
        polygon=poly,
        area=round(poly.area, 3),
        bbox_w=round(bbox_w, 3),
        bbox_d=round(bbox_d, 3),
    )


def _gen_L_shape(rng: random.Random) -> FootprintResult | None:
    W, D = _sample_base_dimensions(rng, area_inflation=1.35)
    cut_w = rng.uniform(*cfg.L_CUT_W_RANGE) * W
    cut_d = rng.uniform(*cfg.L_CUT_D_RANGE) * D
    base = box(0.0, 0.0, W, D)
    corner_cuts = {
        "top_right": box(W - cut_w, D - cut_d, W, D),
        "top_left": box(0.0, D - cut_d, cut_w, D),
        "bottom_left": box(0.0, 0.0, cut_w, cut_d),
        "bottom_right": box(W - cut_w, 0.0, W, cut_d),
    }
    chosen = rng.choice(list(corner_cuts.values()))
    poly = _safe_difference(base, chosen)
    if poly is None or not _validate(poly):
        return None
    return FootprintResult(
        shape_type="L_shape",
        polygon=poly,
        area=round(poly.area, 3),
        bbox_w=round(W, 3),
        bbox_d=round(D, 3),
    )


def _gen_T_shape(rng: random.Random) -> FootprintResult | None:
    W, D = _sample_base_dimensions(rng, area_inflation=1.4)
    base = box(0.0, 0.0, W, D)
    side = rng.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        arm_w = rng.uniform(*cfg.T_CUT_W_RANGE) * W
        arm_d = rng.uniform(*cfg.T_CUT_D_RANGE) * D
        stem_width = W - 2.0 * arm_w
        if stem_width < cfg.T_MIN_STEM_WIDTH:
            return None
        if side == "bottom":
            c1 = box(0.0, 0.0, arm_w, arm_d)
            c2 = box(W - arm_w, 0.0, W, arm_d)
        else:
            c1 = box(0.0, D - arm_d, arm_w, D)
            c2 = box(W - arm_w, D - arm_d, W, D)
    else:
        arm_h = rng.uniform(*cfg.T_CUT_W_RANGE) * D
        arm_d = rng.uniform(*cfg.T_CUT_D_RANGE) * W
        stem_height = D - 2.0 * arm_h
        if stem_height < cfg.T_MIN_STEM_WIDTH:
            return None
        if side == "left":
            c1 = box(0.0, 0.0, arm_d, arm_h)
            c2 = box(0.0, D - arm_h, arm_d, D)
        else:
            c1 = box(W - arm_d, 0.0, W, arm_h)
            c2 = box(W - arm_d, D - arm_h, W, D)
    poly = _safe_difference(base, c1)
    if poly is None:
        return None
    poly = _safe_difference(poly, c2)
    if poly is None or not _validate(poly):
        return None
    return FootprintResult(
        shape_type="T_shape",
        polygon=poly,
        area=round(poly.area, 3),
        bbox_w=round(W, 3),
        bbox_d=round(D, 3),
    )


def _gen_U_shape(rng: random.Random) -> FootprintResult | None:
    W, D = _sample_base_dimensions(rng, area_inflation=1.5)
    base = box(0.0, 0.0, W, D)
    open_side = rng.choice(["top", "bottom", "left", "right"])
    if open_side in ("top", "bottom"):
        notch_w = rng.uniform(*cfg.U_NOTCH_W_RANGE) * W
        notch_d = rng.uniform(*cfg.U_NOTCH_D_RANGE) * D
        arm_width = (W - notch_w) / 2.0
        back_depth = D - notch_d
        if arm_width < cfg.U_MIN_ARM_WIDTH:
            return None
        if back_depth < cfg.U_MIN_ARM_WIDTH:
            return None
        nx_s = (W - notch_w) / 2.0
        nx_e = nx_s + notch_w
        if open_side == "top":
            notch = box(nx_s, D - notch_d, nx_e, D)
        else:
            notch = box(nx_s, 0.0, nx_e, notch_d)
    else:
        notch_h = rng.uniform(*cfg.U_NOTCH_W_RANGE) * D
        notch_d = rng.uniform(*cfg.U_NOTCH_D_RANGE) * W
        arm_height = (D - notch_h) / 2.0
        back_width = W - notch_d
        if arm_height < cfg.U_MIN_ARM_WIDTH:
            return None
        if back_width < cfg.U_MIN_ARM_WIDTH:
            return None
        ny_s = (D - notch_h) / 2.0
        ny_e = ny_s + notch_h
        if open_side == "left":
            notch = box(0.0, ny_s, notch_d, ny_e)
        else:
            notch = box(W - notch_d, ny_s, W, ny_e)
    poly = _safe_difference(base, notch)
    if poly is None or not _validate(poly):
        return None
    return FootprintResult(
        shape_type="U_shape",
        polygon=poly,
        area=round(poly.area, 3),
        bbox_w=round(W, 3),
        bbox_d=round(D, 3),
    )


def _gen_courtyard(rng: random.Random) -> FootprintResult | None:
    W, D = _sample_base_dimensions(rng, area_inflation=1.6)
    void_w = rng.uniform(*cfg.COURT_VOID_W_RANGE) * W
    void_d = rng.uniform(*cfg.COURT_VOID_D_RANGE) * D
    wall_x = (W - void_w) / 2.0
    wall_y = (D - void_d) / 2.0
    if wall_x < cfg.COURT_MIN_WALL_THICK or wall_y < cfg.COURT_MIN_WALL_THICK:
        return None
    max_ox = wall_x * 0.2
    max_oy = wall_y * 0.2
    ox = rng.uniform(-max_ox, max_ox)
    oy = rng.uniform(-max_oy, max_oy)
    vx0 = wall_x + ox
    vy0 = wall_y + oy
    exterior = list(box(0.0, 0.0, W, D).exterior.coords)
    void_box = box(vx0, vy0, vx0 + void_w, vy0 + void_d)
    interior = list(void_box.exterior.coords)
    poly = Polygon(shell=exterior, holes=[interior])
    if not poly.is_valid:
        poly = make_valid(poly)
        if not isinstance(poly, Polygon):
            return None
    if not _validate(poly):
        return None
    return FootprintResult(
        shape_type="courtyard",
        polygon=poly,
        area=round(poly.area, 3),
        bbox_w=round(W, 3),
        bbox_d=round(D, 3),
    )


SHAPE_GENERATORS: dict[str, Callable[[random.Random], FootprintResult | None]] = {
    "rectangle": _gen_rectangle,
    "quadrilateral": _gen_quadrilateral,
    "L_shape": _gen_L_shape,
    "T_shape": _gen_T_shape,
    "U_shape": _gen_U_shape,
    "courtyard": _gen_courtyard,
}


def generate_footprint(shape_type: str | None = None, seed: int | None = None) -> FootprintResult:
    """Sample a validated building footprint from the configured shape distribution."""
    rng = _make_rng(seed)
    if shape_type is None:
        types = list(cfg.SHAPE_WEIGHTS.keys())
        weights = [cfg.SHAPE_WEIGHTS[t] for t in types]
        shape_type = rng.choices(types, weights=weights, k=1)[0]
    if shape_type not in SHAPE_GENERATORS:
        raise ValueError(
            f"Unknown shape_type '{shape_type}'. Valid options: {list(SHAPE_GENERATORS.keys())}"
        )
    generator = SHAPE_GENERATORS[shape_type]
    for attempt in range(1, cfg.MAX_RETRIES + 1):
        result = generator(rng)
        if result is not None:
            logger.info(
                f"[generate_footprint] {shape_type:15s} | area = {result.area:7.1f} m²  W = {result.bbox_w:5.1f} m  D = {result.bbox_d:5.1f} m  (attempt {attempt})"
            )
            return result
    raise RuntimeError(
        f"[generate_footprint] Could not produce a valid '{shape_type}' footprint after {cfg.MAX_RETRIES} attempts.  Consider relaxing constraints in config_hyperparams.py."
    )
