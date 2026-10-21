"""Parameterized building geometric config.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

AREA_RANGE: tuple[float, float] = (100.0, 1000.0)
SHAPE_WEIGHTS: dict[str, float] = {
    "rectangle": 0.25,
    "quadrilateral": 0.15,
    "L_shape": 0.2,
    "T_shape": 0.15,
    "U_shape": 0.15,
    "courtyard": 0.1,
}
WIDTH_RANGE: tuple[float, float] = (8.0, 48.0)
DEPTH_RANGE: tuple[float, float] = (8.0, 48.0)
ASPECT_RATIO_MAX: float = 5.0
QUAD_PERTURB_FRACTION: float = 0.2
L_CUT_W_RANGE: tuple[float, float] = (0.25, 0.65)
L_CUT_D_RANGE: tuple[float, float] = (0.25, 0.65)
T_CUT_W_RANGE: tuple[float, float] = (0.15, 0.35)
T_CUT_D_RANGE: tuple[float, float] = (0.25, 0.5)
T_MIN_STEM_WIDTH: float = 3.0
U_NOTCH_W_RANGE: tuple[float, float] = (0.25, 0.55)
U_NOTCH_D_RANGE: tuple[float, float] = (0.35, 0.65)
U_MIN_ARM_WIDTH: float = 3.5
COURT_VOID_W_RANGE: tuple[float, float] = (0.3, 0.55)
COURT_VOID_D_RANGE: tuple[float, float] = (0.3, 0.55)
COURT_MIN_WALL_THICK: float = 3.5
MAX_RETRIES: int = 150
MIN_EDGE_LENGTH: float = 1.0
RANDOM_SEED: int | None = None
GRID_SPACING: float = 8.0
MIN_ROOM_AREA: float = 12.0
MIN_SEED_COUNT: int = 2
MAX_MERGE_ITERATIONS: int = 200
MIN_WALL_EDGE_LENGTH: float = 0.8
LABEL_ZONE_TYPES: bool = True
MAX_DIVIDE_RETRIES: int = 50
MIN_EXTERIOR_EDGE_LENGTH: float = 1.2
MIN_INTERIOR_EDGE_LENGTH: float = 0.8
MAX_ROOM_AREA: float = 200.0
DIVIDE_RETRY_SPACING_MIN_FACTOR: float = 0.8
DIVIDE_RETRY_SPACING_MAX_FACTOR: float = 1.2
ROTATION_PROBABILITY: float = 0.9
ROTATION_MODE: str = "full"
FLOOR_HEIGHT_RANGE: tuple = (2.7, 4.0)
LOW_RISE_AREA_THRESHOLD: float = 200.0
STOREY_COUNT_WEIGHTS_NORMAL: dict[str, float] = {
    "1": 0.08,
    "2": 0.12,
    "3": 0.25,
    "4": 0.15,
    "5": 0.12,
    "6": 0.18,
    "tall_mid": 0.1,
}
STOREY_COUNT_WEIGHTS_LOW_RISE: dict[str, float] = {"1": 0.3, "2": 0.5, "3": 0.2}
TALL_BUILDING_SIM_FLOORS: int = 4
FLOOR_BC_PROBS: dict = {"Ground": 0.88, "Adiabatic": 0.1, "Outdoors": 0.02}
CEILING_BC_PROBS: dict = {"Outdoors": 0.88, "Adiabatic": 0.12}
EXTERIOR_WALL_TOLERANCE: float = 0.02
WINDOW_STYLE_PROBS: dict[str, float] = {"normal": 1.0, "extreme_large": 0.0, "extreme_small": 0.0}
WWR_SOUTH_RANGE: tuple = (0.18, 0.6)
WWR_NORTH_RANGE: tuple = (0.1, 0.25)
WWR_EAST_RANGE: tuple = (0.12, 0.4)
WWR_WEST_RANGE: tuple = (0.12, 0.35)
SILL_HEIGHT_RANGE: tuple = (0.6, 1.2)
HEAD_HEIGHT_RANGE: tuple = (2.1, 2.6)
WWR_EXTREME_LARGE_RANGE: tuple = (0.7, 0.85)
SILL_EXTREME_LARGE_RANGE: tuple = (0.15, 0.3)
HEAD_EXTREME_LARGE_RANGE: tuple = (2.4, 3.5)
WWR_EXTREME_SMALL_RANGE: tuple = (0.05, 0.12)
SILL_EXTREME_SMALL_RANGE: tuple = (1.35, 1.6)
HEAD_EXTREME_SMALL_RANGE: tuple = (1.9, 2.2)
WINDOW_WALL_MARGIN: float = 0.2
MIN_WALL_WIDTH_FOR_WINDOW: float = 1.2
EXTERIOR_SHADE_PROB_BY_ORIENT: dict[str, float] = {"N": 0.6, "S": 0.9, "E": 0.75, "W": 0.75}
EXTERIOR_SHADE_DEPTH_SOUTH_RANGE: tuple[float, float] = (0.4, 1.2)
EXTERIOR_SHADE_DEPTH_NORTH_RANGE: tuple[float, float] = (0.2, 0.8)
EXTERIOR_SHADE_DEPTH_EAST_RANGE: tuple[float, float] = (0.4, 1.2)
EXTERIOR_SHADE_DEPTH_WEST_RANGE: tuple[float, float] = (0.4, 1.2)
EXTERIOR_SHADE_OVERHANG: float = 0.1
EXTERIOR_SHADE_GAP: float = 0.1
INTERIOR_SHADE_PROBABILITY: float = 0.5
INTERIOR_SHADE_COVERAGE_RANGE: tuple[float, float] = (0.3, 1.0)
INTERIOR_SHADE_OFFSET: float = 0.2
