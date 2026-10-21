"""Voxel semantic codes and physical constants. Inputs: None. Outputs: Shared numerical settings."""

GRID_STEPS = {
    "Wall": 0.1,
    "Aperture": 0.05,
    "RoofCeiling": 0.1,
    "Floor": 0.1,
    "InterFloorSlab": 0.05,
    "InteriorWall": 0.1,
    "OutShade": 0.05,
    "InShade": 0.05,
}
VOXEL_PARENT_SIZE = 0.2
VOXEL_SUB_SIZE = 0.05
VOXEL_CHUNK_MEM_LIMIT_MB = 512
VOXEL_MERGE_EVERY = None
VOXEL_MIN_FILL_RATE = 0.025
TAG_MIN_SUPPORT_RATE = {1: 0.15, 2: 0.05, 3: 0.15, 4: 0.15, 5: 0.15, 6: 0.15, 7: 0.05, 8: 0.05}
MAX_CANDIDATES: int = 1000000
SEMANTIC_TAGS = {
    "Wall": 1,
    "Aperture": 2,
    "RoofCeiling": 3,
    "Floor": 4,
    "InterFloorSlab": 5,
    "InteriorWall": 6,
    "OutShade": 7,
    "InShade": 8,
}
VOXEL_TAG_WEIGHTS = {2: 4, 8: 4, 7: 3, 1: 1, 3: 1, 4: 1, 5: 1, 6: 1}
BC_MAP = {"Outdoors": 0, "Ground": 1, "Surface": 2, "Adiabatic": 3}
AIR_CONDUCTIVITY = 0.026
DEFAULT_OPAQUE_CONDUCTIVITY = 0.5
AIR_DENSITY = 1.28
GLASS_DENSITY = 2500.0
DEFAULT_OPAQUE_DENSITY = 1000.0
AIR_SPECIFIC_HEAT = 1000.0
GLASS_SPECIFIC_HEAT = 840.0
DEFAULT_OPAQUE_SPECIFIC_HEAT = 800.0
MIN_THICKNESS = 0.01
DEFAULT_OPAQUE_SOLAR_ABS = 0.7
DEFAULT_OPAQUE_EMISSIVITY = 0.9
DEFAULT_OPAQUE_TRANSMITTANCE = 0.0
DEFAULT_GLAZING_SOLAR_ABS = 0.1
DEFAULT_GLAZING_TRANSMITTANCE = 0.85
DEFAULT_GLAZING_EMISSIVITY = 0.84
DEFAULT_SHADE_REFLECTANCE = 0.35
DEFAULT_SHADE_EMISSIVITY = 0.9
INSHADE_THICKNESS = 0.008
INSHADE_CONDUCTIVITY = 0.05
INSHADE_DENSITY = 1400.0
INSHADE_SPECIFIC_HEAT = 1300.0
INSHADE_U_VALUE = 5.0
INSHADE_SOLAR_REFLECTANCE = 0.5
INSHADE_SOLAR_TRANSMITTANCE = 0.3
INSHADE_EMISSIVITY = 0.9
R_SI = {
    "Wall": 0.13,
    "Aperture": 0.13,
    "RoofCeiling": 0.1,
    "Floor": 0.17,
    "InteriorWall": 0.13,
    "InterFloorSlab": 0.1,
    "OutShade": 0.1,
    "InShade": 0.13,
}
R_SE_OUTDOORS = 0.04
R_SE_GROUND = 0.0


def _validate():
    import math

    ratio = VOXEL_PARENT_SIZE / VOXEL_SUB_SIZE
    if abs(ratio - round(ratio)) > 1e-06:
        raise ValueError(
            f"[a01_config] VOXEL_PARENT_SIZE ({VOXEL_PARENT_SIZE}) must be an exact integer multiple of VOXEL_SUB_SIZE ({VOXEL_SUB_SIZE}). Current ratio = {ratio:.6f}"
        )
    if VOXEL_CHUNK_MEM_LIMIT_MB < 64:
        raise ValueError(
            f"[a01_config] VOXEL_CHUNK_MEM_LIMIT_MB ({VOXEL_CHUNK_MEM_LIMIT_MB}) is dangerously low (minimum recommended: 64 MB)."
        )


_validate()
import math

_CALIBRATION_CACHE = {}


def get_calibrated_params(d_actual, a_step):
    """Compute get calibrated params for the current building or model state. Inputs: d_actual, a_step."""
    cache_key = (round(d_actual, 5), round(a_step, 5))
    if cache_key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[cache_key]
    d_calc = max(d_actual, a_step)
    if d_calc <= 1.137 * a_step:
        inner_term = 0.5 - 3.0 * d_calc / (math.pi * a_step)
        inner_term = max(-1.0, min(1.0, inner_term))
        angle = 1.0 / 3.0 * math.acos(inner_term) + 4.0 * math.pi / 3.0
        r = a_step * (0.5 + math.cos(angle))
    else:
        r = 0.5 * d_calc + 0.139 * a_step
    _CALIBRATION_CACHE[cache_key] = r
    return r


from pathlib import Path

DEFAULT_EP_PATH = None
EP_OUTPUT_VARIABLES = ["Zone Mean Air Temperature", "Zone Operative Temperature"]
GEOM_DIM = 7
VOXEL_COL_D = 3
VOXEL_COL_LAMBDA = 4
VOXEL_COL_U = 5
VOXEL_COL_CPRHO = 6
VOXEL_COL_SOLAR = 7
VOXEL_COL_TRANS = 8
VOXEL_COL_EMISS = 9
VOXEL_COL_BC = 10
VOXEL_COL_SVF = 11
VOXEL_COL_TAG = 12
VOXEL_COL_FACE_ID = 13
VOXEL_COL_FILL = 14
