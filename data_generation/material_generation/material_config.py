"""Parameterized building material config.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

from __future__ import annotations

STRUCTURAL_SYSTEM_AREA_THRESHOLD: float = 1000.0
STRUCTURAL_SYSTEM_WEIGHTS_LARGE: dict[str, float] = {"CLT": 0.2, "Concrete": 0.55, "Brick": 0.25}
STRUCTURAL_SYSTEM_WEIGHTS_SMALL: dict[str, float] = {
    "LightWood": 0.5,
    "CLT": 0.15,
    "Brick": 0.2,
    "Concrete": 0.15,
}
CONCRETE_SUBTYPE_WEIGHTS: dict[str, float] = {"NormalWeight": 0.7, "Lightweight": 0.3}
INSULATION_WEIGHTS: dict[str, float] = {"EPS": 0.4, "MinWool": 0.4, "PIR": 0.2}
GLAZING_SYSTEM_WEIGHTS: dict[str, float] = {
    "single_clear": 0.04,
    "double_clear": 0.08,
    "double_low_e_high_gain": 0.23,
    "double_low_e_solar_ctrl": 0.1,
    "triple_low_e_high_gain": 0.4,
    "triple_low_e_solar_ctrl": 0.15,
}
WALL_STRUCTURAL_T_DISCRETE: dict[str, dict] = {
    "CLT": {
        "values": [0.12, 0.16, 0.2, 0.24, 0.32, 0.4],
        "weights": [0.135, 0.36, 0.315, 0.11, 0.04, 0.04],
    },
    "Concrete": {
        "values": [0.15, 0.18, 0.2, 0.25, 0.3, 0.45, 0.6],
        "weights": [0.09, 0.18, 0.315, 0.225, 0.11, 0.04, 0.04],
    },
    "Brick": {
        "values": [0.12, 0.24, 0.365, 0.49, 0.615, 0.74],
        "weights": [0.045, 0.315, 0.45, 0.11, 0.04, 0.04],
    },
    "LightWood": {
        "values": [0.09, 0.12, 0.15, 0.18, 0.2, 0.24, 0.3],
        "weights": [0.09, 0.27, 0.315, 0.2, 0.045, 0.04, 0.04],
    },
}
FLOOR_STRUCTURAL_T_DISCRETE: dict[str, dict] = {
    "CLT": {"values": [0.16, 0.18, 0.2, 0.24], "weights": [0.2, 0.35, 0.3, 0.15]},
    "Concrete": {"values": [0.15, 0.18, 0.2, 0.22, 0.25], "weights": [0.1, 0.25, 0.35, 0.2, 0.1]},
    "Brick": {"values": [0.15, 0.18, 0.2, 0.22, 0.25], "weights": [0.1, 0.25, 0.35, 0.2, 0.1]},
    "LightWood": {"values": [0.15, 0.18, 0.2, 0.24, 0.28], "weights": [0.1, 0.25, 0.35, 0.2, 0.1]},
}
ROOF_STRUCTURAL_T_DISCRETE: dict[str, dict] = {
    "CLT": {"values": [0.16, 0.18, 0.2, 0.24, 0.3], "weights": [0.2, 0.32, 0.28, 0.15, 0.05]},
    "Concrete": {"values": [0.15, 0.18, 0.2, 0.22, 0.25], "weights": [0.1, 0.25, 0.35, 0.2, 0.1]},
    "Brick": {"values": [0.15, 0.18, 0.2, 0.22, 0.25], "weights": [0.1, 0.25, 0.35, 0.2, 0.1]},
    "LightWood": {"values": [0.12, 0.15, 0.18, 0.2, 0.24], "weights": [0.1, 0.3, 0.35, 0.2, 0.05]},
}
WALL_INSULATION_T_DISCRETE: dict[str, dict] = {
    "EPS": {
        "values": [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.35, 0.45],
        "weights": [0.05, 0.0425, 0.085, 0.255, 0.2125, 0.2, 0.085, 0.04, 0.03],
    },
    "MinWool": {
        "values": [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.35, 0.45],
        "weights": [0.05, 0.0425, 0.068, 0.2325, 0.255, 0.187, 0.085, 0.04, 0.04],
    },
    "PIR": {
        "values": [0.04, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.25, 0.3],
        "weights": [0.05, 0.068, 0.187, 0.2975, 0.18, 0.105, 0.0425, 0.04, 0.03],
    },
}
ROOF_INSULATION_T_DISCRETE: dict[str, dict] = {
    "EPS": {
        "values": [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6],
        "weights": [0.05, 0.068, 0.187, 0.2975, 0.2125, 0.105, 0.04, 0.04],
    },
    "MinWool": {
        "values": [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6],
        "weights": [0.05, 0.068, 0.173, 0.2975, 0.238, 0.0935, 0.04, 0.04],
    },
    "PIR": {
        "values": [0.06, 0.12, 0.14, 0.16, 0.18, 0.2, 0.24, 0.35, 0.4],
        "weights": [0.05, 0.068, 0.173, 0.255, 0.2125, 0.119, 0.0425, 0.04, 0.04],
    },
}
FLOOR_INSULATION_T_DISCRETE: dict[str, dict] = {
    "EPS": {
        "values": [0.04, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4],
        "weights": [0.05, 0.1275, 0.255, 0.2125, 0.17, 0.105, 0.04, 0.04],
    },
    "MinWool": {
        "values": [0.04, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4],
        "weights": [0.05, 0.102, 0.238, 0.238, 0.187, 0.105, 0.04, 0.04],
    },
    "PIR": {
        "values": [0.03, 0.06, 0.08, 0.1, 0.12, 0.14, 0.2, 0.25],
        "weights": [0.05, 0.085, 0.238, 0.2975, 0.2, 0.0595, 0.04, 0.03],
    },
}
FLOOR_INSULATION_PROB_GROUND: float = 0.92
FLOOR_INSULATION_PROB_OUTDOORS: float = 0.9
INTER_SLAB_INSULATION_PROB: float = 0.3
INTER_SLAB_INSULATION_T_DISCRETE: dict[str, dict] = {
    "EPS": {"values": [0.02, 0.03, 0.04, 0.05, 0.06], "weights": [0.15, 0.3, 0.3, 0.15, 0.1]},
    "MinWool": {"values": [0.025, 0.03, 0.04, 0.05, 0.06], "weights": [0.1, 0.3, 0.3, 0.2, 0.1]},
    "PIR": {"values": [0.02, 0.025, 0.03, 0.04, 0.05], "weights": [0.15, 0.3, 0.3, 0.15, 0.1]},
}
INTERIOR_WALL_T_DISCRETE: dict[str, dict] = {
    "CLT_Interior": {"values": [0.08, 0.1, 0.12], "weights": [0.3, 0.4, 0.3]},
    "Concrete_Block_Interior": {"values": [0.1, 0.15, 0.2], "weights": [0.3, 0.5, 0.2]},
    "Brick_Interior": {"values": [0.1, 0.15, 0.2], "weights": [0.3, 0.5, 0.2]},
    "Gypsum_Board": {"values": [0.08, 0.1, 0.12], "weights": [0.4, 0.4, 0.2]},
}
U_VALUE_WALL_RANGE: tuple[float, float] = (0.08, 0.45)
U_VALUE_ROOF_RANGE: tuple[float, float] = (0.06, 0.25)
U_VALUE_FLOOR_RANGE: tuple[float, float] = (0.08, 0.4)
U_VALUE_WINDOW_RANGE: tuple[float, float] = (0.55, 3.0)
R_SI_WALL: float = 0.13
R_SE_WALL: float = 0.04
R_SI_ROOF: float = 0.1
R_SE_ROOF: float = 0.04
R_SI_FLOOR: float = 0.17
R_SE_FLOOR: float = 0.04
CONTEXT_SCENARIO_WEIGHTS: dict[str, float] = {
    "open": 0.25,
    "suburban": 0.45,
    "urban": 0.2,
    "sheltered": 0.1,
}
SVF_WALL_RANGES: dict[str, tuple[float, float]] = {
    "open": (0.8, 0.95),
    "suburban": (0.55, 0.8),
    "urban": (0.3, 0.55),
    "sheltered": (0.1, 0.3),
}
SVF_ROOF_RANGES: dict[str, tuple[float, float]] = {
    "open": (0.95, 1.0),
    "suburban": (0.85, 0.95),
    "urban": (0.7, 0.85),
    "sheltered": (0.5, 0.7),
}
SVF_HEIGHT_FACTOR_RANGE: tuple[float, float] = (1.02, 1.1)
ACH_DAY_START_RANGE: tuple[int, int] = (6, 10)
ACH_NIGHT_START_RANGE: tuple[int, int] = (19, 23)
MIN_DAY_DURATION: int = 6
ACH_DAY_SCENARIO_WEIGHTS: dict[str, float] = {
    "infiltration": 0.2,
    "mechanical": 0.4,
    "natural_partial": 0.3,
    "natural_full": 0.1,
}
ACH_DAY_RANGES: dict[str, tuple[float, float]] = {
    "infiltration": (0.03, 0.3),
    "mechanical": (0.5, 1.5),
    "natural_partial": (1.2, 2.5),
    "natural_full": (3.0, 7.0),
}
ACH_NIGHT_SCENARIO_WEIGHTS: dict[str, float] = {
    "infiltration": 0.45,
    "mechanical": 0.35,
    "natural_partial": 0.15,
    "natural_full": 0.05,
}
ACH_NIGHT_RANGES: dict[str, tuple[float, float]] = {
    "infiltration": (0.03, 0.3),
    "mechanical": (0.5, 1.5),
    "natural_partial": (1.2, 2.5),
    "natural_full": (3.0, 7.0),
}
ACH_DAY_SCENARIO_WEIGHTS_CORE: dict[str, float] = {
    "infiltration": 0.55,
    "mechanical": 0.35,
    "natural_partial": 0.1,
    "natural_full": 0.0,
}
ACH_DAY_RANGES_CORE: dict[str, tuple[float, float]] = {
    "infiltration": (0.03, 0.3),
    "mechanical": (0.5, 1.0),
    "natural_partial": (0.3, 0.8),
    "natural_full": (1.0, 2.0),
}
ACH_NIGHT_SCENARIO_WEIGHTS_CORE: dict[str, float] = {
    "infiltration": 0.6,
    "mechanical": 0.35,
    "natural_partial": 0.05,
    "natural_full": 0.0,
}
ACH_NIGHT_RANGES_CORE: dict[str, tuple[float, float]] = {
    "infiltration": (0.03, 0.4),
    "mechanical": (0.3, 0.8),
    "natural_partial": (0.3, 0.6),
    "natural_full": (0.5, 1.2),
}
INTERIOR_WALL_MODE: str = "real"
AIR_MIX: float = 0.001
BUILDING_SCENARIO_WEIGHTS: dict[str, float] = {
    "passive": 1.0,
    "occupied_passive": 0.0,
    "occupied_hvac": 0.0,
}
SCHEDULE_SHAPES: dict[str, list[float]] = {
    "zero": [0.0] * 24,
    "daytime": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3,
        0.8,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.8,
        0.5,
        0.3,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "nighttime": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.8,
        0.4,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.6,
        0.9,
        1.0,
        1.0,
        1.0,
    ],
    "always_on": [1.0] * 24,
    "morning_evening": [
        0.2,
        0.1,
        0.1,
        0.1,
        0.1,
        0.2,
        0.5,
        0.9,
        1.0,
        0.4,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.3,
        0.5,
        0.8,
        1.0,
        1.0,
        0.8,
        0.5,
        0.3,
        0.2,
    ],
    "wfh": [
        0.3,
        0.2,
        0.1,
        0.1,
        0.1,
        0.2,
        0.5,
        0.8,
        1.0,
        1.0,
        0.9,
        0.8,
        0.7,
        0.8,
        0.9,
        0.9,
        0.8,
        0.9,
        1.0,
        0.9,
        0.7,
        0.5,
        0.4,
        0.3,
    ],
    "heating_allday": [1.0] * 24,
    "heating_setback": [
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.5,
        0.7,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.9,
        0.7,
        0.5,
        0.4,
        0.4,
    ],
    "heating_daytime": [
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.4,
        0.7,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.9,
        0.7,
        0.5,
        0.4,
        0.3,
        0.3,
        0.3,
        0.3,
    ],
    "cooling_afternoon": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.3,
        0.7,
        0.9,
        1.0,
        1.0,
        1.0,
        0.9,
        0.7,
        0.4,
        0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "cooling_allday": [
        0.5,
        0.4,
        0.4,
        0.4,
        0.4,
        0.5,
        0.6,
        0.8,
        0.9,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.6,
        0.5,
        0.5,
        0.5,
    ],
}
EQUIP_FRACTION_RADIANT: float = 0.3
EQUIP_FRACTION_LATENT: float = 0.0
EQUIP_INTENSITY_WEIGHTS: dict[str, float] = {"low": 0.5, "medium": 0.35, "high": 0.15}
EQUIP_WATTS_RANGES: dict[str, tuple[float, float]] = {
    "low": (2.0, 8.0),
    "medium": (8.0, 18.0),
    "high": (18.0, 40.0),
}
EQUIP_SCHEDULE_OPTIONS: list[str] = ["daytime", "always_on", "morning_evening", "wfh"]
EQUIP_SCHEDULE_WEIGHTS: list[float] = [0.4, 0.25, 0.2, 0.15]
EQUIP_INTENSITY_WEIGHTS_CORE: dict[str, float] = {"low": 0.2, "medium": 0.4, "high": 0.4}
EQUIP_WATTS_RANGES_CORE: dict[str, tuple[float, float]] = {
    "low": (3.0, 10.0),
    "medium": (10.0, 25.0),
    "high": (25.0, 60.0),
}
EQUIP_SCHEDULE_OPTIONS_CORE: list[str] = ["always_on", "daytime", "morning_evening"]
EQUIP_SCHEDULE_WEIGHTS_CORE: list[float] = [0.55, 0.3, 0.15]
OCC_ACTIVITY_LEVEL: float = 120.0
OCC_FRACTION_RADIANT: float = 0.3
OCC_INTENSITY_WEIGHTS: dict[str, float] = {"low": 0.35, "medium": 0.5, "high": 0.15}
OCC_DENSITY_RANGES: dict[str, tuple[float, float]] = {
    "low": (0.01, 0.04),
    "medium": (0.04, 0.08),
    "high": (0.08, 0.15),
}
OCC_SCHEDULE_OPTIONS: list[str] = ["nighttime", "morning_evening", "wfh", "daytime"]
OCC_SCHEDULE_WEIGHTS: list[float] = [0.35, 0.35, 0.2, 0.1]
OCC_INTENSITY_WEIGHTS_CORE: dict[str, float] = {"low": 0.65, "medium": 0.3, "high": 0.05}
OCC_DENSITY_RANGES_CORE: dict[str, tuple[float, float]] = {
    "low": (0.005, 0.02),
    "medium": (0.02, 0.06),
    "high": (0.06, 0.1),
}
OCC_SCHEDULE_OPTIONS_CORE: list[str] = ["daytime", "morning_evening", "zero"]
OCC_SCHEDULE_WEIGHTS_CORE: list[float] = [0.6, 0.3, 0.1]
HVAC_FRACTION_RADIANT_HEATING: float = 0.4
HVAC_FRACTION_RADIANT_COOLING: float = 0.0
HVAC_FRACTION_LATENT: float = 0.0
HVAC_MODE_WEIGHTS: dict[str, float] = {"heating": 0.65, "cooling": 0.35}
HVAC_HEATING_INTENSITY_WEIGHTS: dict[str, float] = {"low": 0.3, "medium": 0.5, "high": 0.2}
HVAC_HEATING_WATTS_RANGES: dict[str, tuple[float, float]] = {
    "low": (5.0, 15.0),
    "medium": (15.0, 35.0),
    "high": (35.0, 65.0),
}
HVAC_HEATING_SCHEDULE_OPTIONS: list[str] = ["heating_allday", "heating_setback", "heating_daytime"]
HVAC_HEATING_SCHEDULE_WEIGHTS: list[float] = [0.4, 0.4, 0.2]
HVAC_COOLING_INTENSITY_WEIGHTS: dict[str, float] = {"low": 0.45, "medium": 0.4, "high": 0.15}
HVAC_COOLING_WATTS_RANGES: dict[str, tuple[float, float]] = {
    "low": (5.0, 15.0),
    "medium": (15.0, 30.0),
    "high": (30.0, 50.0),
}
HVAC_COOLING_SCHEDULE_OPTIONS: list[str] = ["cooling_afternoon", "cooling_allday"]
HVAC_COOLING_SCHEDULE_WEIGHTS: list[float] = [0.65, 0.35]
HVAC_HEATING_INTENSITY_WEIGHTS_CORE: dict[str, float] = {"low": 0.55, "medium": 0.4, "high": 0.05}
HVAC_HEATING_WATTS_RANGES_CORE: dict[str, tuple[float, float]] = {
    "low": (3.0, 10.0),
    "medium": (10.0, 25.0),
    "high": (25.0, 45.0),
}
HVAC_COOLING_INTENSITY_WEIGHTS_CORE: dict[str, float] = {"low": 0.5, "medium": 0.4, "high": 0.1}
HVAC_COOLING_WATTS_RANGES_CORE: dict[str, tuple[float, float]] = {
    "low": (3.0, 12.0),
    "medium": (12.0, 25.0),
    "high": (25.0, 40.0),
}
OUT_SHADE_MATERIAL_WEIGHTS: dict[str, float] = {
    "light_metal": 1.0,
    "dark_metal": 0.0,
    "concrete": 0.0,
}
