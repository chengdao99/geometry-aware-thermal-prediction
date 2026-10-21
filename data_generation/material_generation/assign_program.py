"""Parameterized building assign program.

Inputs: Seeded geometry, physical parameters, and optional overrides.
Outputs: Building geometry, material records, or HBJSON artifacts."""

from __future__ import annotations
import random
from typing import Optional
import logging

logger = logging.getLogger(__name__)
from . import material_config as cfg
from .assign_materials import MaterializedBuildingModel


def _sample_ach(rng: random.Random, scenario_weights: dict, ach_ranges: dict) -> tuple[str, float]:
    scenarios = list(scenario_weights.keys())
    weights = list(scenario_weights.values())
    scenario = rng.choices(scenarios, weights=weights)[0]
    lo, hi = ach_ranges[scenario]
    ach = round(rng.uniform(lo, hi), 3)
    return (scenario, ach)


def _sample_peak(
    rng: random.Random, intensity_weights: dict, value_ranges: dict
) -> tuple[str, float]:
    intensities = list(intensity_weights.keys())
    weights = list(intensity_weights.values())
    intensity = rng.choices(intensities, weights=weights)[0]
    lo, hi = value_ranges[intensity]
    peak = round(rng.uniform(lo, hi), 3)
    return (intensity, peak)


def _sample_shape(rng: random.Random, shape_options: list[str], shape_weights: list[float]) -> str:
    return rng.choices(shape_options, weights=shape_weights)[0]


def _build_ach_schedule_24h(
    ach_day: float, ach_night: float, day_start: int, night_start: int
) -> list[float]:
    schedule = []
    for h in range(24):
        if day_start <= h < night_start:
            schedule.append(ach_day)
        else:
            schedule.append(ach_night)
    return schedule


def _build_feature_schedule_24h(peak: float, shape_name: str, sign: float = 1.0) -> list[float]:
    shape = cfg.SCHEDULE_SHAPES[shape_name]
    return [round(sign * peak * v, 4) for v in shape]


def assign_program(
    mat_model: MaterializedBuildingModel, seed: Optional[int] = None, overrides=None
) -> MaterializedBuildingModel:
    """Assign infiltration and operational schedules using the supplied seed and overrides."""
    rng = random.Random(seed)
    scenario_names = list(cfg.BUILDING_SCENARIO_WEIGHTS.keys())
    scenario_weights = list(cfg.BUILDING_SCENARIO_WEIGHTS.values())
    if overrides and overrides.scenario_type:
        scenario_type = overrides.scenario_type
    else:
        scenario_type = rng.choices(scenario_names, weights=scenario_weights)[0]
    is_occupied = scenario_type in ("occupied_passive", "occupied_hvac")
    has_hvac = scenario_type == "occupied_hvac"
    logger.info(f"[assign_program] Building scenario: {scenario_type}")
    if overrides and overrides.day_start_hour is not None:
        day_start = overrides.day_start_hour
    else:
        day_start = rng.randint(*cfg.ACH_DAY_START_RANGE)
    if overrides and overrides.night_start_hour is not None:
        night_start = overrides.night_start_hour
    else:
        night_start = rng.randint(*cfg.ACH_NIGHT_START_RANGE)
    if night_start - day_start < cfg.MIN_DAY_DURATION:
        logger.warning(f"[assign_program] Day boundary: Day time too short.")
    logger.info(f"[assign_program] Day boundary: {day_start:02d}:00 – {night_start:02d}:00")
    room_infiltration_ach_day = []
    room_infiltration_ach_night = []
    room_ach_day_scenario = []
    room_ach_night_scenario = []
    room_ach_schedule_24h = []
    room_equip_peak = []
    room_equip_intensity = []
    room_equip_shape = []
    room_equip_schedule = []
    room_occ_peak = []
    room_occ_intensity = []
    room_occ_shape = []
    room_occ_schedule = []
    room_hvac_mode = []
    room_hvac_peak = []
    room_hvac_intensity = []
    room_hvac_shape = []
    room_hvac_schedule = []
    for room in mat_model.geometry.rooms:
        is_core = room.zone_type == "core"
        if overrides and overrides.ach_day is not None:
            ach_day = overrides.ach_day
            day_scenario = "Manual"
            ach_night = overrides.ach_night if overrides.ach_night is not None else ach_day
            night_scenario = "Manual"
        elif is_core:
            day_scenario, ach_day = _sample_ach(
                rng, cfg.ACH_DAY_SCENARIO_WEIGHTS_CORE, cfg.ACH_DAY_RANGES_CORE
            )
            night_scenario, ach_night = _sample_ach(
                rng, cfg.ACH_NIGHT_SCENARIO_WEIGHTS_CORE, cfg.ACH_NIGHT_RANGES_CORE
            )
        else:
            day_scenario, ach_day = _sample_ach(
                rng, cfg.ACH_DAY_SCENARIO_WEIGHTS, cfg.ACH_DAY_RANGES
            )
            night_scenario, ach_night = _sample_ach(
                rng, cfg.ACH_NIGHT_SCENARIO_WEIGHTS, cfg.ACH_NIGHT_RANGES
            )
        room_infiltration_ach_day.append(ach_day)
        room_infiltration_ach_night.append(ach_night)
        room_ach_day_scenario.append(day_scenario)
        room_ach_night_scenario.append(night_scenario)
        room_ach_schedule_24h.append(
            _build_ach_schedule_24h(ach_day, ach_night, day_start, night_start)
        )
        if is_occupied:
            if overrides and overrides.equip_peak is not None:
                equip_peak = overrides.equip_peak
                equip_shape = overrides.equip_shape or "daytime"
                equip_intensity = "Manual"
            elif is_core:
                equip_intensity, equip_peak = _sample_peak(
                    rng, cfg.EQUIP_INTENSITY_WEIGHTS_CORE, cfg.EQUIP_WATTS_RANGES_CORE
                )
                equip_shape = _sample_shape(
                    rng, cfg.EQUIP_SCHEDULE_OPTIONS_CORE, cfg.EQUIP_SCHEDULE_WEIGHTS_CORE
                )
            else:
                equip_intensity, equip_peak = _sample_peak(
                    rng, cfg.EQUIP_INTENSITY_WEIGHTS, cfg.EQUIP_WATTS_RANGES
                )
                equip_shape = _sample_shape(
                    rng, cfg.EQUIP_SCHEDULE_OPTIONS, cfg.EQUIP_SCHEDULE_WEIGHTS
                )
        else:
            equip_intensity, equip_peak, equip_shape = ("none", 0.0, "zero")
        room_equip_peak.append(equip_peak)
        room_equip_intensity.append(equip_intensity)
        room_equip_shape.append(equip_shape)
        room_equip_schedule.append(_build_feature_schedule_24h(equip_peak, equip_shape, sign=1.0))
        if is_occupied:
            if overrides and overrides.occ_peak is not None:
                occ_peak = overrides.occ_peak
                occ_shape = overrides.occ_shape or "daytime"
                occ_intensity = "Manual"
            elif is_core:
                occ_intensity, occ_peak = _sample_peak(
                    rng, cfg.OCC_INTENSITY_WEIGHTS_CORE, cfg.OCC_DENSITY_RANGES_CORE
                )
                occ_shape = _sample_shape(
                    rng, cfg.OCC_SCHEDULE_OPTIONS_CORE, cfg.OCC_SCHEDULE_WEIGHTS_CORE
                )
            else:
                occ_intensity, occ_peak = _sample_peak(
                    rng, cfg.OCC_INTENSITY_WEIGHTS, cfg.OCC_DENSITY_RANGES
                )
                occ_shape = _sample_shape(rng, cfg.OCC_SCHEDULE_OPTIONS, cfg.OCC_SCHEDULE_WEIGHTS)
        else:
            occ_intensity, occ_peak, occ_shape = ("none", 0.0, "zero")
        room_occ_peak.append(occ_peak)
        room_occ_intensity.append(occ_intensity)
        room_occ_shape.append(occ_shape)
        room_occ_schedule.append(_build_feature_schedule_24h(occ_peak, occ_shape, sign=1.0))
        if has_hvac:
            if overrides and overrides.hvac_peak is not None:
                hvac_peak = overrides.hvac_peak
                hvac_shape = overrides.hvac_shape or "daytime"
                hvac_intensity = "Manual"
                if hvac_peak > 0.0:
                    hvac_sign = +1.0
                    hvac_mode = "heating"
                else:
                    hvac_sign = -1.0
                    hvac_mode = "cooling"
            else:
                hvac_mode = rng.choices(
                    list(cfg.HVAC_MODE_WEIGHTS.keys()), weights=list(cfg.HVAC_MODE_WEIGHTS.values())
                )[0]
                if hvac_mode == "heating":
                    if is_core:
                        hvac_intensity, hvac_peak = _sample_peak(
                            rng,
                            cfg.HVAC_HEATING_INTENSITY_WEIGHTS_CORE,
                            cfg.HVAC_HEATING_WATTS_RANGES_CORE,
                        )
                    else:
                        hvac_intensity, hvac_peak = _sample_peak(
                            rng, cfg.HVAC_HEATING_INTENSITY_WEIGHTS, cfg.HVAC_HEATING_WATTS_RANGES
                        )
                    hvac_shape = _sample_shape(
                        rng, cfg.HVAC_HEATING_SCHEDULE_OPTIONS, cfg.HVAC_HEATING_SCHEDULE_WEIGHTS
                    )
                    hvac_sign = +1.0
                else:
                    if is_core:
                        hvac_intensity, hvac_peak = _sample_peak(
                            rng,
                            cfg.HVAC_COOLING_INTENSITY_WEIGHTS_CORE,
                            cfg.HVAC_COOLING_WATTS_RANGES_CORE,
                        )
                    else:
                        hvac_intensity, hvac_peak = _sample_peak(
                            rng, cfg.HVAC_COOLING_INTENSITY_WEIGHTS, cfg.HVAC_COOLING_WATTS_RANGES
                        )
                    hvac_shape = _sample_shape(
                        rng, cfg.HVAC_COOLING_SCHEDULE_OPTIONS, cfg.HVAC_COOLING_SCHEDULE_WEIGHTS
                    )
                    hvac_sign = -1.0
        else:
            hvac_mode, hvac_intensity, hvac_peak, hvac_shape, hvac_sign = (
                "none",
                "none",
                0.0,
                "zero",
                1.0,
            )
        room_hvac_mode.append(hvac_mode)
        room_hvac_peak.append(hvac_peak)
        room_hvac_intensity.append(hvac_intensity)
        room_hvac_shape.append(hvac_shape)
        room_hvac_schedule.append(
            _build_feature_schedule_24h(hvac_peak, hvac_shape, sign=hvac_sign)
        )
    bld_meta = mat_model.material_spec.metadata.get("building", {})
    n_floors = bld_meta.get("n_floors", 1)
    ctx_meta = mat_model.material_spec.metadata.get("context", {})
    svf_wall_f0 = ctx_meta.get("svf_wall", 1.0)
    svf_roof = ctx_meta.get("svf_roof", 1.0)
    lo_svf, hi_svf = cfg.SVF_HEIGHT_FACTOR_RANGE
    svf_per_floor = []
    current_svf = svf_wall_f0
    for i in range(n_floors):
        if i == 0:
            wall_svf = min(round(current_svf, 3), 1.0)
            factor_used = 1.0
        else:
            if overrides and overrides.svf_height_factor is not None:
                factor_used = float(overrides.svf_height_factor)
            else:
                factor_used = rng.uniform(lo_svf, hi_svf)
            factor_used = max(0.0, factor_used)
            current_svf = min(current_svf * factor_used, 1.0)
            wall_svf = round(current_svf, 3)
        svf_per_floor.append(
            {"floor_index": i, "svf_wall": wall_svf, "svf_roof": svf_roof, "factor": factor_used}
        )
    program_meta = {
        "scenario_type": scenario_type,
        "room_infiltration_ach_day": room_infiltration_ach_day,
        "room_infiltration_ach_night": room_infiltration_ach_night,
        "room_ach_day_scenario": room_ach_day_scenario,
        "room_ach_night_scenario": room_ach_night_scenario,
        "room_ach_schedule_24h": room_ach_schedule_24h,
        "day_start_hour": day_start,
        "night_start_hour": night_start,
        "room_equip_peak": room_equip_peak,
        "room_equip_intensity": room_equip_intensity,
        "room_equip_shape": room_equip_shape,
        "room_equip_schedule": room_equip_schedule,
        "room_occ_peak": room_occ_peak,
        "room_occ_intensity": room_occ_intensity,
        "room_occ_shape": room_occ_shape,
        "room_occ_schedule": room_occ_schedule,
        "room_hvac_mode": room_hvac_mode,
        "room_hvac_peak": room_hvac_peak,
        "room_hvac_intensity": room_hvac_intensity,
        "room_hvac_shape": room_hvac_shape,
        "room_hvac_schedule": room_hvac_schedule,
        "n_floors": n_floors,
        "svf_per_floor": svf_per_floor,
    }
    mat_model.material_spec.metadata["program"] = program_meta
    n_rooms = len(mat_model.geometry.rooms)
    logger.info(f"[assign_program] Rooms: {n_rooms}")
    logger.info(
        f"[assign_program] ACH day   (first 5): {[round(v, 2) for v in room_infiltration_ach_day[:5]]}"
    )
    logger.info(
        f"[assign_program] ACH night (first 5): {[round(v, 2) for v in room_infiltration_ach_night[:5]]}"
    )
    if is_occupied:
        logger.info(
            f"[assign_program] Equip peak W/m² (first 5): {[round(v, 1) for v in room_equip_peak[:5]]}"
        )
        logger.info(
            f"[assign_program] Occ density people/m2 (first 5): {[round(v, 3) for v in room_occ_peak[:5]]}"
        )
    if has_hvac:
        hvac_summary = [
            f"{m}({('+' if m == 'heating' else '-')}{round(p, 1)})"
            for m, p in zip(room_hvac_mode[:5], room_hvac_peak[:5])
        ]
        logger.info(f"[assign_program] HVAC (first 5): {hvac_summary}")
    svf_info = []
    for i, s in enumerate(svf_per_floor):
        factor_info = f"(x{s['factor']:.3f})" if i > 0 else ""
        svf_info.append(f"F{i}={s['svf_wall']:.3f}{factor_info}")
    logger.info(f"[assign_program] SVF gradient: " + "  ".join(svf_info))
    return mat_model
