"""Translate HBJSON to annual EnergyPlus IDF. Inputs: HBJSON, EPW, and working directory. Outputs: IDF with physical and operational adaptations."""

import json
import logging
from pathlib import Path
from typing import Optional
from .fix_adjacency import fix_adjacency_and_return

logger = logging.getLogger(__name__)


def _find_meta_json(hbjson_path: Path) -> Optional[Path]:
    stem = hbjson_path.stem.removesuffix("_fixed")
    candidates = [
        hbjson_path.parent / f"{stem}_meta.json",
        hbjson_path.parent / "metadata" / f"{stem}_meta.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _idf_any_number_schedule_type() -> str:
    return "\nScheduleTypeLimits,\n    Any Number;              !- Name\n"


def _idf_hvac_schedule_compact(room_id: str, values_24h: list) -> str:
    sched_name = f"{room_id}_HVAC_Sched"
    lines = [
        f"\nSchedule:Compact,",
        f"    {sched_name},    !- Name",
        f"    Any Number,      !- Schedule Type Limits Name",
        f"    Through: 12/31,  !- Field 1",
        f"    For: AllDays,    !- Field 2",
    ]
    for h, val in enumerate(values_24h):
        hour_label = f"{h + 1:02d}:00"
        terminator = ";" if h == 23 else ","
        lines.append(
            f"    Until: {hour_label}, {round(float(val), 4)}{terminator}   !- Hour {h:02d}"
        )
    return "\n".join(lines) + "\n"


def _idf_other_equipment(room_id: str, mode: str) -> str:
    sched_name = f"{room_id}_HVAC_Sched"
    obj_name = f"{room_id}_HVAC"
    frac_radiant = 0.4 if mode == "heating" else 0.0
    return f"\nOtherEquipment,\n    {obj_name},              !- Name\n    None,                    !- Fuel Type\n    {room_id},               !- Zone or ZoneList Name\n    {sched_name},            !- Schedule Name\n    Watts/Area,              !- Design Level Calculation Method\n    ,                        !- Design Level {{W}}\n    1.0,                     !- Watts per Zone Floor Area {{W/m2}}\n    ,                        !- Watts per Person {{W/person}}\n    {frac_radiant},          !- Fraction Radiant\n    0.0,                     !- Fraction Latent\n    0.0;                     !- Fraction Lost\n"


def _strip_indoor_shades(hbjson_path: Path) -> Path:
    with open(hbjson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    count = 0
    for room in data.get("rooms", []):
        for face in room.get("faces", []):
            for aperture in face.get("apertures", []):
                if "indoor_shades" in aperture:
                    del aperture["indoor_shades"]
                    count += 1
    if count == 0:
        return hbjson_path
    tmp_path = hbjson_path.with_name(hbjson_path.stem.removesuffix("_fixed") + "_idf_tmp.hbjson")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info(
        f"  [IDF-prep] Stripped indoor_shades from {count} aperture(s) → temp file: {tmp_path.name}"
    )
    return tmp_path


def _patch_view_factor_to_ground(idf_path: Path, hbjson_path: Path) -> int:
    import re

    with open(hbjson_path, "r", encoding="utf-8") as f:
        hbjson = json.load(f)
    hbjson_vfg: dict[str, float] = {}
    for room in hbjson.get("rooms", []):
        for face in room.get("faces", []):
            bc = face.get("boundary_condition", {})
            if not (isinstance(bc, dict) and bc.get("type") == "Outdoors"):
                continue
            raw_svf = bc.get("view_factor")
            if not isinstance(raw_svf, (int, float)):
                continue
            svf = max(0.0, min(1.0, float(raw_svf)))
            vfg = round(1.0 - svf, 4)
            hbjson_vfg[face["identifier"].upper()] = vfg
    if not hbjson_vfg:
        logger.warning("[VFG] No outdoor faces with SVF found in HBJSON. Skipping patch.")
        return 0
    svf_vals = [round(1.0 - v, 4) for v in hbjson_vfg.values()]
    svf_min, svf_max = (min(svf_vals), max(svf_vals))
    logger.info(
        f"[VFG] SVF map loaded: {len(hbjson_vfg)} outdoor faces | SVF range = [{svf_min:.4f}, {svf_max:.4f}] (VFG = 1-SVF range = [{1 - svf_max:.4f}, {1 - svf_min:.4f}])"
    )
    if svf_max - svf_min < 0.01:
        logger.warning(
            "[VFG] All outdoor faces have nearly identical SVF values — check that assign_program SVF gradient is being saved to HBJSON."
        )
    with open(idf_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.compile("(BuildingSurface:Detailed\\s*,\\s*.*?;)", re.DOTALL | re.IGNORECASE)
    blocks = pattern.findall(content)
    if not blocks:
        logger.warning("[VFG] No BuildingSurface:Detailed blocks found in IDF.")
        return 0

    def _patch_single_block(block: str, vfg: float) -> str:
        VFG_FIELD_CANDIDATES = (9, 10)
        lines = block.split("\n")
        skip_keyword = True
        field_idx = 0
        new_lines = []
        patched = False
        for line in lines:
            if skip_keyword:
                new_lines.append(line)
                skip_keyword = False
                continue
            comment_pos = line.find("!")
            code_part = line[:comment_pos] if comment_pos >= 0 else line
            comment_part = " " + line[comment_pos:].rstrip() if comment_pos >= 0 else ""
            terminators = [c for c in code_part if c in ",;"]
            n_terms = len(terminators)
            if not patched and n_terms == 1:
                for target in VFG_FIELD_CANDIDATES:
                    if field_idx == target:
                        terminator = ";" if ";" in code_part else ","
                        new_lines.append(f"    {vfg}{terminator}{comment_part}")
                        patched = True
                        break
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
            field_idx += n_terms
        return "\n".join(new_lines)

    patched_blocks = []
    patched_count = 0
    unmatched = []
    patch_detail = []
    for block in blocks:
        lines = block.split("\n")
        surface_name = None
        for line in lines[1:]:
            stripped = line.strip()
            if stripped and (not stripped.startswith("!")):
                surface_name = stripped.split(",")[0].split(";")[0].strip().upper()
                break
        if surface_name and surface_name in hbjson_vfg:
            vfg = hbjson_vfg[surface_name]
            patched_block = _patch_single_block(block, vfg)
            patched_blocks.append(patched_block)
            patched_count += 1
            patch_detail.append((surface_name, vfg))
        else:
            patched_blocks.append(block)
            if surface_name:
                unmatched.append(surface_name)
    logger.info(f"[VFG] Patch result: {patched_count} patched / {len(blocks)} total surfaces")
    if patch_detail:
        logger.info("[VFG] Per-surface VFG written to IDF (first 20):")
        for name, vfg in patch_detail[:20]:
            logger.info(f"      {name:<45s} VFG={vfg:.4f}  (SVF={1 - vfg:.4f})")
        if len(patch_detail) > 20:
            logger.info(f"      ... and {len(patch_detail) - 20} more")
    else:
        logger.error(
            "[VFG] CRITICAL: 0 surfaces were patched! Likely cause: face identifiers in IDF do not match HBJSON. Check if _strip_indoor_shades or fix_adjacency renamed any faces."
        )
    if unmatched:
        logger.debug(
            f"[VFG] {len(unmatched)} IDF surfaces NOT matched to HBJSON — these keep autocalculate."
        )
        logger.debug("[VFG] Unmatched IDF surface names (first 30):")
        for name in unmatched[:30]:
            logger.debug(f"      IDF  : {name}")
        logger.debug("[VFG] HBJSON outdoor face identifiers (first 30, uppercased):")
        for name in list(hbjson_vfg.keys())[:30]:
            logger.debug(f"      HBJSON: {name}")
    block_iter = iter(patched_blocks)

    def replacer(match):
        """Compute replacer for the current building or model state. Inputs: match."""
        try:
            return next(block_iter)
        except StopIteration:
            return match.group(0)

    new_content = pattern.sub(replacer, content)
    with open(idf_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    with open(idf_path, "r", encoding="utf-8") as f:
        verify_content = f.read()
    remaining_auto = len(
        re.findall(
            "BuildingSurface:Detailed.*?autocalculate.*?;",
            verify_content,
            re.DOTALL | re.IGNORECASE,
        )
    )
    if remaining_auto > 0:
        logger.debug(
            f"[VFG] POST-WRITE CHECK: {remaining_auto} BuildingSurface:Detailed block(s) still contain 'autocalculate'. SVF patch may be incomplete."
        )
    else:
        logger.info(
            f"[VFG] POST-WRITE CHECK: all outdoor surfaces confirmed written. No residual 'autocalculate' in patched surfaces."
        )
    logger.info(f"[VFG] IDF saved: {idf_path}")
    return patched_count


def _append_hvac_other_equipment(idf_path: Path, hbjson_path: Path) -> int:
    meta_path = _find_meta_json(hbjson_path)
    if meta_path is None:
        logger.info("  [HVAC] No _meta.json found. Skipping OtherEquipment injection.")
        return 0
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    program = meta.get("material", {}).get("program", {})
    hvac_modes = program.get("room_hvac_mode", [])
    hvac_schedules = program.get("room_hvac_schedule", [])
    if not hvac_modes or not hvac_schedules:
        logger.info("  [HVAC] No HVAC data in metadata. Skipping OtherEquipment injection.")
        return 0
    with open(hbjson_path, "r", encoding="utf-8") as f:
        hbjson_data = json.load(f)
    room_ids = [
        r.get("identifier", f"room_{i}") for i, r in enumerate(hbjson_data.get("rooms", []))
    ]
    n = min(len(room_ids), len(hvac_modes), len(hvac_schedules))
    if n < len(room_ids):
        logger.warning(
            f"  [HVAC] Length mismatch: rooms={len(room_ids)}, hvac_modes={len(hvac_modes)}, schedules={len(hvac_schedules)}. Processing only first {n} rooms."
        )
    idf_blocks = []
    any_number_added = False
    count = 0
    for idx in range(n):
        room_id = room_ids[idx]
        mode = hvac_modes[idx]
        sched_24h = hvac_schedules[idx]
        if mode == "none":
            continue
        if not sched_24h or max((abs(v) for v in sched_24h)) < 1e-09:
            logger.debug(f"  [HVAC] Room '{room_id}': schedule is all-zero, skipping.")
            continue
        if not any_number_added:
            idf_blocks.append(_idf_any_number_schedule_type())
            any_number_added = True
        idf_blocks.append(_idf_hvac_schedule_compact(room_id, sched_24h))
        idf_blocks.append(_idf_other_equipment(room_id, mode))
        count += 1
    if idf_blocks:
        with open(idf_path, "a", encoding="utf-8") as f:
            f.write(
                "\n\n! ===== HVAC OtherEquipment — appended by to_idf.py =====\n! Design Level Calculation: Watts/Area = 1.0, schedule = actual W/m² (signed)\n! Positive = heating, Negative = cooling\n\n"
            )
            for block in idf_blocks:
                f.write(block)
        logger.info(
            f"  [HVAC] Appended {count} OtherEquipment objects ({count} Schedule:Compact + 1 ScheduleTypeLimits)."
        )
    else:
        logger.info("  [HVAC] All rooms have mode='none'. No OtherEquipment appended.")
    return count


def _build_surr_temp_schedules(epw_path: Optional[Path]) -> tuple[str, str, str]:
    SKY_SCHED = "SurrSkyTemp"
    GND_SCHED = "SurrGndTemp"
    SURR_SCHED = "SurrBldgTemp"
    type_limits = "\nScheduleTypeLimits,\n    Temperature,    !- Name\n    -100,           !- Lower Limit Value\n    200,            !- Upper Limit Value\n    Continuous,     !- Numeric Type\n    Temperature;    !- Unit Type\n"
    hourly_drybulb: Optional[list] = None
    if epw_path and epw_path.exists():
        try:
            from ladybug.epw import EPW

            epw = EPW(str(epw_path))
            hourly_drybulb = list(epw.dry_bulb_temperature.values)
            logger.info(
                f"[Surr] EPW loaded for temperature schedules: T_db range [{min(hourly_drybulb):.1f}, {max(hourly_drybulb):.1f}] °C"
            )
        except Exception as e:
            logger.warning(f"[Surr] Failed to read EPW ({e}). Falling back to constant schedules.")

    def _compact_from_hourly(name: str, values: list, offset: float = 0.0) -> str:
        import calendar

        vals = [v + offset for v in values]
        month_hour_avgs: list[list[float]] = []
        hour_ptr = 0
        month_ends = []
        for m in range(1, 13):
            days_in_month = calendar.monthrange(2025, m)[1]
            hours_in_month = days_in_month * 24
            month_vals = vals[hour_ptr : hour_ptr + hours_in_month]
            hour_ptr += hours_in_month
            hour_avgs = []
            for h in range(24):
                h_vals = month_vals[h::24]
                hour_avgs.append(round(sum(h_vals) / len(h_vals), 2) if h_vals else 0.0)
            month_hour_avgs.append(hour_avgs)
            last_day = calendar.monthrange(2025, m)[1]
            month_ends.append(f"{m:02d}/{last_day:02d}")
        lines = [f"\nSchedule:Compact,", f"    {name},", f"    Temperature,"]
        for mi, (month_end, hour_avgs) in enumerate(zip(month_ends, month_hour_avgs)):
            terminator_month = "," if mi < 11 else ""
            lines.append(f"    Through: {month_end},")
            lines.append(f"    For: AllDays,")
            for h, val in enumerate(hour_avgs):
                is_last_hour = h == 23
                is_last_month = mi == 11
                term = ";" if is_last_hour and is_last_month else ","
                lines.append(f"    Until: {h + 1:02d}:00, {val}{term}")
        return "\n".join(lines) + "\n"

    def _constant_schedule(name: str, value: float) -> str:
        return f"\nSchedule:Compact,\n    {name},\n    Temperature,\n    Through: 12/31,\n    For: AllDays,\n    Until: 24:00, {value};\n"

    if hourly_drybulb and len(hourly_drybulb) >= 8760:
        sky_text = _compact_from_hourly(SKY_SCHED, hourly_drybulb, offset=-11.0)
        gnd_text = _compact_from_hourly(GND_SCHED, hourly_drybulb, offset=+2.0)
        surr_text = _compact_from_hourly(SURR_SCHED, hourly_drybulb, offset=0.0)
    else:
        logger.warning("[Surr] Using constant 10°C for all surrounding temperature schedules.")
        sky_text = _constant_schedule(SKY_SCHED, -1.0)
        gnd_text = _constant_schedule(GND_SCHED, 12.0)
        surr_text = _constant_schedule(SURR_SCHED, 10.0)
    full_text = type_limits + sky_text + gnd_text + surr_text
    return (SKY_SCHED, GND_SCHED, SURR_SCHED, full_text)


def _append_surrounding_surfaces(
    idf_path: Path, hbjson_path: Path, epw_path: Optional[Path] = None
) -> int:
    import re

    with open(hbjson_path, "r", encoding="utf-8") as f:
        hbjson = json.load(f)
    face_info: dict[str, tuple[float, str]] = {}
    for room in hbjson.get("rooms", []):
        for face in room.get("faces", []):
            bc = face.get("boundary_condition", {})
            if not (isinstance(bc, dict) and bc.get("type") == "Outdoors"):
                continue
            raw_svf = bc.get("view_factor")
            if not isinstance(raw_svf, (int, float)):
                continue
            svf = max(0.0, min(1.0, float(raw_svf)))
            face_type = face.get("face_type", "Wall")
            face_info[face["identifier"]] = (svf, face_type)
    if not face_info:
        logger.info("[Surr] No outdoor faces with SVF found. Skipping SurroundingSurfaces.")
        return 0
    with open(idf_path, "r", encoding="utf-8") as f:
        content = f.read()
    block_pat = re.compile("BuildingSurface:Detailed\\s*,\\s*.*?;", re.DOTALL | re.IGNORECASE)
    idf_faces: dict[str, str] = {}
    for block in block_pat.findall(content):
        for line in block.split("\n")[1:]:
            stripped = line.strip()
            if stripped and (not stripped.startswith("!")):
                name = stripped.split(",")[0].split(";")[0].strip()
                idf_faces[name.upper()] = name
                break
    matched: list[tuple[str, float, str]] = []
    unmatched_hb: list[str] = []
    for hb_id, (svf, face_type) in face_info.items():
        idf_name = idf_faces.get(hb_id.upper())
        if idf_name:
            matched.append((idf_name, svf, face_type))
        else:
            unmatched_hb.append(hb_id)
    if unmatched_hb:
        logger.debug(
            f"[Surr] {len(unmatched_hb)} HBJSON faces not matched in IDF (apertures / sanitized names): "
            + ", ".join(unmatched_hb[:5])
            + ("..." if len(unmatched_hb) > 5 else "")
        )
    if not matched:
        logger.warning("[Surr] No faces matched. Skipping SurroundingSurfaces.")
        return 0
    sky_sched, gnd_sched, surr_sched, sched_text = _build_surr_temp_schedules(epw_path)

    def _surr_block(idf_name: str, svf: float, face_type: str) -> str:
        surr_name = f"{idf_name}_Surr"
        env_name = f"{idf_name}_LocalEnv"
        is_roof = face_type.lower() in ("roofceiling", "roof", "ceiling")
        sky_vf = round(svf, 4)
        remain = round(1.0 - svf, 4)
        gnd_frac = 0.1 if is_roof else 0.5
        gnd_vf = round(remain * gnd_frac, 4)
        surr_vf = round(remain - gnd_vf, 4)
        surr_obj = f"\nSurfaceProperty:SurroundingSurfaces,\n    {surr_name},       !- Name\n    {sky_vf},          !- Sky View Factor\n    {sky_sched},       !- Sky Temperature Schedule Name\n    {gnd_vf},          !- Ground View Factor\n    {gnd_sched},       !- Ground Temperature Schedule Name\n    SurroundingBuildings, !- Surrounding Surface 1 Object Type\n    {surr_vf},         !- Surrounding Surface 1 View Factor\n    {surr_sched};      !- Surrounding Surface 1 Temperature Schedule\n"
        env_obj = f"\nSurfaceProperty:LocalEnvironment,\n    {env_name},        !- Name\n    {idf_name},        !- Exterior Surface Name\n    ,                  !- External Shading Fraction Schedule Name\n    {surr_name},       !- Surrounding Surfaces Object Name\n    ;                  !- Outdoor Air Node Name\n"
        return surr_obj + env_obj

    header = "\n\n! ===== SurfaceProperty:SurroundingSurfaces — appended by to_idf.py =====\n! EP 25.2 compatible: all temperature schedules filled from EPW or constant fallback\n! VF partition: wall: sky=SVF, gnd=(1-SVF)*0.5, surr=(1-SVF)*0.5\n!               roof: sky=SVF, gnd=(1-SVF)*0.1, surr=(1-SVF)*0.9\n\n"
    face_blocks = "".join((_surr_block(n, s, t) for n, s, t in matched))
    with open(idf_path, "a", encoding="utf-8") as f:
        f.write(header + sched_text + face_blocks)
    logger.info(
        f"[Surr] Appended {len(matched)} SurroundingSurfaces pairs. SVF range: [{min((s for _, s, _ in matched)):.4f}, {max((s for _, s, _ in matched)):.4f}]"
    )
    return len(matched)


def convert_hbjson_to_idf(
    hbjson_path: Path, output_dir: Path, auto_fix: bool = True, epw_path: Optional[Path] = None
) -> Path:
    """Translate HBJSON to an annual IDF with the required radiative and heat-flow settings."""
    logger.info(f"[Convert] Starting conversion for: {hbjson_path.name}")
    try:
        from honeybee.model import Model
        from honeybee_energy.simulation.parameter import SimulationParameter
        from honeybee_energy.simulation.runperiod import RunPeriod
        from ladybug.analysisperiod import AnalysisPeriod
        from honeybee_energy.writer import model_to_idf, energyplus_idf_version
    except ImportError as e:
        raise ImportError(
            "[!] Missing dependencies. Please run: pip install honeybee-energy"
        ) from e
    if auto_fix:
        logger.info("  [Topo] Running adjacency topology check...")
        hbjson_path, fixed_count = fix_adjacency_and_return(hbjson_path, save_fixed=True)
        if fixed_count > 0:
            logger.info(
                f"  [Topo] Topology repaired. Continuing with fixed file: {hbjson_path.name}"
            )
    hbjson_for_idf = _strip_indoor_shades(hbjson_path)
    model = Model.from_file(str(hbjson_for_idf))
    logger.info(
        f"  Model loaded successfully: {len(model.rooms)} rooms, {len(model.orphaned_shades)} orphaned shades."
    )
    sim_par = SimulationParameter()
    full_year_period = AnalysisPeriod(st_month=1, st_day=1, end_month=12, end_day=31)
    sim_par.run_period = RunPeriod.from_analysis_period(full_year_period)
    model.properties.energy.simulation_parameter = sim_par
    output_dir.mkdir(parents=True, exist_ok=True)
    idf_path = output_dir / f"{hbjson_path.stem.removesuffix('_fixed')}.idf"
    logger.info(f"  Writing IDF to: {idf_path}")
    version_str = energyplus_idf_version() or ""
    sim_par_str = sim_par.to_idf()
    model_str = model_to_idf(model)
    with open(idf_path, "w", encoding="utf-8") as f:
        if version_str:
            f.write(version_str + "\n\n")
        f.write(sim_par_str + "\n\n")
        f.write(model_str + "\n")
    logger.info(f"[Success] IDF generated at: {idf_path}")
    vfg_count = _patch_view_factor_to_ground(idf_path, hbjson_path)
    if vfg_count > 0:
        logger.info(f"[VFG] {vfg_count} 'View Factor to Ground' fields updated in IDF.")
    hvac_count = _append_hvac_other_equipment(idf_path, hbjson_path)
    if hvac_count > 0:
        logger.info(f"[HVAC] {hvac_count} OtherEquipment blocks appended to IDF.")
    surr_count = _append_surrounding_surfaces(idf_path, hbjson_path, epw_path=epw_path)
    if surr_count > 0:
        logger.info(f"[Surr] {surr_count} SurroundingSurfaces pairs appended to IDF.")
    if hbjson_for_idf != hbjson_path:
        hbjson_for_idf.unlink(missing_ok=True)
    return idf_path
