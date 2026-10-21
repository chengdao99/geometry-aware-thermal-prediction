"""Run EnergyPlus and extract hourly zone labels. Inputs: IDF, HBJSON metadata, and EPW. Outputs: Annual label NPZ."""

import logging
import subprocess
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .. import config

logger = logging.getLogger(__name__)
DEFAULT_EP_PATH = config.DEFAULT_EP_PATH
EP_OUTPUT_VARIABLES = config.EP_OUTPUT_VARIABLES
GEOM_DIM = 13


def inject_output_variables(idf_path: Path):
    """Compute inject output variables for the current building or model state. Inputs: idf_path."""
    logger.info("[Inject] Scanning IDF for Output:Variable requests...")
    with open(idf_path, "r", encoding="utf-8") as f:
        content = f.read()
    content_lower = content.lower()
    injections = []
    for var_name in EP_OUTPUT_VARIABLES:
        if var_name.lower() not in content_lower:
            injections.append(
                f"\nOutput:Variable,\n  *, !- Key Value\n  {var_name}, !- Variable Name\n  Hourly; !- Reporting Frequency\n"
            )
            logger.info(f"  [+] Injected: {var_name}")
    if "outputcontrol:files" not in content_lower:
        injections.append(
            "\nOutputControl:Files,\n  Yes, !- Output CSV\n  No, !- Output MTR\n  No, !- Output ESO\n  No, !- Output EIO\n  No, !- Output Tabular\n  No, !- Output SQLite\n  No, !- Output JSON\n  No, !- Output AUDIT\n  No, !- Output Zone Sizing\n  No, !- Output System Sizing\n  No, !- Output DXF\n  No, !- Output BND\n  No, !- Output RDD\n  No, !- Output MDD\n  No, !- Output MTD\n  No, !- Output END\n  No, !- Output SHD\n  No, !- Output DFS\n  No; !- Output GLHE\n"
        )
    if injections:
        with open(idf_path, "a", encoding="utf-8") as f:
            f.write("\n\n! ========== Auto-injected by idf_simulator.py ==========\n")
            for block in injections:
                f.write(block)
        logger.info(f"  Injection complete. Added {len(injections)} blocks.")
    else:
        logger.info("  No injection needed. Variables already exist.")


def run_energyplus(ep_exe: Path, idf_path: Path, epw_path: Path, output_dir: Path) -> float:
    """Run EnergyPlus with an IDF and EPW and reject unsuccessful simulations."""
    logger.info(f"[Simulate] Running EnergyPlus engine...")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(ep_exe), "-w", str(epw_path), "-d", str(output_dir), str(idf_path)]
    logger.info(f"  Command: {' '.join(cmd)}")
    start_time = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("[!] EnergyPlus simulation timed out (>1 hour).")
    elapsed = time.perf_counter() - start_time
    if result.returncode != 0:
        err_log_path = output_dir / "ep_stderr.log"
        with open(err_log_path, "w") as f:
            f.write(result.stderr or "")
            f.write(result.stdout or "")
        raise RuntimeError(
            f"[!] EnergyPlus failed (returncode={result.returncode}). Check logs at: {err_log_path}"
        )
    err_file = output_dir / "eplusout.err"
    if err_file.exists():
        with open(err_file, "r", encoding="utf-8", errors="ignore") as f:
            if "** Fatal **" in f.read():
                raise RuntimeError(f"[!] EnergyPlus encountered a FATAL error. Check: {err_file}")
    logger.info(f"  Simulation completed successfully in {elapsed:.1f} seconds.")
    return elapsed


def _polygon_area_3d(pts):
    if len(pts) < 3:
        return 0.0
    pts_arr = np.array(pts, dtype=np.float32)
    cross_sum = np.zeros(3, dtype=np.float32)
    for i in range(len(pts_arr)):
        cross_sum += np.cross(pts_arr[i], pts_arr[(i + 1) % len(pts_arr)])
    return 0.5 * np.linalg.norm(cross_sum)


def _polygon_area_3d(pts):
    if len(pts) < 3:
        return 0.0
    pts_arr = np.array(pts, dtype=np.float32)
    cross_sum = np.zeros(3, dtype=np.float32)
    for i in range(len(pts_arr)):
        cross_sum += np.cross(pts_arr[i], pts_arr[(i + 1) % len(pts_arr)])
    return 0.5 * np.linalg.norm(cross_sum)


def _compute_face_normal(pts) -> np.ndarray:
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    pts_arr = np.array(pts, dtype=np.float32)
    cross_sum = np.zeros(3, dtype=np.float32)
    for i in range(len(pts_arr)):
        cross_sum += np.cross(pts_arr[i], pts_arr[(i + 1) % len(pts_arr)])
    norm = np.linalg.norm(cross_sum)
    if norm < 1e-06:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (-cross_sum / norm).astype(np.float32)


def _extract_zone_ach_matrix(
    data: dict, geom_features: np.ndarray, n_hours: int = 8760
) -> np.ndarray:
    rooms = data.get("rooms", [])
    if not rooms:
        logger.warning("[ACH] No rooms found in HBJSON.")
        return np.zeros((n_hours, 0), dtype=np.float32)
    schedules = data.get("properties", {}).get("energy", {}).get("schedules", [])
    sched_dict = {}
    for s in schedules:
        if s.get("type") == "ScheduleRulesetAbridged":
            sched_dict[s["identifier"]] = s
    Z = len(rooms)
    ach_matrix = np.zeros((n_hours, Z), dtype=np.float32)
    for idx, room in enumerate(rooms):
        room_id = room.get("identifier", f"room_{idx}")
        try:
            infil = room.get("properties", {}).get("energy", {}).get("infiltration", {})
            if not infil:
                raise ValueError("Missing infiltration object")
            sched_id = infil.get("schedule")
            if not sched_id or sched_id not in sched_dict:
                raise ValueError(f"Schedule '{sched_id}' not found")
            fpa = infil.get("flow_per_exterior_area", 0.0)
            volume = geom_features[idx, 0]
            floor_area = geom_features[idx, 1]
            if volume <= 1e-06:
                raise ValueError(f"Invalid volume: {volume:.3f}")
            ext_area_gross = geom_features[idx, 3] + geom_features[idx, 4] + geom_features[idx, 5]
            if ext_area_gross > 1e-06:
                ach_day = fpa * 3600.0 * (ext_area_gross / volume)
            elif floor_area > 1e-06:
                ach_day = fpa * 3600.0 * (floor_area / volume)
            else:
                raise ValueError(
                    f"No valid reference area (ext={ext_area_gross:.3f}, floor={floor_area:.3f})"
                )
            if ach_day < 0:
                raise ValueError(f"Negative ach_day: {ach_day:.4f}")
            sched = sched_dict[sched_id]
            day_schedules = sched.get("day_schedules", [])
            if not day_schedules:
                raise ValueError("No day_schedules in schedule")
            day_sched = day_schedules[0]
            raw_times = day_sched.get("times", [])
            raw_values = day_sched.get("values", [])
            if not raw_times or not raw_values or len(raw_times) != len(raw_values):
                raise ValueError("Times/values malformed")
            start_hours = [t[0] if isinstance(t, (list, tuple)) else int(t) for t in raw_times]
            fractions = [float(v) for v in raw_values]
            daily_frac = np.zeros(24, dtype=np.float32)
            for i, (h_start, frac) in enumerate(zip(start_hours, fractions)):
                h_end = start_hours[i + 1] if i + 1 < len(start_hours) else 24
                daily_frac[h_start:h_end] = frac
            repeats = n_hours // 24
            remainder = n_hours % 24
            hourly_frac = np.concatenate([np.tile(daily_frac, repeats), daily_frac[:remainder]])
            ach_matrix[:, idx] = ach_day * hourly_frac
        except Exception as e:
            logger.warning(
                f"[ACH] Failed to extract ACH for room '{room_id}': {e}. Setting ACH series to 0 for this room."
            )
    non_zero_rooms = np.sum(np.any(ach_matrix > 0, axis=0))
    logger.info(f"[ACH] Successfully extracted ACH series for {non_zero_rooms}/{Z} rooms.")
    return ach_matrix


def _broadcast_24h_to_8760(schedule_24h: list, n_hours: int = 8760) -> np.ndarray:
    daily = np.array(schedule_24h, dtype=np.float32)
    if len(daily) != 24:
        logger.warning(
            f"[Broadcast] schedule_24h has length {len(daily)}, expected 24. Padding/truncating to 24."
        )
        tmp = np.zeros(24, dtype=np.float32)
        tmp[: min(len(daily), 24)] = daily[:24]
        daily = tmp
    repeats = n_hours // 24
    remainder = n_hours % 24
    return np.concatenate([np.tile(daily, repeats), daily[:remainder]])


def _extract_feature_schedules(
    hbjson_path: Path, zone_names: np.ndarray, n_hours: int = 8760
) -> dict[str, np.ndarray]:
    Z = len(zone_names)
    zero_result = {
        "equip": np.zeros((n_hours, Z), dtype=np.float32),
        "occ": np.zeros((n_hours, Z), dtype=np.float32),
        "hvac": np.zeros((n_hours, Z), dtype=np.float32),
    }
    meta_path = _find_meta_json(hbjson_path)
    if meta_path is None:
        logger.warning(
            f"[FeatureSched] No _meta.json found for {hbjson_path.name}. Returning zero matrices for equip/occ/hvac."
        )
        return zero_result
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    program = meta.get("material", {}).get("program", {})
    if not program:
        logger.warning("[FeatureSched] Empty program metadata. Returning zero matrices.")
        return zero_result
    equip_schedules = program.get("room_equip_schedule", [])
    occ_schedules = program.get("room_occ_schedule", [])
    hvac_schedules = program.get("room_hvac_schedule", [])
    with open(hbjson_path, "r", encoding="utf-8") as f:
        hbjson_data = json.load(f)
    meta_room_ids = [
        r.get("identifier", f"room_{i}").upper() for i, r in enumerate(hbjson_data.get("rooms", []))
    ]
    room_id_to_idx: dict[str, int] = {rid: i for i, rid in enumerate(meta_room_ids)}
    equip_matrix = np.zeros((n_hours, Z), dtype=np.float32)
    occ_matrix = np.zeros((n_hours, Z), dtype=np.float32)
    hvac_matrix = np.zeros((n_hours, Z), dtype=np.float32)
    n_meta = len(meta_room_ids)
    missing = []
    for col_idx, zone_name in enumerate(zone_names):
        meta_idx = room_id_to_idx.get(zone_name)
        if meta_idx is None:
            missing.append(zone_name)
            continue
        if meta_idx < len(equip_schedules) and equip_schedules[meta_idx]:
            equip_matrix[:, col_idx] = _broadcast_24h_to_8760(equip_schedules[meta_idx], n_hours)
        if meta_idx < len(occ_schedules) and occ_schedules[meta_idx]:
            occ_matrix[:, col_idx] = _broadcast_24h_to_8760(occ_schedules[meta_idx], n_hours)
        if meta_idx < len(hvac_schedules) and hvac_schedules[meta_idx]:
            hvac_matrix[:, col_idx] = _broadcast_24h_to_8760(hvac_schedules[meta_idx], n_hours)
    if missing:
        logger.warning(
            f"[FeatureSched] {len(missing)} zones not found in meta room list, set to zero: {missing[:5]}{('...' if len(missing) > 5 else '')}"
        )
    scenario = program.get("scenario_type", "unknown")
    non_zero_equip = int(np.any(equip_matrix != 0, axis=0).sum())
    non_zero_hvac = int(np.any(hvac_matrix != 0, axis=0).sum())
    logger.info(
        f"[FeatureSched] scenario={scenario} | equip active zones: {non_zero_equip}/{Z} | hvac active zones: {non_zero_hvac}/{Z}"
    )
    return {"equip": equip_matrix, "occ": occ_matrix, "hvac": hvac_matrix}


def extract_zone_spatial_features(data: dict, zone_names: np.ndarray):
    """Extract ordered zone centroids, physical geometry, and contact areas."""
    room_dict: dict[str, dict] = {}
    for room in data.get("rooms", []):
        room_dict[room.get("identifier", "").upper()] = room
        disp = room.get("display_name", "").upper()
        if disp:
            room_dict[disp] = room
    Z = len(zone_names)
    zone_to_idx = {name.upper(): i for i, name in enumerate(zone_names)}
    contact_matrix = np.zeros((Z, Z), dtype=np.float32)
    centroids = []
    geom_features = []
    missing_zones = []
    for z_name in zone_names:
        if room_dict.get(z_name) is None:
            missing_zones.append(z_name)
    for tgt_idx, z_name in enumerate(zone_names):
        room = room_dict.get(z_name)
        if room is None:
            logger.warning(f"  [!] Room not found for Zone: {z_name}. Using zero vectors.")
            centroids.append(np.zeros(3, dtype=np.float32))
            geom_features.append(np.zeros(GEOM_DIM, dtype=np.float32))
            continue
        all_pts = []
        roof_z = []
        floor_z = []
        f_total_floor = 0.0
        f_ground_floor = 0.0
        f_ext_wall_nsew = np.zeros(4, dtype=np.float32)
        f_ext_win_nsew = np.zeros(4, dtype=np.float32)
        f_ext_roof = 0.0
        f_svf_weighted = 0.0
        f_ext_total = 0.0
        for face in room.get("faces", []):
            pts = face["geometry"]["boundary"]
            all_pts.extend(pts)
            area = _polygon_area_3d(pts)
            face_type = face.get("face_type", "")
            bc_dict = face.get("boundary_condition", {})
            bc_type = bc_dict.get("type", "")
            if bc_type == "Surface":
                bc_objs = bc_dict.get("boundary_condition_objects", [])
                if isinstance(bc_objs, list) and len(bc_objs) >= 2:
                    adj_room_id = str(bc_objs[1]).upper()
                    if adj_room_id in zone_to_idx:
                        nb_idx = zone_to_idx[adj_room_id]
                        contact_matrix[tgt_idx, nb_idx] += area
            normal = _compute_face_normal(pts)
            n_x, n_y, n_z = (normal[0], normal[1], normal[2])
            win_area = 0.0
            if "apertures" in face:
                for ap in face["apertures"]:
                    ap_area = _polygon_area_3d(ap["geometry"]["boundary"])
                    win_area += ap_area
                    if bc_type == "Outdoors":
                        if n_y > 0:
                            f_ext_win_nsew[0] += ap_area * n_y
                        if n_y < 0:
                            f_ext_win_nsew[1] += ap_area * -n_y
                        if n_x > 0:
                            f_ext_win_nsew[2] += ap_area * n_x
                        if n_x < 0:
                            f_ext_win_nsew[3] += ap_area * -n_x
            net_area = max(0.0, area - win_area)
            if face_type == "Floor":
                f_total_floor += area
                floor_z.extend([p[2] for p in pts])
                if bc_type == "Ground":
                    f_ground_floor += area
            else:
                if bc_type == "Outdoors":
                    if n_y > 0:
                        f_ext_wall_nsew[0] += net_area * n_y
                    if n_y < 0:
                        f_ext_wall_nsew[1] += net_area * -n_y
                    if n_x > 0:
                        f_ext_wall_nsew[2] += net_area * n_x
                    if n_x < 0:
                        f_ext_wall_nsew[3] += net_area * -n_x
                    if n_z > 0:
                        f_ext_roof += net_area * n_z
                if face_type == "RoofCeiling":
                    roof_z.extend([p[2] for p in pts])
            if bc_type == "Outdoors":
                svf = float(face.get("boundary_condition", {}).get("view_factor", 0.0))
                f_svf_weighted += svf * area
                f_ext_total += area
        weighted_sum = np.zeros(3, dtype=np.float32)
        total_weight = 0.0
        for face in room.get("faces", []):
            pts = face["geometry"]["boundary"]
            face_centroid = np.mean(np.array(pts, dtype=np.float32), axis=0)
            area = _polygon_area_3d(pts)
            weighted_sum += face_centroid * area
            total_weight += area
        centroid = weighted_sum / total_weight if total_weight > 0 else np.zeros(3)
        all_pts_arr = np.array(all_pts, dtype=np.float32)
        if roof_z and floor_z:
            avg_roof_z = (max(roof_z) + min(roof_z)) / 2.0
            avg_floor_z = sum(floor_z) / len(floor_z)
            avg_height = max(0.1, avg_roof_z - avg_floor_z)
        elif len(all_pts_arr) > 0:
            avg_height = max(0.1, np.max(all_pts_arr[:, 2]) - np.min(all_pts_arr[:, 2]))
        else:
            avg_height = 0.0
        volume = f_total_floor * avg_height
        zone_svf = f_svf_weighted / f_ext_total if f_ext_total > 1e-06 else 0.0
        centroids.append(centroid)
        geom_features.append(
            [
                volume,
                f_total_floor,
                f_ground_floor,
                f_ext_wall_nsew[0],
                f_ext_wall_nsew[1],
                f_ext_wall_nsew[2],
                f_ext_wall_nsew[3],
                f_ext_win_nsew[0],
                f_ext_win_nsew[1],
                f_ext_win_nsew[2],
                f_ext_win_nsew[3],
                f_ext_roof,
                zone_svf,
            ]
        )
    if missing_zones:
        raise ValueError(
            f"Zone alignment failed! {len(missing_zones)} zones from EnergyPlus CSV were not found in HBJSON: {missing_zones}"
        )
    return (
        np.array(centroids, dtype=np.float32),
        np.array(geom_features, dtype=np.float32),
        contact_matrix,
    )


def _find_meta_json(hbjson_path: Path):
    stem = hbjson_path.stem.removesuffix("_fixed")
    candidates = [
        hbjson_path.parent / f"{stem}_meta.json",
        hbjson_path.parent / "metadata" / f"{stem}_meta.json",
        hbjson_path.parent.parent / "json" / f"{stem}_meta.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _extract_shade_params_from_meta(hbjson_path: Path) -> dict:
    meta_path = _find_meta_json(hbjson_path)
    defaults = {
        "shade_depth_NSEW": np.zeros(4, dtype=np.float32),
        "head_height": 2.4,
        "sill_height": 0.9,
        "shade_gap": 0.1,
    }
    if meta_path is None or not meta_path.exists():
        logger.warning(
            f"[_extract_shade_params] _meta.json not found for {hbjson_path.name}. Using zero shading depth (no exterior shading)."
        )
        return defaults
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning(f"[_extract_shade_params] Failed to parse {meta_path}: {e}. Using defaults.")
        return defaults
    geom = meta.get("geometry", {})
    depth_dict = geom.get("out_shade_depth_by_orientation", {})
    shade_depth_NSEW = np.array(
        [
            float(depth_dict.get("N", 0.0)),
            float(depth_dict.get("S", 0.0)),
            float(depth_dict.get("E", 0.0)),
            float(depth_dict.get("W", 0.0)),
        ],
        dtype=np.float32,
    )
    head_height = float(geom.get("head_height_m", 2.4))
    sill_height = float(geom.get("sill_height_m", 0.9))
    shade_gap = float(geom.get("out_shade_gap_m", 0.1))
    logger.debug(
        f"[_extract_shade_params] depth N/S/E/W = {shade_depth_NSEW} | head={head_height:.2f} sill={sill_height:.2f} gap={shade_gap:.2f}"
    )
    return {
        "shade_depth_NSEW": shade_depth_NSEW,
        "head_height": head_height,
        "sill_height": sill_height,
        "shade_gap": shade_gap,
    }


def extract_and_save_labels_npz(
    output_dir: Path, model_name: str, export_dir: Path, hbjson_path: Path
):
    """Align annual EnergyPlus temperatures and zone metadata and write a label NPZ."""
    logger.info("[Parse] Extracting temperature matrices and aligning physical features...")
    with open(hbjson_path, "r", encoding="utf-8") as f:
        hbjson_data = json.load(f)
    csv_candidates = [output_dir / "eplusout.csv"] + list(output_dir.glob("*out.csv"))
    csv_path = next((c for c in csv_candidates if c.exists() and c.stat().st_size > 100), None)
    if not csv_path:
        raise FileNotFoundError(f"[!] No valid EnergyPlus CSV output found in {output_dir}")
    df_raw = pd.read_csv(csv_path, low_memory=False)
    time_col = df_raw.columns[0]
    df_raw = df_raw.drop(columns=[time_col])

    def get_matrix_and_zones(keyword):
        """Compute get matrix and zones for the current building or model state. Inputs: keyword."""
        cols = [c for c in df_raw.columns if keyword in c]
        if not cols:
            return (np.empty((0, 0), dtype=np.float32), np.array([], dtype=str))
        zones = np.array([c.split(":")[0].strip().upper() for c in cols], dtype=str)
        matrix = df_raw[cols].values.astype(np.float32)
        return (matrix, zones)

    air_matrix, air_zones = get_matrix_and_zones("Zone Mean Air Temperature")
    op_matrix, op_zones = get_matrix_and_zones("Zone Operative Temperature")
    if air_matrix.size == 0 and op_matrix.size == 0:
        raise RuntimeError(
            "[!] No temperature output variables found in CSV. Did simulation run correctly?"
        )
    num_timesteps = air_matrix.shape[0] if air_matrix.size > 0 else op_matrix.shape[0]
    logger.info(f"  Extracted timesteps: {num_timesteps}, zones: {len(air_zones)}")
    centroids, geom_features, contact_matrix = extract_zone_spatial_features(hbjson_data, air_zones)
    ach_matrix = _extract_zone_ach_matrix(hbjson_data, geom_features, n_hours=num_timesteps or 8760)
    feature_schedules = _extract_feature_schedules(
        hbjson_path, air_zones, n_hours=num_timesteps or 8760
    )
    equip_matrix = feature_schedules["equip"]
    occ_matrix = feature_schedules["occ"]
    hvac_matrix = feature_schedules["hvac"]
    shade_params = _extract_shade_params_from_meta(hbjson_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    npz_label = export_dir / f"{model_name}_label.npz"
    np.savez_compressed(
        npz_label,
        air=air_matrix,
        op=op_matrix,
        zones=air_zones,
        label_schema_version=np.asarray(2, dtype=np.int16),
        geom_features_default_model_input=np.asarray(0, dtype=np.uint8),
        centroids=centroids,
        geom_features=geom_features,
        ach_matrix=ach_matrix,
        equip_matrix=equip_matrix,
        occ_matrix=occ_matrix,
        hvac_matrix=hvac_matrix,
        shade_depth_NSEW=shade_params["shade_depth_NSEW"],
        head_height=np.float32(shade_params["head_height"]),
        sill_height=np.float32(shade_params["sill_height"]),
        shade_gap=np.float32(shade_params["shade_gap"]),
        contact_matrix=contact_matrix,
    )
    logger.info(f"  [Save] Label NPZ -> {npz_label}")
    logger.info(f"  Matrices  : {air_matrix.shape} (Air), {op_matrix.shape} (Op)")
    logger.info(f"  Anchors   : {centroids.shape} (Centroids)")
    logger.info(
        f"  Features  : {geom_features.shape} (13D: Vol/FloorA/GndFloor/ExtWall_NSEW/ExtWin_NSEW/Roof/ZoneSVF)"
    )
    logger.info(f"  ACH       : {ach_matrix.shape}")
    logger.info(
        f"  Equip     : {equip_matrix.shape}  non-zero zones: {int(np.any(equip_matrix != 0, axis=0).sum())}"
    )
    logger.info(
        f"  Occ       : {occ_matrix.shape}    non-zero zones: {int(np.any(occ_matrix != 0, axis=0).sum())}"
    )
    logger.info(
        f"  HVAC      : {hvac_matrix.shape}   non-zero zones: {int(np.any(hvac_matrix != 0, axis=0).sum())}"
    )
