"""Repair HBJSON face adjacency. Inputs: HBJSON. Outputs: A repaired working copy and count."""

import json
import uuid
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def _do_fix(data: dict) -> tuple[dict, int]:
    rooms = data.get("rooms", [])
    face_map = defaultdict(list)
    for room in rooms:
        for face in room.get("faces", []):
            face_map[face["identifier"]].append((room, face))
    duplicates = {k: v for k, v in face_map.items() if len(v) > 1}
    fixed_count = 0
    for orig_id, occurrences in duplicates.items():
        if len(occurrences) != 2:
            logger.warning(
                f"  [WARN] Face '{orig_id}' has {len(occurrences)} occurrences (expected 2). Skipping."
            )
            continue
        room_A, face_A = occurrences[0]
        room_B, face_B = occurrences[1]
        new_id = f"{orig_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"  [Fix]  {orig_id}")
        logger.info(f"           Room A ({room_A['identifier']}): keeps '{orig_id}'")
        logger.info(f"           Room B ({room_B['identifier']}): renamed -> '{new_id}'")
        face_B["identifier"] = new_id
        bc_A = face_A.get("boundary_condition", {})
        if isinstance(bc_A, dict) and "boundary_condition_objects" in bc_A:
            if bc_A["boundary_condition_objects"][0] == orig_id:
                bc_A["boundary_condition_objects"][0] = new_id
        bc_B = face_B.get("boundary_condition", {})
        if isinstance(bc_B, dict) and "boundary_condition_objects" in bc_B:
            if bc_B["boundary_condition_objects"][0] == orig_id:
                bc_B["boundary_condition_objects"][0] = orig_id
        fixed_count += 1
    seen_shade_ids: set = set()
    shade_dedup_count = 0
    for shade in data.get("orphaned_shades", []):
        seen_shade_ids.add(shade["identifier"])
    shades_to_promote: list = []
    for room in rooms:
        for shade_key in ("outdoor_shades", "indoor_shades"):
            original_shades = room.get(shade_key, [])
            if not original_shades:
                continue
            kept, promoted = ([], [])
            for shade in original_shades:
                sid = shade["identifier"]
                if sid not in seen_shade_ids:
                    seen_shade_ids.add(sid)
                    kept.append(shade)
                    shades_to_promote.append(shade)
                else:
                    promoted.append(sid)
                    shade_dedup_count += 1
            if promoted:
                logger.info(
                    f"  [ShadeFix] Room '{room['identifier']}' — removed {len(promoted)} duplicate shade(s): {promoted}"
                )
            room[shade_key] = kept
    if shade_dedup_count > 0:
        existing_orphan_ids = {s["identifier"] for s in data.get("orphaned_shades", [])}
        for room in rooms:
            for shade_key in ("outdoor_shades", "indoor_shades"):
                room[shade_key] = [
                    s for s in room.get(shade_key, []) if s["identifier"] not in existing_orphan_ids
                ]
        logger.info(
            f"  [ShadeFix] Shade deduplication complete. Total duplicates removed: {shade_dedup_count}"
        )
        fixed_count += shade_dedup_count
    return (data, fixed_count)


def fix_adjacency_and_return(
    hbjson_path: Path, save_fixed: bool = True, output_dir: Path = None
) -> tuple[Path, int]:
    """Repair adjacent face identifiers and return the resulting path and repair count."""
    with open(hbjson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data, fixed_count = _do_fix(data)
    if fixed_count == 0:
        return (hbjson_path, 0)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{hbjson_path.stem}_fixed{hbjson_path.suffix}"
    else:
        out_path = (
            hbjson_path.parent / f"{hbjson_path.stem}_fixed{hbjson_path.suffix}"
            if save_fixed
            else hbjson_path
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return (out_path, fixed_count)
