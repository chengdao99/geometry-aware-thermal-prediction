"""Export physical voxel arrays. Inputs: Points and incidence metadata. Outputs: NPZ and explicitly requested auxiliary formats."""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)
_PLY_FLOAT_PROPS = [
    "x",
    "y",
    "z",
    "thickness",
    "lambda",
    "u_value",
    "cp_rho",
    "solar_abs",
    "transmittance",
    "emissivity",
    "bc_code",
    "svf",
    "fill_rate",
]
_PLY_INT_PROPS = ["tag", "face_id"]
_PLY_FLOAT_COL_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]
_PLY_INT_COL_IDX = [12, 13]
_PLY_DTYPE = np.dtype(
    [(name, "<f4") for name in _PLY_FLOAT_PROPS] + [(name, "<i4") for name in _PLY_INT_PROPS]
)
_CSV_HEADER = "x,y,z,thickness,lambda,u_value,cp_rho,solar_abs,transmittance,emissivity,bc_code,svf,tag,face_id,fill_rate"
_CSV_FMT = [
    "%.4f",
    "%.4f",
    "%.4f",
    "%.4f",
    "%.4f",
    "%.4f",
    "%.1f",
    "%.3f",
    "%.3f",
    "%.3f",
    "%d",
    "%.4f",
    "%d",
    "%d",
    "%.4f",
]


def _validate(voxel_data: np.ndarray) -> bool:
    if voxel_data is None or len(voxel_data) == 0:
        logger.warning("No voxel data to export.")
        return False
    if voxel_data.ndim != 2 or voxel_data.shape[1] != 15:
        logger.error(
            f"[Voxel Exporter] Expected (V, 15) matrix,Actual shape is {voxel_data.shape}。Please ensure that the data was generated using hbjson_to_voxel.voxelize_from_faces()."
        )
        return False
    return True


def _ensure_dir(output_dir: str) -> bool:
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            logger.error(f"[Voxel Exporter] Unable to create output directory {output_dir}: {e}")
            return False
    return True


def _build_ply_header(num_points: int) -> bytes:
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {num_points}"]
    for name in _PLY_FLOAT_PROPS:
        lines.append(f"property float {name}")
    for name in _PLY_INT_PROPS:
        lines.append(f"property int {name}")
    lines.append("end_header")
    lines.append("")
    return "\n".join(lines).encode("ascii")


def _export_npz(
    voxel_data: np.ndarray, path: str, zone_voxel_map: "dict[str, np.ndarray] | None" = None
) -> None:
    payload = {"points": voxel_data.astype(np.float32)}
    if zone_voxel_map:
        payload.update(zone_voxel_map)
    np.savez_compressed(path, **payload)
    logger.info(f"  - [OK] NPZ: {path}  ({len(voxel_data):,} voxels)")


def _export_ply_binary(voxel_data: np.ndarray, path: str) -> None:
    num_points = len(voxel_data)
    structured = np.empty(num_points, dtype=_PLY_DTYPE)
    for ply_col, data_col in zip(_PLY_FLOAT_PROPS, _PLY_FLOAT_COL_IDX):
        structured[ply_col] = voxel_data[:, data_col].astype(np.float32)
    for ply_col, data_col in zip(_PLY_INT_PROPS, _PLY_INT_COL_IDX):
        structured[ply_col] = voxel_data[:, data_col].astype(np.int32)
    with open(path, "wb") as f:
        f.write(_build_ply_header(num_points))
        f.write(structured.tobytes())
    logger.info(f"  - [OK] PLY (binary): {path}  ({num_points:,} voxels)")


def _export_csv(voxel_data: np.ndarray, path: str) -> None:
    np.savetxt(path, voxel_data, delimiter=",", header=_CSV_HEADER, fmt=_CSV_FMT, comments="")
    logger.info(f"  - [OK] CSV: {path}  ({len(voxel_data):,} voxels)")


def export_voxels(
    voxel_data: np.ndarray,
    base_name: str,
    output_dir: str = "results",
    formats: list | None = None,
    zone_voxel_map: "dict[str, np.ndarray] | None" = None,
) -> None:
    """Export voxel points and zone-to-voxel metadata in the requested formats."""
    if formats is None:
        formats = ["npz", "ply"]
    if not _validate(voxel_data):
        return
    if not _ensure_dir(output_dir):
        return
    logger.info(
        f"[Voxel Exporter] Export to: {output_dir}  (format={formats}, voxels={len(voxel_data):,})"
    )
    if "npz" in formats:
        path = os.path.join(output_dir, f"{base_name}.npz")
        try:
            _export_npz(voxel_data, path, zone_voxel_map)
        except Exception as e:
            logger.error(f"  - [Fail] NPZ: {e}")
    if "ply" in formats:
        path = os.path.join(output_dir, f"{base_name}_voxels.ply")
        try:
            _export_ply_binary(voxel_data, path)
        except Exception as e:
            logger.error(f"  - [Fail] PLY: {e}")
    if "csv" in formats:
        path = os.path.join(output_dir, f"{base_name}.csv")
        try:
            _export_csv(voxel_data, path)
        except Exception as e:
            logger.error(f"  - [Fail] CSV: {e}")
