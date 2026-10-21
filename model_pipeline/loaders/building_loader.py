"""Load paired building physics and temporal drivers. Inputs: Label and voxel NPZ, EPW, and training normalization. Outputs: Validated sparse-model batches."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset

EPW_COLS = (6, 8, 13, 14, 15, 21)
SUMMER_START_H = 2880
SUMMER_END_H = 6552
CONTINUOUS_PHYS_COLS = (3, 4, 5, 6, 7, 8, 9, 11)
BC_COL = 10
TAG_COL = 12
N_BC_CLASSES = 4
N_TAG_CLASSES = 8
EPW_FEATURE_DIM = 13
WEATHER_FEATURE_DIM = 17
NEIGHBOR_FEATURE_DIM = 4
VOXEL_FEATURE_DIM = len(CONTINUOUS_PHYS_COLS) + N_BC_CLASSES + N_TAG_CLASSES
VOXEL_PHYSICAL_BOUNDARY_DIM = len(CONTINUOUS_PHYS_COLS) + N_BC_CLASSES


@dataclass(frozen=True)
class VoxelFeatureConfig:
    """Represent voxel feature config and its associated geometry or physical configuration."""

    use_element_onehot: bool
    input_dim: int
    mode: str


def get_voxel_feature_config(use_element_onehot: bool = False) -> VoxelFeatureConfig:
    """Describe physical/boundary channels and optional element encoding. Public training and inference require the 12D configuration without element one-hot channels."""
    use_element_onehot = bool(use_element_onehot)
    if use_element_onehot:
        return VoxelFeatureConfig(True, VOXEL_FEATURE_DIM, "physical + boundary + element")
    return VoxelFeatureConfig(False, VOXEL_PHYSICAL_BOUNDARY_DIM, "physical + boundary")


def resolve_voxel_feature_config(config: Mapping[str, object] | None) -> VoxelFeatureConfig:
    """Compute resolve voxel feature config for the current building or model state. Inputs: config."""
    voxel_config = {} if config is None else config.get("voxel_features", {})
    if voxel_config is None:
        voxel_config = {}
    if not isinstance(voxel_config, Mapping):
        raise TypeError("voxel_features config must be a mapping")
    return get_voxel_feature_config(voxel_config.get("use_element_onehot", False))


REQUIRED_ZONE_VOXEL_KEYS = (
    "points",
    "zone_voxel_schema_version",
    "zone_ids",
    "zone_ptr",
    "zone_voxel_indices",
    "zone_voxel_roles",
    "voxel_zone_counts",
    "voxel_size_m",
    "coordinate_scale",
    "coordinate_offset_m",
)
WEATHER_FEATURE_NAMES = (
    "dry_bulb",
    "relative_humidity",
    "global_horizontal_irradiance",
    "direct_normal_irradiance",
    "diffuse_horizontal_irradiance",
    "wind_speed",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
    "sin_solar_azimuth",
    "cos_solar_azimuth",
    "sin_solar_elevation",
    "air_change_rate",
    "equipment_gain",
    "occupancy_density",
    "hvac_heat_flux",
)
NEIGHBOR_FEATURE_NAMES = (
    "air_change_rate",
    "equipment_gain",
    "occupancy_density",
    "hvac_heat_flux",
)


def _decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values.tolist()
    ]


def _summer_slice(length: int) -> slice:
    start = min(SUMMER_START_H, max(length - 1, 0))
    end = min(SUMMER_END_H + 1, length)
    return slice(start, end)


def _as_integer_codes(values: np.ndarray, minimum: int, maximum: int, name: str) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    rounded = np.rint(values).astype(np.int64)
    if not np.allclose(values, rounded, atol=1e-05):
        raise ValueError(f"{name} must contain integer category codes")
    if rounded.size and (rounded.min() < minimum or rounded.max() > maximum):
        raise ValueError(
            f"{name} code range [{rounded.min()}, {rounded.max()}] is outside [{minimum}, {maximum}]"
        )
    return rounded


def _one_hot(codes: np.ndarray, class_count: int, offset: int = 0) -> np.ndarray:
    result = np.zeros((len(codes), class_count), dtype=np.float32)
    if len(codes):
        result[np.arange(len(codes)), codes - offset] = 1.0
    return result


def parse_epw(epw_path: str | Path) -> np.ndarray:
    """Read EPW weather and derive calendar and solar-position features on the source time axis."""
    import pandas as pd
    import pvlib

    epw_path = Path(epw_path)
    with epw_path.open("r", encoding="utf-8", errors="ignore") as stream:
        location = stream.readline().strip().split(",")
    if len(location) < 9:
        raise ValueError(f"Invalid EPW LOCATION header: {epw_path}")
    latitude = float(location[6])
    longitude = float(location[7])
    utc_offset = float(location[8])
    rows: list[list[float]] = []
    with epw_path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line_number, line in enumerate(stream):
            if line_number < 8:
                continue
            parts = line.strip().split(",")
            if len(parts) < 22:
                continue
            rows.append([float(parts[col]) if parts[col] else 0.0 for col in EPW_COLS])
    climate = np.asarray(rows, dtype=np.float32)
    if climate.ndim != 2 or climate.shape[1] != len(EPW_COLS):
        raise ValueError(f"No valid hourly weather records in {epw_path}")
    hours = np.arange(len(climate), dtype=np.float32)
    hour_of_day = hours % 24.0
    day_of_year = hours / 24.0
    calendar = np.stack(
        [
            np.sin(2.0 * np.pi * hour_of_day / 24.0),
            np.cos(2.0 * np.pi * hour_of_day / 24.0),
            np.sin(2.0 * np.pi * day_of_year / 365.0),
            np.cos(2.0 * np.pi * day_of_year / 365.0),
        ],
        axis=1,
    ).astype(np.float32)
    sign = "+" if utc_offset >= 0 else "-"
    abs_offset = abs(utc_offset)
    offset_hour = int(abs_offset)
    offset_minute = int(round((abs_offset - offset_hour) * 60.0))
    timezone = (
        f"Etc/GMT{-int(utc_offset):+d}"
        if offset_minute == 0
        else f"{sign}{offset_hour:02d}:{offset_minute:02d}"
    )
    timestamps = pd.date_range("2001-01-01 00:30", periods=len(climate), freq="1h", tz=timezone)
    solar_position = pvlib.solarposition.get_solarposition(
        time=timestamps, latitude=latitude, longitude=longitude, method="nrel_numpy"
    )
    azimuth = np.radians(solar_position["azimuth"].to_numpy(dtype=np.float32))
    elevation = np.radians(solar_position["apparent_elevation"].to_numpy(dtype=np.float32))
    solar = np.stack([np.sin(azimuth), np.cos(azimuth), np.sin(elevation)], axis=1).astype(
        np.float32
    )
    return np.concatenate([climate, calendar, solar], axis=1)


class NormStats:
    """Fit and apply training-set normalization for temperatures, weather, physics, and drivers."""

    def __init__(self) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        self.temp_mean: float | None = None
        self.temp_std: float | None = None
        self.epw_mean: np.ndarray | None = None
        self.epw_std: np.ndarray | None = None
        self.phys_mean: np.ndarray | None = None
        self.phys_std: np.ndarray | None = None
        self.ach_mean: float | None = None
        self.ach_std: float | None = None
        self.equip_mean: float | None = None
        self.equip_std: float | None = None
        self.occ_mean: float | None = None
        self.occ_std: float | None = None
        self.hvac_mean: float | None = None
        self.hvac_std: float | None = None

    @staticmethod
    def _collect_label_feature(label_paths: Sequence[Path], key: str) -> np.ndarray:
        rows: list[np.ndarray] = []
        for path in label_paths:
            with np.load(path, allow_pickle=False) as label:
                if key not in label.files:
                    continue
                values = label[key].astype(np.float32)
                rows.append(values[_summer_slice(values.shape[0])].reshape(-1))
        return np.concatenate(rows) if rows else np.zeros(1, dtype=np.float32)

    def fit(self, label_paths: List[Path], voxel_paths: List[Path], epw_full: np.ndarray) -> None:
        """Fit training-only statistics in place. Temperature statistics use summer air and operative temperatures; weather normalization uses six physical EPW channels; voxel normalization uses eight continuous material channels."""
        if not label_paths or not voxel_paths:
            raise ValueError("NormStats.fit requires non-empty label and voxel path lists")
        if epw_full.ndim != 2 or epw_full.shape[1] != EPW_FEATURE_DIM:
            raise ValueError(f"epw_full must have shape (T, {EPW_FEATURE_DIM})")
        temperatures: list[np.ndarray] = []
        for path in label_paths:
            with np.load(path, allow_pickle=False) as label:
                if "op" not in label.files:
                    raise KeyError(f"{path.name} has no 'op' temperature matrix")
                op = label["op"].astype(np.float32)
                air = label["air"].astype(np.float32) if "air" in label.files else op
                time_slice = _summer_slice(op.shape[0])
                temperatures.extend([air[time_slice].reshape(-1), op[time_slice].reshape(-1)])
        temperature_values = np.concatenate(temperatures)
        self.temp_mean = float(temperature_values.mean())
        self.temp_std = max(float(temperature_values.std()), 1e-06)
        epw_summer = epw_full[_summer_slice(epw_full.shape[0]), :6]
        self.epw_mean = epw_summer.mean(axis=0).astype(np.float32)
        self.epw_std = np.maximum(epw_summer.std(axis=0), 1e-06).astype(np.float32)
        phys_rows: list[np.ndarray] = []
        for path in voxel_paths:
            with np.load(path, allow_pickle=False) as voxel_file:
                if "points" not in voxel_file.files:
                    raise KeyError(f"{path.name} is missing the required voxel 'points' key")
                points = voxel_file["points"].astype(np.float32)
                if points.ndim != 2 or points.shape[1] <= max(CONTINUOUS_PHYS_COLS):
                    raise ValueError(f"Invalid points matrix in {path.name}: {points.shape}")
                if len(points):
                    phys_rows.append(points[:, CONTINUOUS_PHYS_COLS])
        if not phys_rows:
            raise ValueError("No voxel points were available while fitting physical statistics")
        physical_values = np.concatenate(phys_rows, axis=0)
        self.phys_mean = physical_values.mean(axis=0).astype(np.float32)
        self.phys_std = np.maximum(physical_values.std(axis=0), 1e-06).astype(np.float32)
        ach = self._collect_label_feature(label_paths, "ach_matrix")
        equip = self._collect_label_feature(label_paths, "equip_matrix")
        occupancy = self._collect_label_feature(label_paths, "occ_matrix")
        hvac = self._collect_label_feature(label_paths, "hvac_matrix")
        self.ach_mean, self.ach_std = (float(ach.mean()), max(float(ach.std()), 1e-06))
        self.equip_mean, self.equip_std = (float(equip.mean()), max(float(equip.std()), 1e-06))
        self.occ_mean, self.occ_std = (float(occupancy.mean()), max(float(occupancy.std()), 1e-06))
        self.hvac_mean, self.hvac_std = (float(hvac.mean()), max(float(hvac.std()), 1e-06))
        print(f"[NormStats] fitted on {len(label_paths)} buildings")
        print(f"  temperature: mean={self.temp_mean:.3f}, std={self.temp_std:.3f}")
        print(f"  ACH:         mean={self.ach_mean:.4f}, std={self.ach_std:.4f}")
        print(f"  voxel continuous channels: {len(CONTINUOUS_PHYS_COLS)}")

    def norm_temp(self, values: np.ndarray) -> np.ndarray:
        """Return float32 temperatures standardized with the fitted indoor mean and standard deviation."""
        return ((values - self.temp_mean) / self.temp_std).astype(np.float32)

    def norm_epw(self, epw: np.ndarray) -> np.ndarray:
        """Standardize the six physical weather channels and retain calendar and solar-position features unchanged."""
        result = epw.copy().astype(np.float32)
        result[:, :6] = (result[:, :6] - self.epw_mean) / self.epw_std
        return result

    def norm_phys(self, values: np.ndarray) -> np.ndarray:
        """Standardize the eight continuous physical voxel channels using training statistics."""
        return ((values - self.phys_mean) / self.phys_std).astype(np.float32)

    def norm_ach(self, values: np.ndarray) -> np.ndarray:
        """Compute norm ach for the current building or model state. Inputs: values."""
        return ((values - self.ach_mean) / self.ach_std).astype(np.float32)

    def norm_equip(self, values: np.ndarray) -> np.ndarray:
        """Compute norm equip for the current building or model state. Inputs: values."""
        return ((values - self.equip_mean) / self.equip_std).astype(np.float32)

    def norm_occ(self, values: np.ndarray) -> np.ndarray:
        """Compute norm occ for the current building or model state. Inputs: values."""
        return ((values - self.occ_mean) / self.occ_std).astype(np.float32)

    def norm_hvac(self, values: np.ndarray) -> np.ndarray:
        """Compute norm hvac for the current building or model state. Inputs: values."""
        return ((values - self.hvac_mean) / self.hvac_std).astype(np.float32)


class BuildingCache:
    """Validate paired label and voxel schemas and cache normalized building inputs."""

    def __init__(
        self, label_path: Path, voxel_path: Path, norm: NormStats, epw_raw: np.ndarray | None = None
    ) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        del epw_raw
        label_path = Path(label_path)
        voxel_path = Path(voxel_path)
        self.bld_name = label_path.stem.removesuffix("_label")
        with np.load(label_path, allow_pickle=False) as label:
            required = {"op", "centroids", "zones", "geom_features", "ach_matrix"}
            missing = sorted(required.difference(label.files))
            if missing:
                raise KeyError(f"{label_path.name} is missing label keys: {missing}")
            op = label["op"].astype(np.float32)
            self.zone_names = _decode_strings(label["zones"])
            self.Z = len(self.zone_names)
            if op.ndim != 2 or op.shape[1] != self.Z:
                raise ValueError(
                    f"{label_path.name}: op shape {op.shape} disagrees with {self.Z} zones"
                )
            self.op_norm = norm.norm_temp(op)
            self.centroids_m = label["centroids"].astype(np.float32)
            if self.centroids_m.shape != (self.Z, 3):
                raise ValueError(
                    f"{label_path.name}: invalid centroids shape {self.centroids_m.shape}"
                )
            self.geom_metadata = label["geom_features"].astype(np.float32)
            if self.geom_metadata.ndim != 2 or self.geom_metadata.shape[0] != self.Z:
                raise ValueError(f"{label_path.name}: geom_features zone count mismatch")
            self.zone_volumes_m3 = np.maximum(self.geom_metadata[:, 0], 0.0).astype(np.float32)
            if "contact_matrix" in label.files:
                self.contact_matrix_raw = label["contact_matrix"].astype(np.float32)
                if self.contact_matrix_raw.shape != (self.Z, self.Z):
                    raise ValueError(f"{label_path.name}: invalid contact_matrix shape")
            else:
                self.contact_matrix_raw = np.zeros((self.Z, self.Z), dtype=np.float32)

            def load_time_matrix(key: str) -> np.ndarray:
                """Compute load time matrix for the current building or model state. Inputs: key."""
                if key in label.files:
                    matrix = label[key].astype(np.float32)
                else:
                    matrix = np.zeros_like(op, dtype=np.float32)
                if matrix.shape != op.shape:
                    raise ValueError(
                        f"{label_path.name}: {key} shape {matrix.shape} != op shape {op.shape}"
                    )
                return matrix

            self.ach_matrix_norm = norm.norm_ach(load_time_matrix("ach_matrix"))
            self.equip_matrix_norm = norm.norm_equip(load_time_matrix("equip_matrix"))
            self.occ_matrix_norm = norm.norm_occ(load_time_matrix("occ_matrix"))
            self.hvac_matrix_norm = norm.norm_hvac(load_time_matrix("hvac_matrix"))
        self._load_voxels(voxel_path, norm)

    def _load_voxels(self, voxel_path: Path, norm: NormStats) -> None:
        """Validate physical voxels and CSR, align zone names to labels, transform centroids into the voxel coordinate frame, and cache sparse-grid inputs."""
        with np.load(voxel_path, allow_pickle=False) as voxel_file:
            missing = sorted(set(REQUIRED_ZONE_VOXEL_KEYS).difference(voxel_file.files))
            if missing:
                raise KeyError(
                    f"{voxel_path.name} has no Zone-Voxel CSR schema; missing keys: {missing}. Regenerate it with data_pipeline using Zone-Voxel CSR metadata."
                )
            schema_version = int(np.asarray(voxel_file["zone_voxel_schema_version"]).reshape(-1)[0])
            if schema_version < 1:
                raise ValueError(f"{voxel_path.name}: unsupported schema version {schema_version}")
            points = voxel_file["points"].astype(np.float32)
            voxel_zone_names = _decode_strings(voxel_file["zone_ids"])
            source_ptr = voxel_file["zone_ptr"].astype(np.int64)
            source_indices = voxel_file["zone_voxel_indices"].astype(np.int64)
            source_roles = voxel_file["zone_voxel_roles"].astype(np.uint8)
            self.voxel_size_m = float(np.asarray(voxel_file["voxel_size_m"]).reshape(-1)[0])
            stored_counts = voxel_file["voxel_zone_counts"].astype(np.int64)
            coordinate_scale = float(np.asarray(voxel_file["coordinate_scale"]).reshape(-1)[0])
            coordinate_offset_m = voxel_file["coordinate_offset_m"].astype(np.float32)
        if points.ndim != 2 or points.shape[1] < 15:
            raise ValueError(
                f"{voxel_path.name}: points must have shape (V, >=15), got {points.shape}"
            )
        if len(points) == 0:
            raise ValueError(f"{voxel_path.name}: empty voxel point matrix")
        if not np.isfinite(points).all():
            raise ValueError(f"{voxel_path.name}: points contain NaN or infinite values")
        if not np.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0.0:
            raise ValueError(f"{voxel_path.name}: invalid voxel_size_m={self.voxel_size_m}")
        if not np.isfinite(coordinate_scale) or coordinate_scale <= 0.0:
            raise ValueError(f"{voxel_path.name}: invalid coordinate_scale")
        if coordinate_offset_m.shape != (3,) or not np.isfinite(coordinate_offset_m).all():
            raise ValueError(f"{voxel_path.name}: invalid coordinate_offset_m")
        self.centroids_m = (self.centroids_m * coordinate_scale + coordinate_offset_m).astype(
            np.float32
        )
        if coordinate_scale != 1.0:
            self.geom_metadata = self.geom_metadata.copy()
            self.geom_metadata[:, 0] *= coordinate_scale**3
            self.geom_metadata[:, 1:12] *= coordinate_scale**2
            self.contact_matrix_raw *= coordinate_scale**2
            self.zone_volumes_m3 = np.maximum(self.geom_metadata[:, 0], 0.0).astype(np.float32)
        self._validate_source_csr(
            voxel_path,
            voxel_zone_names,
            source_ptr,
            source_indices,
            source_roles,
            len(points),
            stored_counts,
        )
        self.zone_voxel_ptr, self.zone_voxel_indices, self.zone_voxel_roles = self._align_zone_csr(
            voxel_path, voxel_zone_names, source_ptr, source_indices, source_roles
        )
        continuous = norm.norm_phys(points[:, CONTINUOUS_PHYS_COLS])
        bc_codes = _as_integer_codes(points[:, BC_COL], 0, N_BC_CLASSES - 1, "bc_code")
        tag_codes = _as_integer_codes(points[:, TAG_COL], 1, N_TAG_CLASSES, "tag")
        self.voxel_features = np.concatenate(
            [
                continuous,
                _one_hot(bc_codes, N_BC_CLASSES),
                _one_hot(tag_codes, N_TAG_CLASSES, offset=1),
            ],
            axis=1,
        ).astype(np.float32)
        if self.voxel_features.shape[1] != VOXEL_FEATURE_DIM:
            raise AssertionError("Unexpected voxel feature dimension")
        self.voxel_coords_m = points[:, :3].astype(np.float32)
        raw_grid = self.voxel_coords_m / self.voxel_size_m - 0.5
        raw_ijk = np.rint(raw_grid).astype(np.int64)
        reconstructed = (raw_ijk.astype(np.float32) + 0.5) * self.voxel_size_m
        tolerance = max(0.0001, self.voxel_size_m * 0.001)
        max_error = float(np.max(np.abs(reconstructed - self.voxel_coords_m)))
        if max_error > tolerance:
            raise ValueError(
                f"{voxel_path.name}: voxel centers are not aligned to voxel_size_m (max error {max_error:.6g} m)"
            )
        ijk_min = raw_ijk.min(axis=0)
        shifted = raw_ijk - ijk_min
        if np.any(shifted > np.iinfo(np.int32).max):
            raise ValueError(f"{voxel_path.name}: sparse grid extent exceeds int32")
        self.voxel_ijk = shifted.astype(np.int32)
        self.grid_origin_m = ijk_min.astype(np.float32) * self.voxel_size_m
        self.spatial_shape = (self.voxel_ijk.max(axis=0) + 1).astype(np.int64).tolist()

    @staticmethod
    def _validate_source_csr(
        voxel_path: Path,
        zone_names: Sequence[str],
        ptr: np.ndarray,
        indices: np.ndarray,
        roles: np.ndarray,
        voxel_count: int,
        stored_counts: np.ndarray | None,
    ) -> None:
        if len(set((name.casefold() for name in zone_names))) != len(zone_names):
            raise ValueError(f"{voxel_path.name}: duplicate zone_ids")
        if ptr.ndim != 1 or len(ptr) != len(zone_names) + 1:
            raise ValueError(f"{voxel_path.name}: invalid zone_ptr length")
        if len(ptr) == 0 or ptr[0] != 0 or np.any(ptr[1:] < ptr[:-1]):
            raise ValueError(f"{voxel_path.name}: zone_ptr must start at zero and be monotonic")
        if ptr[-1] != len(indices) or len(indices) != len(roles):
            raise ValueError(f"{voxel_path.name}: CSR edge array lengths disagree")
        if len(indices) and (indices.min() < 0 or indices.max() >= voxel_count):
            raise ValueError(f"{voxel_path.name}: zone_voxel_indices are out of range")
        if len(roles) and (roles.min() < 1 or roles.max() > 6):
            raise ValueError(f"{voxel_path.name}: unknown zone_voxel_roles")
        if stored_counts is not None:
            if stored_counts.shape != (voxel_count,):
                raise ValueError(f"{voxel_path.name}: invalid voxel_zone_counts shape")
            calculated = np.bincount(indices, minlength=voxel_count)
            if not np.array_equal(calculated, stored_counts):
                raise ValueError(f"{voxel_path.name}: voxel_zone_counts disagrees with CSR")

    def _align_zone_csr(
        self,
        voxel_path: Path,
        source_zone_names: Sequence[str],
        source_ptr: np.ndarray,
        source_indices: np.ndarray,
        source_roles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source_lookup = {name.casefold(): index for index, name in enumerate(source_zone_names)}
        missing = [name for name in self.zone_names if name.casefold() not in source_lookup]
        if missing:
            raise ValueError(
                f"{voxel_path.name}: zones from label are absent in Zone-Voxel mapping: {missing}"
            )
        ptr = [0]
        index_chunks: list[np.ndarray] = []
        role_chunks: list[np.ndarray] = []
        empty: list[str] = []
        for zone_name in self.zone_names:
            source_index = source_lookup[zone_name.casefold()]
            start = int(source_ptr[source_index])
            end = int(source_ptr[source_index + 1])
            if start == end:
                empty.append(zone_name)
            index_chunks.append(source_indices[start:end])
            role_chunks.append(source_roles[start:end])
            ptr.append(ptr[-1] + end - start)
        if empty:
            raise ValueError(
                f"{voxel_path.name}: zones have no claimed voxels after activation thresholding: {empty}"
            )
        return (
            np.asarray(ptr, dtype=np.int64),
            np.concatenate(index_chunks).astype(np.int64, copy=False),
            np.concatenate(role_chunks).astype(np.uint8, copy=False),
        )


class SegLenBatchSampler:
    """Give each training batch a common randomly selected forecast length."""

    def __init__(
        self, data_source_len: int, batch_size: int, min_len: int, max_len: int, seed: int = 42
    ) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        if data_source_len < 0 or batch_size <= 0 or min_len <= 0 or (max_len < min_len):
            raise ValueError("Invalid SegLenBatchSampler arguments")
        self.data_source_len = data_source_len
        self.batch_size = batch_size
        self.min_len = min_len
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(data_source_len)

    def __iter__(self):
        self.rng.shuffle(self.indices)
        for start in range(0, len(self.indices), self.batch_size):
            raw_indices = self.indices[start : start + self.batch_size]
            seg_len = int(self.rng.integers(self.min_len, self.max_len + 1))
            yield [(int(index), seg_len) for index in raw_indices]

    def __len__(self) -> int:
        return (self.data_source_len + self.batch_size - 1) // self.batch_size


class End2EndDataset(Dataset):
    """Sample temporal segments, target zones, and known-neighbor operating conditions."""

    def __init__(
        self,
        buildings: List[BuildingCache],
        epw_norm: np.ndarray,
        segs_per_building: int = 10,
        seed: int = 42,
        min_seg_len: int = 72,
        max_seg_len: int = 240,
        context_len: int = 72,
        cold_start_prob: float = 0.5,
        max_zones: int | None = 10,
        zone_sample_seed: int | None = None,
        fixed: bool = False,
        nb_k: int = 15,
        nb_dropout_max: int = 5,
        nb_dropout_min_valid: int = 1,
        enable_neighbor_dropout: bool | None = None,
        use_element_onehot: bool = False,
    ) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        if not buildings:
            raise ValueError("End2EndDataset requires at least one building")
        if epw_norm.ndim != 2 or epw_norm.shape[1] != EPW_FEATURE_DIM:
            raise ValueError(f"epw_norm must have shape (T, {EPW_FEATURE_DIM})")
        if context_len <= 0 or min_seg_len <= 0 or max_seg_len < min_seg_len:
            raise ValueError("Invalid context or segment length")
        if not 0.0 <= cold_start_prob <= 1.0:
            raise ValueError("cold_start_prob must be in [0, 1]")
        if nb_k < 0 or nb_dropout_min_valid < 0 or nb_dropout_max < 0:
            raise ValueError("Invalid neighbor configuration")
        self.buildings = buildings
        self.epw_norm = epw_norm.astype(np.float32, copy=False)
        self.context_len = context_len
        self.cold_start_prob = cold_start_prob
        self.max_zones = max_zones
        self.zone_sample_seed = zone_sample_seed
        self.fixed = fixed
        self.min_seg_len = min_seg_len
        self.max_seg_len = max_seg_len
        self.nb_k = nb_k
        self.nb_dropout_max = nb_dropout_max
        self.nb_dropout_min_valid = nb_dropout_min_valid
        self.enable_neighbor_dropout = (
            not fixed if enable_neighbor_dropout is None else bool(enable_neighbor_dropout)
        )
        self.voxel_feature_config = get_voxel_feature_config(use_element_onehot)
        self.use_element_onehot = self.voxel_feature_config.use_element_onehot
        self.voxel_feature_dim = self.voxel_feature_config.input_dim
        available_hours = min(len(epw_norm), *(building.op_norm.shape[0] for building in buildings))
        self.t0_min = SUMMER_START_H + context_len
        self.t0_max = min(SUMMER_END_H, available_hours - 1) - max_seg_len
        if self.t0_max < self.t0_min:
            raise ValueError("The available time series is too short for context_len + max_seg_len")
        if fixed:
            rng = np.random.default_rng(seed)
            self.index: list[tuple[int, int, int, int]] = []
            for building_index in range(len(buildings)):
                for _ in range(segs_per_building):
                    self.index.append(
                        (
                            building_index,
                            int(rng.integers(self.t0_min, self.t0_max + 1)),
                            int(rng.integers(min_seg_len, max_seg_len + 1)),
                            int(rng.integers(0, 2**31)),
                        )
                    )
        else:
            self.index = [
                building_index
                for building_index in range(len(buildings))
                for _ in range(segs_per_building)
            ]

    def __len__(self) -> int:
        return len(self.index)

    def _resolve_index(self, index) -> tuple[BuildingCache, int, int, np.random.Generator]:
        explicit_seg_len = None
        if isinstance(index, tuple):
            raw_index, explicit_seg_len = index
        else:
            raw_index = index
        if self.fixed:
            building_index, t0, stored_seg_len, zone_seed = self.index[int(raw_index)]
            seg_len = int(explicit_seg_len) if explicit_seg_len is not None else stored_seg_len
            rng = np.random.default_rng(zone_seed)
        else:
            building_index = int(self.index[int(raw_index)])
            seg_len = (
                int(explicit_seg_len)
                if explicit_seg_len is not None
                else int(np.random.randint(self.min_seg_len, self.max_seg_len + 1))
            )
            t0 = int(np.random.randint(self.t0_min, self.t0_max + 1))
            rng = np.random.default_rng(self.zone_sample_seed)
        if not self.min_seg_len <= seg_len <= self.max_seg_len:
            raise ValueError(
                f"Requested seg_len={seg_len}, expected [{self.min_seg_len}, {self.max_seg_len}]"
            )
        return (self.buildings[building_index], t0, seg_len, rng)

    def _sample_zones(self, building: BuildingCache, rng: np.random.Generator) -> np.ndarray:
        if self.max_zones is not None and building.Z > self.max_zones:
            return np.sort(rng.choice(building.Z, self.max_zones, replace=False)).astype(np.int64)
        return np.arange(building.Z, dtype=np.int64)

    def _make_neighbors(
        self,
        building: BuildingCache,
        zone_indices: np.ndarray,
        warmup_start: int,
        sequence_end: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select nearest known neighbors and assemble ACH, equipment, occupancy, and HVAC drivers with distance, relative volume, contact strength, and dropout masks. Neighbor temperature truth is excluded."""
        zone_count = len(zone_indices)
        neighbor_indices = np.zeros((zone_count, self.nb_k), dtype=np.int64)
        neighbor_mask = np.ones((zone_count, self.nb_k), dtype=bool)
        if self.nb_k > 0 and building.Z > 1:
            target_centroids = building.centroids_m[zone_indices]
            distance = np.linalg.norm(
                target_centroids[:, None, :] - building.centroids_m[None, :, :], axis=-1
            )
            distance[np.arange(zone_count), zone_indices] = np.inf
            valid_count = min(building.Z - 1, self.nb_k)
            nearest = np.argsort(distance, axis=-1)[:, :valid_count]
            neighbor_indices[:, :valid_count] = nearest
            neighbor_mask[:, :valid_count] = False
            if self.enable_neighbor_dropout and valid_count > self.nb_dropout_min_valid:
                for zone_row in range(zone_count):
                    valid_slots = np.flatnonzero(~neighbor_mask[zone_row])
                    max_drop = min(
                        self.nb_dropout_max, len(valid_slots) - self.nb_dropout_min_valid
                    )
                    if max_drop <= 0:
                        continue
                    drop_count = int(rng.integers(0, max_drop + 1))
                    if drop_count:
                        dropped = rng.choice(valid_slots, drop_count, replace=False)
                        neighbor_mask[zone_row, dropped] = True
        target_centroids = building.centroids_m[zone_indices]
        neighbor_centroids = building.centroids_m[neighbor_indices]
        neighbor_dist2 = np.sum(
            (neighbor_centroids - target_centroids[:, None, :]) ** 2, axis=-1
        ).astype(np.float32)
        absolute_contact = building.contact_matrix_raw[zone_indices[:, None], neighbor_indices]
        absolute_contact = absolute_contact.copy()
        absolute_contact[neighbor_mask] = 0.0
        row_max = (
            absolute_contact.max(axis=-1, keepdims=True)
            if self.nb_k
            else np.zeros((zone_count, 1), dtype=np.float32)
        )
        contact_ratio = np.divide(
            absolute_contact,
            row_max,
            out=np.zeros_like(absolute_contact, dtype=np.float32),
            where=row_max > 1e-06,
        )
        target_volume = building.zone_volumes_m3[zone_indices, None]
        neighbor_volume = building.zone_volumes_m3[neighbor_indices]
        volume_ratio = np.divide(
            neighbor_volume,
            target_volume,
            out=np.zeros_like(neighbor_volume, dtype=np.float32),
            where=target_volume > 1e-06,
        )
        full_neighbor_series = np.stack(
            [
                building.ach_matrix_norm,
                building.equip_matrix_norm,
                building.occ_matrix_norm,
                building.hvac_matrix_norm,
            ],
            axis=-1,
        )
        neighbor_features = np.transpose(
            full_neighbor_series[warmup_start:sequence_end, neighbor_indices, :], (1, 0, 2, 3)
        ).astype(np.float32)
        neighbor_features *= (~neighbor_mask)[:, None, :, None]
        neighbor_dist2[neighbor_mask] = 0.0
        contact_ratio[neighbor_mask] = 0.0
        volume_ratio[neighbor_mask] = 0.0
        return (
            neighbor_features,
            neighbor_dist2,
            neighbor_mask,
            volume_ratio.astype(np.float32),
            contact_ratio.astype(np.float32),
        )

    def __getitem__(self, index) -> dict:
        """Build one temporal/zone sample and its sparse incidence. Cold starts use the normalized weather temperature channel; target tensors are loss-only fields."""
        building, t0, seg_len, rng = self._resolve_index(index)
        zone_indices = self._sample_zones(building, rng)
        zone_count = len(zone_indices)
        warmup_start = t0 - self.context_len
        sequence_end = t0 + seg_len

        def make_zone_weather(start: int, end: int) -> np.ndarray:
            """Compute make zone weather for the current building or model state. Inputs: start, end."""
            epw = np.broadcast_to(
                self.epw_norm[start:end][None, :, :], (zone_count, end - start, EPW_FEATURE_DIM)
            )
            drivers = np.stack(
                [
                    building.ach_matrix_norm[start:end, zone_indices],
                    building.equip_matrix_norm[start:end, zone_indices],
                    building.occ_matrix_norm[start:end, zone_indices],
                    building.hvac_matrix_norm[start:end, zone_indices],
                ],
                axis=-1,
            )
            drivers = np.transpose(drivers, (1, 0, 2))
            weather = np.concatenate([epw, drivers], axis=-1).astype(np.float32)
            if weather.shape[-1] != WEATHER_FEATURE_DIM:
                raise AssertionError("Unexpected weather feature dimension")
            return weather

        weather_warmup = make_zone_weather(warmup_start, t0)
        weather_ar = make_zone_weather(t0, sequence_end)
        if rng.random() < self.cold_start_prob:
            warmup_indoor = weather_warmup[:, :, 0].copy()
        else:
            warmup_indoor = building.op_norm[warmup_start:t0, zone_indices].T.copy()
        target_ar = building.op_norm[t0:sequence_end, zone_indices].T.copy()
        op_norm_t0 = building.op_norm[t0, zone_indices].copy()
        (
            neighbor_features,
            neighbor_dist2,
            neighbor_mask,
            neighbor_volume_ratio,
            neighbor_contact_ratio,
        ) = self._make_neighbors(building, zone_indices, warmup_start, sequence_end, rng)
        selected_ptr = [0]
        incidence_indices: list[np.ndarray] = []
        incidence_roles: list[np.ndarray] = []
        for zone_index in zone_indices:
            start = int(building.zone_voxel_ptr[zone_index])
            end = int(building.zone_voxel_ptr[zone_index + 1])
            incidence_indices.append(building.zone_voxel_indices[start:end])
            incidence_roles.append(building.zone_voxel_roles[start:end])
            selected_ptr.append(selected_ptr[-1] + end - start)
        zone_voxel_indices = np.concatenate(incidence_indices).astype(np.int64, copy=False)
        zone_voxel_roles = np.concatenate(incidence_roles).astype(np.uint8, copy=False)
        sampled_zone_names = [building.zone_names[int(zone_index)] for zone_index in zone_indices]
        voxel_features = building.voxel_features[..., : self.voxel_feature_dim]
        if voxel_features.ndim != 2 or voxel_features.shape[1] != self.voxel_feature_dim:
            raise AssertionError("Unexpected selected voxel feature dimension")
        return {
            "weather_warmup": torch.from_numpy(weather_warmup),
            "warmup_indoor": torch.from_numpy(warmup_indoor),
            "weather_ar": torch.from_numpy(weather_ar),
            "target_ar": torch.from_numpy(target_ar),
            "op_norm_t0": torch.from_numpy(op_norm_t0),
            "voxel_features": torch.from_numpy(voxel_features),
            "voxel_ijk": torch.from_numpy(building.voxel_ijk),
            "voxel_coords_m": torch.from_numpy(building.voxel_coords_m),
            "spatial_shape": building.spatial_shape,
            "grid_origin_m": torch.from_numpy(building.grid_origin_m),
            "voxel_size_m": torch.tensor(building.voxel_size_m, dtype=torch.float32),
            "zone_voxel_ptr": torch.tensor(selected_ptr, dtype=torch.int64),
            "zone_voxel_indices": torch.from_numpy(zone_voxel_indices),
            "zone_voxel_roles": torch.from_numpy(zone_voxel_roles),
            "zone_indices": torch.from_numpy(zone_indices),
            "zone_centroids_m": torch.from_numpy(building.centroids_m[zone_indices]),
            "geom_metadata": torch.from_numpy(building.geom_metadata[zone_indices]),
            "Z": zone_count,
            "nb_warmup": torch.from_numpy(neighbor_features[:, : self.context_len]),
            "nb_ar": torch.from_numpy(neighbor_features[:, self.context_len :]),
            "nb_dist2": torch.from_numpy(neighbor_dist2),
            "nb_mask": torch.from_numpy(neighbor_mask),
            "nb_v_ratio": torch.from_numpy(neighbor_volume_ratio),
            "nb_contact_ratio": torch.from_numpy(neighbor_contact_ratio),
            "bld_name": building.bld_name,
            "t0": t0,
            "sampled_zone_names": sampled_zone_names,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Combine variable-sized building samples into a padded sparse-model batch."""
    if not batch:
        raise ValueError("collate_fn received an empty batch")
    voxel_feature_dims: set[int] = set()
    for item in batch:
        features = item["voxel_features"]
        if features.ndim != 2:
            raise ValueError("voxel_features must have shape (N, C)")
        voxel_feature_dims.add(int(features.shape[1]))
    if len(voxel_feature_dims) != 1:
        raise ValueError("All samples in a batch must use one voxel input dimension")
    segment_lengths = {item["weather_ar"].shape[1] for item in batch}
    if len(segment_lengths) != 1:
        raise ValueError("All samples in a batch must use one seg_len; use SegLenBatchSampler")
    batch_size = len(batch)
    zone_counts = torch.tensor([item["Z"] for item in batch], dtype=torch.int64)
    max_zones = int(zone_counts.max())

    def pad_z(tensor: torch.Tensor, value=0) -> torch.Tensor:
        """Compute pad z for the current building or model state. Inputs: tensor, value."""
        padding = max_zones - tensor.shape[0]
        if padding == 0:
            return tensor
        fill = torch.full(
            (padding, *tensor.shape[1:]), value, dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor, fill], dim=0)

    weather_warmup = torch.stack([pad_z(item["weather_warmup"]) for item in batch], dim=0)
    warmup_indoor = torch.stack([pad_z(item["warmup_indoor"]) for item in batch], dim=0)
    weather_ar = torch.stack([pad_z(item["weather_ar"]) for item in batch], dim=0)
    target_ar = torch.stack([pad_z(item["target_ar"]) for item in batch], dim=0)
    op_norm_t0 = torch.stack([pad_z(item["op_norm_t0"]) for item in batch], dim=0)
    nb_warmup = torch.stack([pad_z(item["nb_warmup"]) for item in batch], dim=0)
    nb_ar = torch.stack([pad_z(item["nb_ar"]) for item in batch], dim=0)
    nb_dist2 = torch.stack([pad_z(item["nb_dist2"]) for item in batch], dim=0)
    nb_mask = torch.stack([pad_z(item["nb_mask"], value=True) for item in batch], dim=0)
    nb_v_ratio = torch.stack([pad_z(item["nb_v_ratio"]) for item in batch], dim=0)
    nb_contact_ratio = torch.stack([pad_z(item["nb_contact_ratio"]) for item in batch], dim=0)
    zone_centroids_m = torch.stack([pad_z(item["zone_centroids_m"]) for item in batch], dim=0)
    geom_metadata = torch.stack([pad_z(item["geom_metadata"]) for item in batch], dim=0)
    zone_mask = torch.arange(max_zones, dtype=torch.int64)[None, :] < zone_counts[:, None]
    voxel_sizes = torch.stack([item["voxel_size_m"] for item in batch])
    if not torch.allclose(voxel_sizes, voxel_sizes[:1], rtol=0.0, atol=1e-06):
        raise ValueError("One sparse CNN batch cannot mix buildings with different voxel_size_m")
    voxel_features: list[torch.Tensor] = []
    voxel_indices: list[torch.Tensor] = []
    voxel_coords_m: list[torch.Tensor] = []
    global_voxel_ptr = [0]
    max_shape = [0, 0, 0]
    for batch_index, item in enumerate(batch):
        count = item["voxel_ijk"].shape[0]
        sparse_index = torch.empty((count, 4), dtype=torch.int32)
        sparse_index[:, 0] = batch_index
        sparse_index[:, 1:] = item["voxel_ijk"].to(torch.int32)
        voxel_indices.append(sparse_index)
        voxel_features.append(item["voxel_features"])
        voxel_coords_m.append(item["voxel_coords_m"])
        global_voxel_ptr.append(global_voxel_ptr[-1] + count)
        max_shape = [
            max(current, int(size)) for current, size in zip(max_shape, item["spatial_shape"])
        ]
    building_zone_ptr = [0]
    combined_zone_voxel_ptr = [0]
    combined_incidence_indices: list[torch.Tensor] = []
    combined_incidence_roles: list[torch.Tensor] = []
    zone_batch_indices: list[torch.Tensor] = []
    zone_local_indices: list[torch.Tensor] = []
    zone_original_indices: list[torch.Tensor] = []
    for batch_index, item in enumerate(batch):
        zone_count = item["Z"]
        local_ptr = item["zone_voxel_ptr"]
        voxel_offset = global_voxel_ptr[batch_index]
        for zone_index in range(zone_count):
            start = int(local_ptr[zone_index])
            end = int(local_ptr[zone_index + 1])
            combined_incidence_indices.append(item["zone_voxel_indices"][start:end] + voxel_offset)
            combined_incidence_roles.append(item["zone_voxel_roles"][start:end])
            combined_zone_voxel_ptr.append(combined_zone_voxel_ptr[-1] + end - start)
        building_zone_ptr.append(building_zone_ptr[-1] + zone_count)
        zone_batch_indices.append(torch.full((zone_count,), batch_index, dtype=torch.int64))
        zone_local_indices.append(torch.arange(zone_count, dtype=torch.int64))
        zone_original_indices.append(item["zone_indices"].to(torch.int64))
    return {
        "weather_warmup": weather_warmup,
        "warmup_indoor": warmup_indoor,
        "weather_ar": weather_ar,
        "target_ar": target_ar,
        "op_norm_t0": op_norm_t0,
        "zone_counts": zone_counts,
        "zone_mask": zone_mask,
        "zone_padding_mask": ~zone_mask,
        "voxel_features": torch.cat(voxel_features, dim=0),
        "voxel_indices": torch.cat(voxel_indices, dim=0),
        "voxel_coords_m": torch.cat(voxel_coords_m, dim=0),
        "spatial_shape": max_shape,
        "batch_size": batch_size,
        "global_voxel_ptr": torch.tensor(global_voxel_ptr, dtype=torch.int64),
        "grid_origins_m": torch.stack([item["grid_origin_m"] for item in batch], dim=0),
        "voxel_sizes_m": voxel_sizes,
        "zone_ptr": torch.tensor(building_zone_ptr, dtype=torch.int64),
        "zone_batch_indices": torch.cat(zone_batch_indices, dim=0),
        "zone_local_indices": torch.cat(zone_local_indices, dim=0),
        "zone_original_indices": torch.cat(zone_original_indices, dim=0),
        "zone_voxel_ptr": torch.tensor(combined_zone_voxel_ptr, dtype=torch.int64),
        "zone_voxel_indices": torch.cat(combined_incidence_indices, dim=0),
        "zone_voxel_roles": torch.cat(combined_incidence_roles, dim=0),
        "zone_centroids_m": zone_centroids_m,
        "geom_metadata": geom_metadata,
        "nb_warmup": nb_warmup,
        "nb_ar": nb_ar,
        "nb_dist2": nb_dist2,
        "nb_mask": nb_mask,
        "nb_v_ratio": nb_v_ratio,
        "nb_contact_ratio": nb_contact_ratio,
        "bld_names": [item["bld_name"] for item in batch],
        "t0s": [item["t0"] for item in batch],
        "sampled_zone_names": [item["sampled_zone_names"] for item in batch],
    }


__all__ = [
    "BuildingCache",
    "End2EndDataset",
    "NEIGHBOR_FEATURE_DIM",
    "VOXEL_PHYSICAL_BOUNDARY_DIM",
    "VoxelFeatureConfig",
    "get_voxel_feature_config",
    "resolve_voxel_feature_config",
    "NormStats",
    "SegLenBatchSampler",
    "VOXEL_FEATURE_DIM",
    "WEATHER_FEATURE_DIM",
    "collate_fn",
    "parse_epw",
]
