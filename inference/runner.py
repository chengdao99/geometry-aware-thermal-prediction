"""Predict one continuous window with MAIN. Inputs: Selected weights, normalization, NPZ pairs, and EPW. Outputs: One readable XLSX per building."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from data_pipeline.spreadsheets import hour_label, outdoor_dry_bulb, write_workbook
from model_pipeline.loaders.building_loader import (
    BuildingCache,
    End2EndDataset,
    collate_fn,
    parse_epw,
)
from model_pipeline.networks.end2end_model import ZoneOverheatPredictor
from model_pipeline.training.data import paired_files
from model_pipeline.training.normalization import load_norm_stats, move_batch_to_device


def forecast_batch(building, weather, start: int, hours: int, model_config: dict, nb_k: int = 15):
    """Build the established cold-start inputs and remove target-only loss tensors before inference."""
    dataset = End2EndDataset(
        [building],
        weather,
        segs_per_building=1,
        seed=2026,
        min_seg_len=hours,
        max_seg_len=hours,
        context_len=int(model_config["context_len"]),
        cold_start_prob=1.0,
        max_zones=None,
        fixed=True,
        nb_k=nb_k,
        nb_dropout_max=0,
        nb_dropout_min_valid=1,
        enable_neighbor_dropout=False,
        use_element_onehot=False,
    )
    dataset.index = [(0, start, hours, 2026)]
    sample = dataset[0]
    if not np.array_equal(
        sample["warmup_indoor"].numpy(), sample["weather_warmup"][:, :, 0].numpy()
    ):
        raise ValueError("Cold-start indoor input differs from the established weather feature.")
    batch = collate_fn([sample])
    batch.pop("target_ar", None)
    batch.pop("op_norm_t0", None)
    return batch, sample["zone_indices"].numpy()


def run_inference(
    checkpoint: Path,
    norm_path: Path,
    run_config: dict,
    voxel_dir: Path,
    label_dir: Path,
    epw: Path,
    output_dir: Path,
    start_hour: int,
    hours: int,
    device_name: str = "cuda",
    building_ids: list[str] | None = None,
) -> list[dict]:
    """Load the selected model strictly and write exactly aligned per-building forecast workbooks."""
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected model runtime.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = dict(saved["model_config"])
    if int(config.get("cnn_in_dim", -1)) != 12 or config.get("quantiles") != [0.1, 0.5, 0.9]:
        raise ValueError(
            "The selected checkpoint does not declare the required 12D, q10/q50/q90 MAIN contract."
        )
    if int(config.get("context_len", 0)) != 72:
        raise ValueError("The selected checkpoint must use the established 72-hour context.")
    norm = load_norm_stats(norm_path)
    if len(norm.phys_mean) != 8 or len(norm.epw_mean) != 6:
        raise ValueError("Normalization dimensions do not match the selected MAIN input schema.")
    device = torch.device(device_name)
    model = ZoneOverheatPredictor(config).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval()
    raw_weather = parse_epw(epw)
    outdoor = outdoor_dry_bulb(epw)
    if start_hour < config["context_len"] or start_hour + hours > len(raw_weather) or hours < 1:
        raise ValueError("Forecast or preceding warm-up falls outside the EPW year.")
    weather = norm.norm_epw(raw_weather)
    pairs = paired_files(label_dir, voxel_dir)
    if building_ids:
        names = {p.stem.removesuffix("_label") for p, _ in pairs}
        if set(building_ids) - names:
            raise ValueError("Requested building ID has no matching input pair.")
        pairs = [(p, v) for p, v in pairs if p.stem.removesuffix("_label") in set(building_ids)]
    records = []
    headers = [
        "building_id",
        "zone_id",
        "timestamp",
        "q10_C",
        "q50_C",
        "q90_C",
        "reference_op_C",
        "outdoor_temperature_C",
    ]
    for label, voxel in pairs:
        building_id = label.stem.removesuffix("_label")
        path = output_dir / f"{building_id}_prediction.xlsx"
        if path.exists():
            raise FileExistsError(f"Prediction workbook already exists: {path.name}")
        building = BuildingCache(label, voxel, norm, raw_weather)
        batch, zone_indices = forecast_batch(
            building,
            weather,
            start_hour,
            hours,
            config,
            int(run_config.get("data", {}).get("nb_k", 15)),
        )
        batch = move_batch_to_device(batch, device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            prediction = model(batch)["pred"][0].float().cpu().numpy()
        prediction = prediction * norm.temp_std + norm.temp_mean
        with np.load(label, allow_pickle=False) as archive:
            truth, zones = archive["op"], archive["zones"].astype(str)
        if truth.shape != (8760, len(zones)) or prediction.shape != (len(zones), hours, 3):
            raise ValueError(
                "Prediction dimensions or annual reference do not match the zone schema."
            )
        if not np.isfinite(prediction).all():
            raise FloatingPointError("The selected checkpoint produced non-finite predictions.")
        violations = int(
            np.count_nonzero(
                (prediction[..., 0] > prediction[..., 1])
                | (prediction[..., 1] > prediction[..., 2])
            )
        )
        lookup = {int(original): row for row, original in enumerate(zone_indices)}

        def rows():
            for index in sorted(lookup, key=lambda i: str(zones[i])):
                for hour in range(hours):
                    position = start_hour + hour
                    yield [
                        building_id,
                        str(zones[index]),
                        hour_label(position),
                        *prediction[lookup[index], hour],
                        float(truth[position, index]),
                        float(outdoor[position]),
                    ]

        write_workbook(path, "Predictions", headers, rows(), len(zones) * hours)
        records.append(
            {
                "building_id": building_id,
                "zones": len(zones),
                "rows": len(zones) * hours,
                "quantile_order_violations": violations,
            }
        )
        print(
            f"Predicted {building_id}: {len(zones)} zones, {hours} hours, crossings={violations}",
            flush=True,
        )
    return records
