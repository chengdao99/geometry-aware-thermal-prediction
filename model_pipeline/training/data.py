"""Prepare explicit train and validation pairs. Inputs: NPZ directories, EPW, and sampling configuration. Outputs: DataLoaders and training-only normalization."""

from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader
from ..loaders.building_loader import (
    BuildingCache,
    End2EndDataset,
    NormStats,
    SegLenBatchSampler,
    collate_fn,
    parse_epw,
)


def paired_files(label_dir: Path, voxel_dir: Path) -> list[tuple[Path, Path]]:
    """Pair inputs strictly by building ID and reject missing or duplicate identities."""
    labels = sorted(label_dir.glob("*_label.npz"))
    if not labels:
        raise FileNotFoundError(f"No label NPZ files in {label_dir.name}")
    pairs, seen = [], set()
    for label in labels:
        building_id = label.name.removesuffix("_label.npz")
        if building_id.casefold() in seen:
            raise ValueError(f"Duplicate building ID: {building_id}")
        seen.add(building_id.casefold())
        voxel = voxel_dir / f"{building_id}_voxel.npz"
        if not voxel.is_file():
            raise FileNotFoundError(f"Missing voxel pair: {building_id}")
        pairs.append((label, voxel))
    return pairs


def prepare_data(
    config: dict,
    train_labels: Path,
    train_voxels: Path,
    val_labels: Path,
    val_voxels: Path,
    epw: Path,
):
    """Fit training statistics and prepare serial loaders without implicit data splitting."""
    train_pairs = paired_files(train_labels, train_voxels)
    val_pairs = paired_files(val_labels, val_voxels)
    train_ids = {p.stem.removesuffix("_label").casefold() for p, _ in train_pairs}
    val_ids = {p.stem.removesuffix("_label").casefold() for p, _ in val_pairs}
    if train_ids & val_ids:
        raise ValueError("Training and validation building IDs overlap.")
    weather = parse_epw(epw)
    norm = NormStats()
    norm.fit([p for p, _ in train_pairs], [p for _, p in train_pairs], weather)
    normalized_weather = norm.norm_epw(weather)
    data, model, training = config["data"], config["model"], config["training"]
    loaders = []
    for is_train, pairs in [(True, train_pairs), (False, val_pairs)]:
        buildings = [BuildingCache(label, voxel, norm, weather) for label, voxel in pairs]
        segments = (
            data["segs"]
            if is_train
            else max(data["val_min_segs"], data["segs"] // data["val_segs_divisor"])
        )
        dataset = End2EndDataset(
            buildings,
            normalized_weather,
            segs_per_building=segments,
            seed=training["seed"] + (0 if is_train else 1),
            context_len=model["context_len"],
            min_seg_len=data["min_seg_len"],
            max_seg_len=data["max_seg_len"],
            cold_start_prob=data["cold_start_prob"] if is_train else data["val_cold_start_prob"],
            max_zones=data["max_zones"] if is_train else data.get("val_max_zones"),
            zone_sample_seed=(
                data.get("zone_sample_seed") if is_train else data.get("val_zone_sample_seed")
            ),
            fixed=False if is_train else data.get("val_fixed", False),
            nb_k=data["nb_k"],
            nb_dropout_max=data["nb_dropout_max"],
            nb_dropout_min_valid=data["nb_dropout_min_valid"],
            enable_neighbor_dropout=is_train,
            use_element_onehot=False,
        )
        batch_size = training["batch_size"] if is_train else training["val_batch_size"]
        sampler = SegLenBatchSampler(
            len(dataset),
            batch_size,
            data["min_seg_len"],
            data["max_seg_len"],
            seed=training["seed"] + (0 if is_train else 1),
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=int(data.get("workers", 0)),
            pin_memory=True,
        )
        loaders.append(loader)
    return (
        loaders[0],
        loaders[1],
        norm,
        {"train_pairs": len(train_pairs), "val_pairs": len(val_pairs)},
    )
