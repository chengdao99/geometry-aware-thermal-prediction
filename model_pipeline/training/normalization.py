"""Serialize model normalization and move batches to the selected device. Inputs: NormStats or batch dictionaries. Outputs: Matching NPZ statistics or device tensors."""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import torch
from ..loaders.building_loader import NormStats

logger = logging.getLogger(__name__)


def save_norm_stats(norm: NormStats, path: Path | str) -> None:
    """Serialize normalization statistics without changing their numerical representation."""
    np.savez(
        str(path),
        temp_mean=np.array([norm.temp_mean], dtype=np.float32),
        temp_std=np.array([norm.temp_std], dtype=np.float32),
        epw_mean=norm.epw_mean,
        epw_std=norm.epw_std,
        phys_mean=norm.phys_mean,
        phys_std=norm.phys_std,
        ach_mean=np.array([norm.ach_mean], dtype=np.float32),
        ach_std=np.array([norm.ach_std], dtype=np.float32),
        equip_mean=np.array([norm.equip_mean], dtype=np.float32),
        equip_std=np.array([norm.equip_std], dtype=np.float32),
        occ_mean=np.array([norm.occ_mean], dtype=np.float32),
        occ_std=np.array([norm.occ_std], dtype=np.float32),
        hvac_mean=np.array([norm.hvac_mean], dtype=np.float32),
        hvac_std=np.array([norm.hvac_std], dtype=np.float32),
    )
    logger.info(f"[NormStats] Saved → {path}")


def load_norm_stats(path: Path | str) -> NormStats:
    """Read the normalization statistics required by the saved model configuration."""
    d = np.load(str(path))
    norm = NormStats()
    norm.temp_mean = float(d["temp_mean"][0])
    norm.temp_std = float(d["temp_std"][0])
    norm.epw_mean = d["epw_mean"].astype(np.float32)
    norm.epw_std = d["epw_std"].astype(np.float32)
    norm.phys_mean = d["phys_mean"].astype(np.float32)
    norm.phys_std = d["phys_std"].astype(np.float32)
    norm.ach_mean = float(d["ach_mean"][0])
    norm.ach_std = float(d["ach_std"][0])
    norm.equip_mean = float(d["equip_mean"][0]) if "equip_mean" in d else 0.0
    norm.equip_std = float(d["equip_std"][0]) if "equip_std" in d else 1.0
    norm.occ_mean = float(d["occ_mean"][0]) if "occ_mean" in d else 0.0
    norm.occ_std = float(d["occ_std"][0]) if "occ_std" in d else 1.0
    norm.hvac_mean = float(d["hvac_mean"][0]) if "hvac_mean" in d else 0.0
    norm.hvac_std = float(d["hvac_std"][0]) if "hvac_std" in d else 1.0
    logger.info(f"[NormStats] Loaded ← {path}")
    return norm


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor inputs to the selected device while retaining non-tensor metadata."""
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            moved[k] = [t.to(device) for t in v]
        else:
            moved[k] = v
    return moved
