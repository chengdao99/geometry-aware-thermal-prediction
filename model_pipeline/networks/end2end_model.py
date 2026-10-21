"""Predict zone operative-temperature quantiles from spatial and temporal conditions.

Inputs: Spatial tensors, temporal drivers, and model configuration.
Outputs: Encoded features, forecasts, or loss terms."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
from .causal_decoder import ZoneCausalDecoder
from ..loaders.building_loader import VOXEL_FEATURE_DIM
from .cnn_attention import BuildingCNNAttentionEncoder


@dataclass(frozen=True)
class GeometryCache:
    """Store learned geometry and neighbor topology for repeated temporal forecasts."""

    spatial_context: torch.Tensor
    padding_mask: torch.Tensor
    neighbor_relation_embedding: torch.Tensor | None = None
    neighbor_attention_bias: torch.Tensor | None = None
    neighbor_valid: torch.Tensor | None = None


class ZoneOverheatPredictor(nn.Module):
    """Predict zone operative-temperature quantiles from spatial and temporal conditions."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        spatial_dim = int(config.get("cnn_out_dim", 256))
        loss_config = dict(config.get("loss", {}))
        self.cnn_attention = BuildingCNNAttentionEncoder(
            in_channels=int(config.get("cnn_in_dim", VOXEL_FEATURE_DIM)),
            d_model=spatial_dim,
            n_heads=int(config.get("attn_heads", 8)),
            dropout=float(config.get("dropout", 0.1)),
            max_zone_queries=int(config.get("max_zone_queries", 256)),
            max_global_tokens=int(config.get("max_global_tokens", 4096)),
            init_sigma_m=float(config.get("spatial_bias_sigma", 20.0)),
            distance_bias_clip=float(config.get("distance_bias_clip", 12.0)),
            distance_chunk_size=int(config.get("distance_chunk_size", 64)),
        )
        self.decoder = ZoneCausalDecoder(
            d_static=spatial_dim,
            d_model=int(config.get("decoder_d_model", 128)),
            n_heads=int(config.get("decoder_heads", 4)),
            n_layers=int(config.get("decoder_layers", 4)),
            context_len=int(config.get("context_len", 72)),
            d_weather=int(config.get("d_weather", 17)),
            n_quantiles=int(config.get("n_quantiles", 3)),
            dropout=float(config.get("dropout", 0.1)),
            tbptt_k=int(config.get("tbptt_k", 72)),
            quantiles=tuple(config.get("quantiles", [0.1, 0.5, 0.9])),
            use_weather_delta=bool(config.get("use_weather_delta", True)),
            weather_delta_indices=tuple(config.get("weather_delta_indices", [0, 1, 2, 3, 4, 5])),
            use_indoor_velocity=bool(config.get("use_indoor_velocity", True)),
            nb_feat_dim=int(config.get("nb_feat_dim", 4)),
            neighbor_distance_sigma_m=float(config.get("neighbor_distance_sigma_m", 4.0)),
            neighbor_distance_bias_clip=float(config.get("neighbor_distance_bias_clip", 12.0)),
            temperature_scale_c=float(loss_config.get("temperature_scale_c", 1.0)),
            change_weight_alpha=float(loss_config.get("change_weight_alpha", 2.0)),
            change_weight_gamma=float(loss_config.get("change_weight_gamma", 2.0)),
            change_threshold_c=float(loss_config.get("change_threshold_c", 0.5)),
            transition_loss_weight=float(loss_config.get("transition_loss_weight", 0.5)),
            transition_beta_c=float(loss_config.get("transition_beta_c", 0.25)),
            focal_loss_weight=float(loss_config.get("focal_loss_weight", 0.25)),
            focal_threshold_c=float(loss_config.get("focal_threshold_c", 1.0)),
            focal_gamma=float(loss_config.get("focal_gamma", 2.0)),
            t0_loss_weight=float(loss_config.get("t0_loss_weight", 0.25)),
            crossing_loss_weight=float(loss_config.get("crossing_loss_weight", 0.05)),
        )

    @staticmethod
    def _flatten_valid(tensor: torch.Tensor, valid_flat: torch.Tensor) -> torch.Tensor:
        padded_zone_count = valid_flat.numel()
        if tensor.shape[0] * tensor.shape[1] != padded_zone_count:
            raise ValueError("zone-padded tensor shape disagrees with padding mask")
        return tensor.reshape(padded_zone_count, *tensor.shape[2:])[valid_flat]

    def encode_spatial(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute encode spatial for the current building or model state. Inputs: batch."""
        return self.cnn_attention(batch)

    def encode_static_geometry(
        self, batch: dict[str, Any], spatial_output: dict[str, torch.Tensor] | None = None
    ) -> GeometryCache:
        """Encode static geometry and neighbor topology for reuse across forecasts."""
        if spatial_output is None:
            spatial_output = self.encode_spatial(batch)
        zone_features = spatial_output["zone_features"]
        padding_mask = spatial_output["padding_mask"]
        valid_flat = (~padding_mask).reshape(-1)
        if not bool(valid_flat.any()):
            raise ValueError("batch contains no valid zones")
        batch_size, max_zones, spatial_dim = zone_features.shape
        padded_count = batch_size * max_zones
        valid_features = zone_features.reshape(padded_count, spatial_dim)[valid_flat]
        valid_context = self.decoder.spatial_projection(valid_features)
        spatial_context = valid_context.new_zeros(padded_count, valid_context.shape[-1])
        valid_positions = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
        spatial_context = spatial_context.index_copy(0, valid_positions, valid_context)
        spatial_context = spatial_context.view(batch_size, max_zones, -1)
        static_names = ("nb_dist2", "nb_mask", "nb_v_ratio", "nb_contact_ratio")
        present = [name in batch and batch[name] is not None for name in static_names]
        if any(present) and (not all(present)):
            missing = [name for name, ok in zip(static_names, present) if not ok]
            raise KeyError(f"incomplete static neighbor topology; missing {missing}")
        if not all(present):
            return GeometryCache(spatial_context=spatial_context, padding_mask=padding_mask)
        neighbors = int(batch["nb_dist2"].shape[-1])
        topology = self.decoder.neighbor_attention.prepare_topology(
            batch["nb_dist2"].reshape(padded_count, neighbors),
            batch["nb_v_ratio"].reshape(padded_count, neighbors),
            batch["nb_contact_ratio"].reshape(padded_count, neighbors),
            batch["nb_mask"].reshape(padded_count, neighbors),
        )
        heads = self.decoder.neighbor_attention.n_heads
        return GeometryCache(
            spatial_context=spatial_context,
            padding_mask=padding_mask,
            neighbor_relation_embedding=topology["relation_embedding"].view(
                batch_size, max_zones, neighbors, self.decoder.d_model
            ),
            neighbor_attention_bias=topology["attention_bias"].view(
                batch_size, max_zones, heads, 1, neighbors
            ),
            neighbor_valid=topology["valid"].view(batch_size, max_zones, 1, 1, neighbors),
        )

    def _forecast(
        self,
        batch: dict[str, Any],
        *,
        zone_features: torch.Tensor | None,
        padding_mask: torch.Tensor,
        geometry_cache: GeometryCache | None = None,
    ) -> dict[str, Any]:
        required = {"weather_warmup", "warmup_indoor", "weather_ar"}
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"end-to-end batch is missing keys: {missing}")
        valid_flat = (~padding_mask).reshape(-1)
        if not bool(valid_flat.any()):
            raise ValueError("batch contains no valid zones")
        batch_size, max_zones = padding_mask.shape
        padded_count = batch_size * max_zones
        valid_zone_features = None
        valid_spatial_context = None
        topology = None
        if geometry_cache is None:
            if zone_features is None:
                raise ValueError("zone_features are required without a geometry cache")
            valid_zone_features = zone_features.reshape(padded_count, -1)[valid_flat]
        else:
            if geometry_cache.padding_mask.shape != padding_mask.shape:
                raise ValueError("geometry cache padding mask shape disagrees with batch")
            valid_spatial_context = geometry_cache.spatial_context.reshape(padded_count, -1)[
                valid_flat
            ]
            topology_fields = (
                geometry_cache.neighbor_relation_embedding,
                geometry_cache.neighbor_attention_bias,
                geometry_cache.neighbor_valid,
            )
            if any((value is not None for value in topology_fields)):
                if not all((value is not None for value in topology_fields)):
                    raise ValueError("geometry cache contains incomplete neighbor topology")
                topology = {
                    "relation_embedding": topology_fields[0].reshape(
                        padded_count, *topology_fields[0].shape[2:]
                    )[valid_flat],
                    "attention_bias": topology_fields[1].reshape(
                        padded_count, *topology_fields[1].shape[2:]
                    )[valid_flat],
                    "valid": topology_fields[2].reshape(
                        padded_count, *topology_fields[2].shape[2:]
                    )[valid_flat],
                }
        weather_warmup = self._flatten_valid(batch["weather_warmup"], valid_flat)
        warmup_indoor = self._flatten_valid(batch["warmup_indoor"], valid_flat)
        weather_ar = self._flatten_valid(batch["weather_ar"], valid_flat)
        forecast_steps = weather_ar.shape[1]
        target_ar = (
            self._flatten_valid(batch["target_ar"], valid_flat)
            if batch.get("target_ar") is not None
            else None
        )
        op_norm_t0 = (
            self._flatten_valid(batch["op_norm_t0"], valid_flat)
            if batch.get("op_norm_t0") is not None
            else None
        )
        neighbor_names = (
            "nb_warmup",
            "nb_ar",
            "nb_dist2",
            "nb_mask",
            "nb_v_ratio",
            "nb_contact_ratio",
        )
        neighbor_presence = [name in batch and batch[name] is not None for name in neighbor_names]
        if any(neighbor_presence) and (not all(neighbor_presence)):
            missing_neighbors = [
                name for name, ok in zip(neighbor_names, neighbor_presence) if not ok
            ]
            raise KeyError(f"incomplete neighbor input set; missing {missing_neighbors}")
        neighbor_inputs = (
            {name: self._flatten_valid(batch[name], valid_flat) for name in neighbor_names}
            if all(neighbor_presence)
            else {}
        )
        decoder_output = self.decoder(
            z_geom=valid_zone_features,
            spatial_context=valid_spatial_context,
            nb_topology=topology,
            weather_warmup=weather_warmup,
            warmup_indoor=warmup_indoor,
            weather_ar=weather_ar,
            target_ar=target_ar,
            op_norm_t0=op_norm_t0,
            **neighbor_inputs,
        )
        flat_prediction = decoder_output["pred"]
        valid_positions = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
        padded_prediction = flat_prediction.new_zeros(
            padded_count, forecast_steps, flat_prediction.shape[-1]
        ).index_copy(0, valid_positions, flat_prediction)
        prediction = padded_prediction.view(
            batch_size, max_zones, forecast_steps, flat_prediction.shape[-1]
        )
        flat_t0 = decoder_output["indoor_t0"]
        padded_t0 = flat_t0.new_zeros(padded_count).index_copy(0, valid_positions, flat_t0)
        return {
            "pred": prediction,
            "loss": decoder_output["loss"],
            "loss_components": decoder_output["loss_components"],
            "padding_mask": padding_mask,
            "indoor_t0": padded_t0.view(batch_size, max_zones),
        }

    def forward(self, batch: dict[str, Any], spatial_output=None) -> dict[str, Any]:
        """Evaluate the configured computation and return predictions and optional loss terms."""
        if spatial_output is None:
            spatial_output = self.encode_spatial(batch)
        return self._forecast(
            batch,
            zone_features=spatial_output["zone_features"],
            padding_mask=spatial_output["padding_mask"],
        )

    def forecast_from_geometry_cache(
        self, geometry_cache: GeometryCache, dynamic_inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Forecast from cached geometry and fresh temporal inputs."""
        return self._forecast(
            dynamic_inputs,
            zone_features=None,
            padding_mask=geometry_cache.padding_mask,
            geometry_cache=geometry_cache,
        )

    @torch.no_grad()
    def predict(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute predict for the current building or model state. Inputs: batch."""
        return self.forward(batch)["pred"]


__all__ = ["ZoneOverheatPredictor"]
