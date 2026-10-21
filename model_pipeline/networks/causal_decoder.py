"""Forecast autoregressive quantiles using weather, geometry, and known-neighbor drivers.

Inputs: Spatial tensors, temporal drivers, and model configuration.
Outputs: Encoded features, forecasts, or loss terms."""

from __future__ import annotations
import math
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_zone_mean(values: torch.Tensor, zone_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("masked zone reduction expects a one-dimensional tensor")
    if zone_mask is None:
        return values.mean()
    if zone_mask.shape != values.shape:
        raise ValueError("zone_mask shape does not match zone-wise loss")
    weights = zone_mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _weighted_time_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.shape != weights.shape:
        raise ValueError("time-loss values and weights must have equal shapes")
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-06)


class ForecastLoss(nn.Module):
    """Combine event-weighted pinball, transition, focal, initialization, and crossing losses."""

    def __init__(
        self,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        temperature_scale_c: float = 1.0,
        change_weight_alpha: float = 2.0,
        change_weight_gamma: float = 2.0,
        change_threshold_c: float = 0.5,
        transition_loss_weight: float = 0.5,
        transition_beta_c: float = 0.25,
        focal_loss_weight: float = 0.25,
        focal_threshold_c: float = 1.0,
        focal_gamma: float = 2.0,
        t0_loss_weight: float = 0.25,
        crossing_loss_weight: float = 0.05,
    ) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        quantile_values = tuple((float(value) for value in quantiles))
        if not quantile_values or tuple(sorted(quantile_values)) != quantile_values:
            raise ValueError("quantiles must be a non-empty increasing sequence")
        if not 0.0 < quantile_values[0] or not quantile_values[-1] < 1.0:
            raise ValueError("quantiles must lie strictly between zero and one")
        if temperature_scale_c <= 0.0:
            raise ValueError("temperature_scale_c must be positive")
        if change_threshold_c <= 0.0 or transition_beta_c <= 0.0:
            raise ValueError("change and transition thresholds must be positive")
        if focal_threshold_c < 0.0 or focal_gamma <= 0.0:
            raise ValueError("invalid focal loss settings")
        for name, value in {
            "change_weight_alpha": change_weight_alpha,
            "change_weight_gamma": change_weight_gamma,
            "transition_loss_weight": transition_loss_weight,
            "focal_loss_weight": focal_loss_weight,
            "t0_loss_weight": t0_loss_weight,
            "crossing_loss_weight": crossing_loss_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.register_buffer(
            "quantile_levels", torch.tensor(quantile_values, dtype=torch.float32), persistent=True
        )
        self.temperature_scale_c = float(temperature_scale_c)
        self.change_weight_alpha = float(change_weight_alpha)
        self.change_weight_gamma = float(change_weight_gamma)
        self.change_threshold_c = float(change_threshold_c)
        self.transition_loss_weight = float(transition_loss_weight)
        self.transition_beta_c = float(transition_beta_c)
        self.focal_loss_weight = float(focal_loss_weight)
        self.focal_threshold_c = float(focal_threshold_c)
        self.focal_gamma = float(focal_gamma)
        self.t0_loss_weight = float(t0_loss_weight)
        self.crossing_loss_weight = float(crossing_loss_weight)
        self.median_index = min(
            range(len(quantile_values)), key=lambda index: abs(quantile_values[index] - 0.5)
        )

    def _event_weights(self, target_c: torch.Tensor) -> torch.Tensor:
        target_delta_c = torch.zeros_like(target_c)
        if target_c.shape[1] > 1:
            target_delta_c[:, 1:] = target_c[:, 1:] - target_c[:, :-1]
        excess_ratio = (
            (target_delta_c.abs() - self.change_threshold_c).clamp_min(0.0)
            / self.change_threshold_c
        ).clamp_max(1.0)
        weights = 1.0 + self.change_weight_alpha * excess_ratio**self.change_weight_gamma
        return weights.detach()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        indoor_t0: Optional[torch.Tensor] = None,
        target_t0: Optional[torch.Tensor] = None,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted quantile, transition, focal, initialization, and crossing objective plus its components. Physical thresholds use degrees Celsius through temperature_scale_c."""
        if pred.ndim != 3 or target.shape != pred.shape[:2]:
            raise ValueError("pred must be (N,T,Q) and target must be (N,T)")
        if pred.shape[-1] != len(self.quantile_levels):
            raise ValueError("prediction quantile count disagrees with configured levels")
        if pred.shape[1] == 0:
            raise ValueError("forecast loss received an empty time dimension")
        scale = self.temperature_scale_c
        target_c = target * scale
        pred_c = pred * scale
        q = self.quantile_levels.to(device=pred.device, dtype=pred.dtype)
        errors_c = target_c.unsqueeze(-1) - pred_c
        pinball_matrix = torch.maximum((q - 1.0) * errors_c, q * errors_c).mean(dim=-1)
        event_weights = self._event_weights(target_c)
        pinball = _masked_zone_mean(_weighted_time_mean(pinball_matrix, event_weights), zone_mask)
        median_c = pred_c[..., self.median_index]
        absolute_error_c = (median_c - target_c).abs()
        focal_matrix = (absolute_error_c - self.focal_threshold_c).clamp_min(
            0.0
        ) ** self.focal_gamma
        focal = _masked_zone_mean(_weighted_time_mean(focal_matrix, event_weights), zone_mask)
        if pred.shape[1] > 1:
            pred_delta_c = median_c[:, 1:] - median_c[:, :-1]
            target_delta_c = target_c[:, 1:] - target_c[:, :-1]
            transition_matrix = F.smooth_l1_loss(
                pred_delta_c, target_delta_c, beta=self.transition_beta_c, reduction="none"
            )
            transition = _masked_zone_mean(
                _weighted_time_mean(transition_matrix, event_weights[:, 1:]), zone_mask
            )
        else:
            transition = pred.sum() * 0.0
        if pred.shape[-1] > 1:
            crossing_matrix = F.relu(pred_c[..., :-1] - pred_c[..., 1:]).mean(dim=-1)
            crossing = _masked_zone_mean(crossing_matrix.mean(dim=-1), zone_mask)
        else:
            crossing = pred.sum() * 0.0
        if indoor_t0 is not None and target_t0 is not None:
            if indoor_t0.shape != target_t0.shape:
                raise ValueError("indoor_t0 and target_t0 shapes disagree")
            t0_per_zone = F.smooth_l1_loss(
                indoor_t0 * scale, target_t0 * scale, beta=self.transition_beta_c, reduction="none"
            )
            t0 = _masked_zone_mean(t0_per_zone, zone_mask)
        else:
            t0 = pred.sum() * 0.0
        total = (
            pinball
            + self.transition_loss_weight * transition
            + self.focal_loss_weight * focal
            + self.t0_loss_weight * t0
            + self.crossing_loss_weight * crossing
        )
        components = {
            "total": total,
            "pinball": pinball,
            "transition": transition,
            "focal": focal,
            "t0": t0,
            "crossing": crossing,
            "event_weight_mean": event_weights.mean(),
        }
        return (total, components)


class TokenBuilder(nn.Module):
    """Embed previous indoor state, weather/operating conditions, warm-up indicator, and configured temporal differences into decoder tokens."""

    def __init__(
        self,
        d_weather: int = 17,
        d_model: int = 128,
        dropout: float = 0.1,
        use_weather_delta: bool = True,
        weather_delta_indices: Sequence[int] = (0, 1, 2, 3, 4, 5),
        use_indoor_velocity: bool = True,
    ) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        self.d_weather = int(d_weather)
        self.use_weather_delta = bool(use_weather_delta)
        self.use_indoor_velocity = bool(use_indoor_velocity)
        indices = tuple((int(index) for index in weather_delta_indices))
        if self.use_weather_delta:
            if not indices or min(indices) < 0 or max(indices) >= self.d_weather:
                raise ValueError("weather_delta_indices are outside d_weather")
        self.register_buffer(
            "weather_delta_indices", torch.tensor(indices, dtype=torch.long), persistent=True
        )
        input_dim = 1 + self.d_weather + 1
        if self.use_indoor_velocity:
            input_dim += 1
        if self.use_weather_delta:
            input_dim += len(indices)
        self.input_dim = input_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        indoor_prev: torch.Tensor,
        weather: torch.Tensor,
        is_warmup: torch.Tensor,
        indoor_velocity: Optional[torch.Tensor] = None,
        weather_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate the configured computation and return predictions and optional loss terms."""
        if weather.ndim != 2 or weather.shape[-1] != self.d_weather:
            raise ValueError("weather has the wrong feature dimension")
        if indoor_prev.ndim != 1 or is_warmup.ndim != 1:
            raise ValueError("indoor_prev and is_warmup must be one-dimensional")
        parts = [indoor_prev.unsqueeze(-1)]
        if self.use_indoor_velocity:
            if indoor_velocity is None or indoor_velocity.shape != indoor_prev.shape:
                raise ValueError("indoor_velocity is required by this TokenBuilder")
            parts.append(indoor_velocity.unsqueeze(-1))
        parts.append(weather)
        if self.use_weather_delta:
            if weather_delta is None:
                raise ValueError("weather_delta is required by this TokenBuilder")
            expected = (weather.shape[0], len(self.weather_delta_indices))
            if weather_delta.shape != expected:
                raise ValueError(
                    f"weather_delta must have shape {expected}, got {tuple(weather_delta.shape)}"
                )
            parts.append(weather_delta)
        parts.append(is_warmup.unsqueeze(-1))
        return self.projection(torch.cat(parts, dim=-1))


class NeighborContextAttention(nn.Module):
    """Attend to known neighbor operating drivers using static distance, volume, and contact topology; neighbor indoor truth is not an input."""

    def __init__(
        self,
        d_model: int = 128,
        d_neighbor: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        init_distance_sigma_m: float = 4.0,
        distance_bias_clip: float = 12.0,
    ) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if d_neighbor <= 0 or init_distance_sigma_m <= 0.0:
            raise ValueError("invalid neighbor attention dimensions")
        self.d_model = d_model
        self.d_neighbor = d_neighbor
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = float(dropout)
        self.distance_bias_clip = float(distance_bias_clip)
        self.dynamic_projection = nn.Sequential(
            nn.Linear(d_neighbor, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.relation_projection = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.topology_bias = nn.Linear(2, n_heads, bias=False)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_value_projection = nn.Linear(d_model, 2 * d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        inverse_softplus = math.log(math.expm1(init_distance_sigma_m))
        self.raw_distance_sigma_m = nn.Parameter(
            torch.tensor(inverse_softplus, dtype=torch.float32)
        )

    @property
    def distance_sigma_m(self) -> torch.Tensor:
        """Compute distance sigma m for the current building or model state."""
        return F.softplus(self.raw_distance_sigma_m) + 0.0001

    def _to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(batch, tokens, self.n_heads, self.d_head).transpose(1, 2)

    def prepare_topology(
        self,
        distance_squared_m: torch.Tensor,
        volume_ratio: torch.Tensor,
        contact_ratio: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute prepare topology for the current building or model state. Inputs: distance_squared_m, volume_ratio, contact_ratio, padding_mask."""
        if distance_squared_m.ndim != 2:
            raise ValueError("distance_squared_m must have shape (M,K)")
        batch, neighbors = distance_squared_m.shape
        expected = (batch, neighbors)
        for name, value in {
            "volume_ratio": volume_ratio,
            "contact_ratio": contact_ratio,
            "padding_mask": padding_mask,
        }.items():
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        distance_m = distance_squared_m.clamp_min(0.0).sqrt()
        log_volume_ratio = volume_ratio.clamp_min(1e-06).log()
        contact = contact_ratio.clamp(0.0, 1.0)
        relation = torch.stack([torch.log1p(distance_m), log_volume_ratio, contact], dim=-1)
        relation_embedding = self.relation_projection(relation)
        sigma = self.distance_sigma_m.to(
            device=distance_squared_m.device, dtype=distance_squared_m.dtype
        )
        metric_bias = (-distance_squared_m / (2.0 * sigma.square())).clamp(
            min=-self.distance_bias_clip, max=0.0
        )
        topology_bias = self.topology_bias(
            torch.stack([log_volume_ratio, contact], dim=-1)
        ).permute(0, 2, 1)
        return {
            "relation_embedding": relation_embedding,
            "attention_bias": topology_bias.unsqueeze(2) + metric_bias[:, None, None, :],
            "valid": (~padding_mask.bool())[:, None, None, :],
        }

    def prepare_dynamic(
        self, neighbor_features: torch.Tensor, topology: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute prepare dynamic for the current building or model state. Inputs: neighbor_features, topology."""
        if neighbor_features.ndim != 3:
            raise ValueError("neighbor_features must have shape (M,K,d_neighbor)")
        batch, neighbors, feature_dim = neighbor_features.shape
        if feature_dim != self.d_neighbor:
            raise ValueError("neighbor feature dimension disagrees with d_neighbor")
        relation_embedding = topology["relation_embedding"]
        if relation_embedding.shape != (batch, neighbors, self.d_model):
            raise ValueError("cached neighbor relation embedding has the wrong shape")
        neighbor_tokens = self.dynamic_projection(neighbor_features) + relation_embedding
        key, value = self.key_value_projection(neighbor_tokens).chunk(2, dim=-1)
        return {
            "key": self._to_heads(key),
            "value": self._to_heads(value),
            "attention_bias": topology["attention_bias"],
            "valid": topology["valid"],
        }

    def prepare(
        self,
        neighbor_features: torch.Tensor,
        distance_squared_m: torch.Tensor,
        volume_ratio: torch.Tensor,
        contact_ratio: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute prepare for the current building or model state. Inputs: neighbor_features, distance_squared_m, volume_ratio, contact_ratio, padding_mask."""
        topology = self.prepare_topology(
            distance_squared_m, volume_ratio, contact_ratio, padding_mask
        )
        return self.prepare_dynamic(neighbor_features, topology)

    def forward_prepared(
        self, self_token: torch.Tensor, prepared: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute forward prepared for the current building or model state. Inputs: self_token, prepared."""
        batch = self_token.shape[0]
        query = self._to_heads(self.query_projection(self.query_norm(self_token)).unsqueeze(1))
        logits = torch.matmul(query, prepared["key"].transpose(-2, -1)) * self.d_head ** (-0.5)
        logits = logits + prepared["attention_bias"].to(dtype=logits.dtype)
        valid = prepared["valid"]
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        attention = attention * valid.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-08)
        attention = F.dropout(
            attention, p=self.dropout_p if self.training else 0.0, training=self.training
        )
        output = torch.matmul(attention, prepared["value"])
        output = output.transpose(1, 2).contiguous().view(batch, self.d_model)
        token = self_token + self.output_projection(output)
        token = self.output_norm(token)
        token = token + self.ffn(self.ffn_norm(token))
        has_neighbor = valid.any(dim=-1).reshape(batch)
        return torch.where(has_neighbor.unsqueeze(-1), token, self_token)

    def forward(
        self,
        self_token: torch.Tensor,
        neighbor_features: torch.Tensor,
        distance_squared_m: torch.Tensor,
        volume_ratio: torch.Tensor,
        contact_ratio: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the configured computation and return predictions and optional loss terms."""
        return self.forward_prepared(
            self_token,
            self.prepare(
                neighbor_features, distance_squared_m, volume_ratio, contact_ratio, padding_mask
            ),
        )


class CausalDecoderLayer(nn.Module):
    """Combine causal temporal attention, geometry conditioning, and feed-forward transformations for one decoder layer."""

    def __init__(
        self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1, context_len: int = 72
    ) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.context_len = int(context_len)
        self.dropout_p = float(dropout)
        self.self_attention_projection = nn.Linear(d_model, 3 * d_model, bias=False)
        self.self_attention_output = nn.Linear(d_model, d_model)
        self.self_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def _to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = tensor.shape
        return tensor.view(batch, steps, self.n_heads, self.d_head).transpose(1, 2)

    def _from_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, steps, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch, steps, -1)

    def forward_parallel(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute forward parallel for the current building or model state. Inputs: tokens."""
        q, key, value = self.self_attention_projection(self.self_norm(tokens)).chunk(3, dim=-1)
        q = self._to_heads(q)
        key = self._to_heads(key)
        value = self._to_heads(value)
        attended = F.scaled_dot_product_attention(
            q, key, value, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0
        )
        tokens = tokens + self.self_attention_output(self._from_heads(attended))
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return (tokens, (key, value))

    def forward_single(
        self, token: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute forward single for the current building or model state. Inputs: token, cache."""
        q, key_new, value_new = self.self_attention_projection(self.self_norm(token)).chunk(
            3, dim=-1
        )
        q = self._to_heads(q)
        key_new = self._to_heads(key_new)
        value_new = self._to_heads(value_new)
        key_cache, value_cache = cache
        key = torch.cat([key_cache, key_new], dim=2)
        value = torch.cat([value_cache, value_new], dim=2)
        if key.shape[2] > self.context_len:
            key = key[:, :, -self.context_len :, :]
            value = value[:, :, -self.context_len :, :]
        attended = F.scaled_dot_product_attention(
            q, key, value, dropout_p=self.dropout_p if self.training else 0.0
        )
        token = token + self.self_attention_output(self._from_heads(attended))
        token = token + self.ffn(self.ffn_norm(token))
        return (token, (key, value))


class ZoneCausalDecoder(nn.Module):
    """Forecast autoregressive quantiles using weather, geometry, and known-neighbor drivers."""

    def __init__(
        self,
        d_static: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        context_len: int = 72,
        d_weather: int = 17,
        n_quantiles: int = 3,
        dropout: float = 0.1,
        tbptt_k: int = 72,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        use_weather_delta: bool = True,
        weather_delta_indices: Sequence[int] = (0, 1, 2, 3, 4, 5),
        use_indoor_velocity: bool = True,
        nb_feat_dim: int = 4,
        neighbor_distance_sigma_m: float = 4.0,
        neighbor_distance_bias_clip: float = 12.0,
        temperature_scale_c: float = 1.0,
        change_weight_alpha: float = 2.0,
        change_weight_gamma: float = 2.0,
        change_threshold_c: float = 0.5,
        transition_loss_weight: float = 0.5,
        transition_beta_c: float = 0.25,
        focal_loss_weight: float = 0.25,
        focal_threshold_c: float = 1.0,
        focal_gamma: float = 2.0,
        t0_loss_weight: float = 0.25,
        crossing_loss_weight: float = 0.05,
    ) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        if n_quantiles != len(quantiles):
            raise ValueError("n_quantiles must equal len(quantiles)")
        if context_len <= 0 or n_layers <= 0 or tbptt_k <= 0:
            raise ValueError("context_len, n_layers and tbptt_k must be positive")
        self.context_len = int(context_len)
        self.d_model = int(d_model)
        self.d_weather = int(d_weather)
        self.n_quantiles = int(n_quantiles)
        self.nb_feat_dim = int(nb_feat_dim)
        self.tbptt_k = int(tbptt_k)
        self.quantiles = tuple((float(value) for value in quantiles))
        self.median_index = min(
            range(len(self.quantiles)), key=lambda index: abs(self.quantiles[index] - 0.5)
        )
        self.spatial_projection = nn.Sequential(
            nn.Linear(d_static, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.token_builder = TokenBuilder(
            d_weather=d_weather,
            d_model=d_model,
            dropout=dropout,
            use_weather_delta=use_weather_delta,
            weather_delta_indices=weather_delta_indices,
            use_indoor_velocity=use_indoor_velocity,
        )
        self.layers = nn.ModuleList(
            [
                CausalDecoderLayer(
                    d_model=d_model, n_heads=n_heads, dropout=dropout, context_len=context_len
                )
                for _ in range(n_layers)
            ]
        )
        self.initial_temperature_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, n_quantiles)
        self.neighbor_attention = NeighborContextAttention(
            d_model=d_model,
            d_neighbor=nb_feat_dim,
            n_heads=n_heads,
            dropout=dropout,
            init_distance_sigma_m=neighbor_distance_sigma_m,
            distance_bias_clip=neighbor_distance_bias_clip,
        )
        self.forecast_loss = ForecastLoss(
            quantiles=quantiles,
            temperature_scale_c=temperature_scale_c,
            change_weight_alpha=change_weight_alpha,
            change_weight_gamma=change_weight_gamma,
            change_threshold_c=change_threshold_c,
            transition_loss_weight=transition_loss_weight,
            transition_beta_c=transition_beta_c,
            focal_loss_weight=focal_loss_weight,
            focal_threshold_c=focal_threshold_c,
            focal_gamma=focal_gamma,
            t0_loss_weight=t0_loss_weight,
            crossing_loss_weight=crossing_loss_weight,
        )

    def _validate_inputs(
        self,
        z_geom: torch.Tensor,
        weather_warmup: torch.Tensor,
        warmup_indoor: torch.Tensor,
        weather_ar: torch.Tensor,
        target_ar: Optional[torch.Tensor],
        op_norm_t0: Optional[torch.Tensor],
        zone_mask: Optional[torch.Tensor],
    ) -> None:
        if z_geom.ndim != 2:
            raise ValueError("z_geom must have shape (N,d_static)")
        batch = z_geom.shape[0]
        if weather_warmup.shape != (batch, self.context_len, self.d_weather):
            raise ValueError("weather_warmup shape disagrees with context_len or d_weather")
        if warmup_indoor.shape != (batch, self.context_len):
            raise ValueError("warmup_indoor shape disagrees with context_len")
        if (
            weather_ar.ndim != 3
            or weather_ar.shape[0] != batch
            or weather_ar.shape[2] != self.d_weather
            or (weather_ar.shape[1] == 0)
        ):
            raise ValueError("weather_ar must have shape (N,T,d_weather) with T > 0")
        if target_ar is not None and target_ar.shape != weather_ar.shape[:2]:
            raise ValueError("target_ar shape disagrees with weather_ar")
        if op_norm_t0 is not None and op_norm_t0.shape != (batch,):
            raise ValueError("op_norm_t0 must have shape (N,)")
        if zone_mask is not None and zone_mask.shape != (batch,):
            raise ValueError("zone_mask must have shape (N,)")

    def _weather_deltas(
        self, weather_warmup: torch.Tensor, weather_ar: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.token_builder.use_weather_delta:
            return (None, None)
        indices = self.token_builder.weather_delta_indices
        warm = weather_warmup.index_select(-1, indices)
        future = weather_ar.index_select(-1, indices)
        warm_delta = torch.zeros_like(warm)
        warm_delta[:, 1:] = warm[:, 1:] - warm[:, :-1]
        future_delta = torch.zeros_like(future)
        future_delta[:, 0] = future[:, 0] - warm[:, -1]
        if future.shape[1] > 1:
            future_delta[:, 1:] = future[:, 1:] - future[:, :-1]
        return (warm_delta, future_delta)

    @staticmethod
    def _neighbor_inputs_complete(values: Sequence[Optional[torch.Tensor]]) -> bool:
        present = [value is not None for value in values]
        if any(present) and (not all(present)):
            raise ValueError("neighbor inputs must be provided together")
        return all(present)

    def _prepare_neighbor_sequence(
        self,
        neighbor_features: torch.Tensor,
        distance_squared_m: torch.Tensor,
        volume_ratio: torch.Tensor,
        contact_ratio: torch.Tensor,
        padding_mask: torch.Tensor,
        topology: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        batch, steps, neighbors, feature_dim = neighbor_features.shape
        if feature_dim != self.nb_feat_dim:
            raise ValueError("neighbor feature dimension disagrees with nb_feat_dim")
        if topology is None:
            topology = self.neighbor_attention.prepare_topology(
                distance_squared_m, volume_ratio, contact_ratio, padding_mask
            )
        expanded = {
            "relation_embedding": topology["relation_embedding"][:, None]
            .expand(batch, steps, neighbors, self.d_model)
            .reshape(batch * steps, neighbors, self.d_model),
            "attention_bias": topology["attention_bias"][:, None]
            .expand(batch, steps, *topology["attention_bias"].shape[1:])
            .reshape(batch * steps, *topology["attention_bias"].shape[1:]),
            "valid": topology["valid"][:, None]
            .expand(batch, steps, *topology["valid"].shape[1:])
            .reshape(batch * steps, *topology["valid"].shape[1:]),
        }
        prepared = self.neighbor_attention.prepare_dynamic(
            neighbor_features.reshape(batch * steps, neighbors, feature_dim), expanded
        )
        return {
            "key": prepared["key"].reshape(
                batch, steps, self.neighbor_attention.n_heads, neighbors, -1
            ),
            "value": prepared["value"].reshape(
                batch, steps, self.neighbor_attention.n_heads, neighbors, -1
            ),
            "attention_bias": prepared["attention_bias"].reshape(
                batch, steps, self.neighbor_attention.n_heads, 1, neighbors
            ),
            "valid": prepared["valid"].reshape(batch, steps, 1, 1, neighbors),
        }

    @staticmethod
    def _prepared_step(prepared: dict[str, torch.Tensor], step: int) -> dict[str, torch.Tensor]:
        return {name: value[:, step] for name, value in prepared.items()}

    def forward_ar(
        self,
        z_geom: Optional[torch.Tensor],
        weather_warmup: torch.Tensor,
        warmup_indoor: torch.Tensor,
        weather_ar: torch.Tensor,
        target_ar: Optional[torch.Tensor] = None,
        op_norm_t0: Optional[torch.Tensor] = None,
        zone_mask: Optional[torch.Tensor] = None,
        nb_warmup: Optional[torch.Tensor] = None,
        nb_ar: Optional[torch.Tensor] = None,
        nb_dist2: Optional[torch.Tensor] = None,
        nb_mask: Optional[torch.Tensor] = None,
        nb_v_ratio: Optional[torch.Tensor] = None,
        nb_contact_ratio: Optional[torch.Tensor] = None,
        spatial_context: Optional[torch.Tensor] = None,
        nb_topology: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, object]:
        """Embed the preceding context and autoregressively predict operative-temperature quantiles. Target temperatures and op_norm_t0 contribute only to optional loss calculation, never prediction feedback."""
        if spatial_context is None:
            if z_geom is None:
                raise ValueError("z_geom is required when spatial_context is absent")
            spatial_context = self.spatial_projection(z_geom)
        self._validate_inputs(
            spatial_context,
            weather_warmup,
            warmup_indoor,
            weather_ar,
            target_ar,
            op_norm_t0,
            zone_mask,
        )
        batch = spatial_context.shape[0]
        forecast_steps = weather_ar.shape[1]
        device = spatial_context.device
        warm_weather_delta, future_weather_delta = self._weather_deltas(weather_warmup, weather_ar)
        warm_indoor_delta = None
        if self.token_builder.use_indoor_velocity:
            warm_indoor_delta = torch.zeros_like(warmup_indoor)
            warm_indoor_delta[:, 1:] = warmup_indoor[:, 1:] - warmup_indoor[:, :-1]
        warm_tokens = self.token_builder(
            indoor_prev=warmup_indoor.reshape(-1),
            weather=weather_warmup.reshape(-1, self.d_weather),
            is_warmup=torch.ones(
                batch * self.context_len, device=device, dtype=weather_warmup.dtype
            ),
            indoor_velocity=(
                warm_indoor_delta.reshape(-1) if warm_indoor_delta is not None else None
            ),
            weather_delta=(
                warm_weather_delta.reshape(-1, len(self.token_builder.weather_delta_indices))
                if warm_weather_delta is not None
                else None
            ),
        ).reshape(batch, self.context_len, self.d_model)
        warm_tokens = warm_tokens + spatial_context.unsqueeze(1)
        neighbor_values = (nb_warmup, nb_ar, nb_dist2, nb_mask, nb_v_ratio, nb_contact_ratio)
        use_neighbors = self._neighbor_inputs_complete(neighbor_values)
        if use_neighbors:
            if nb_warmup.shape[:2] != (batch, self.context_len):
                raise ValueError("nb_warmup shape disagrees with warmup inputs")
            if nb_ar.shape[:2] != (batch, forecast_steps):
                raise ValueError("nb_ar shape disagrees with forecast inputs")
            neighbors = nb_dist2.shape[1]
            if neighbors > 0:
                warm_prepared = self._prepare_neighbor_sequence(
                    nb_warmup, nb_dist2, nb_v_ratio, nb_contact_ratio, nb_mask, topology=nb_topology
                )
                warm_tokens = self.neighbor_attention.forward_prepared(
                    warm_tokens.reshape(batch * self.context_len, self.d_model),
                    {
                        name: value.reshape(batch * self.context_len, *value.shape[2:])
                        for name, value in warm_prepared.items()
                    },
                ).reshape(batch, self.context_len, self.d_model)
                future_prepared = self._prepare_neighbor_sequence(
                    nb_ar, nb_dist2, nb_v_ratio, nb_contact_ratio, nb_mask, topology=nb_topology
                )
            else:
                use_neighbors = False
                future_prepared = None
        else:
            future_prepared = None
        hidden = warm_tokens
        caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            hidden, cache = layer.forward_parallel(hidden)
            caches.append(cache)
        indoor_t0 = self.initial_temperature_head(hidden[:, -1]).squeeze(-1)
        indoor_prev = indoor_t0
        indoor_prev_last = warmup_indoor[:, -1]
        predictions: list[torch.Tensor] = []
        for step in range(forecast_steps):
            if step > 0 and step % self.tbptt_k == 0:
                indoor_prev = indoor_prev.detach()
                indoor_prev_last = indoor_prev_last.detach()
                caches = [(key.detach(), value.detach()) for key, value in caches]
            indoor_velocity = (
                indoor_prev - indoor_prev_last if self.token_builder.use_indoor_velocity else None
            )
            token = self.token_builder(
                indoor_prev=indoor_prev,
                weather=weather_ar[:, step],
                is_warmup=torch.zeros(batch, device=device, dtype=weather_ar.dtype),
                indoor_velocity=indoor_velocity,
                weather_delta=(
                    future_weather_delta[:, step] if future_weather_delta is not None else None
                ),
            )
            token = token + spatial_context
            if use_neighbors:
                token = self.neighbor_attention.forward_prepared(
                    token, self._prepared_step(future_prepared, step)
                )
            hidden_step = token.unsqueeze(1)
            new_caches = []
            for layer, cache in zip(self.layers, caches):
                hidden_step, new_cache = layer.forward_single(hidden_step, cache)
                new_caches.append(new_cache)
            caches = new_caches
            step_prediction = self.output_projection(self.output_norm(hidden_step.squeeze(1)))
            predictions.append(step_prediction)
            indoor_prev_last = indoor_prev
            indoor_prev = step_prediction[:, self.median_index]
        pred = torch.stack(predictions, dim=1)
        if target_ar is not None:
            loss, components = self.forecast_loss(
                pred=pred,
                target=target_ar,
                indoor_t0=indoor_t0,
                target_t0=op_norm_t0,
                zone_mask=zone_mask,
            )
        else:
            loss = None
            components = {}
        return {"pred": pred, "loss": loss, "loss_components": components, "indoor_t0": indoor_t0}

    def forward(self, *args, **kwargs):
        """Evaluate the configured computation and return predictions and optional loss terms."""
        return self.forward_ar(*args, **kwargs)


__all__ = [
    "CausalDecoderLayer",
    "ForecastLoss",
    "NeighborContextAttention",
    "TokenBuilder",
    "ZoneCausalDecoder",
]
