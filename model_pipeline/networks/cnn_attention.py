"""Encode sparse building voxels and condition each zone on global building tokens.

Inputs: Spatial tensors, temporal drivers, and model configuration.
Outputs: Encoded features, forecasts, or loss terms."""

from __future__ import annotations
import math
from typing import Any
from ..loaders.building_loader import VOXEL_FEATURE_DIM
import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor
except ImportError as exc:
    raise ImportError("cnn_attention requires spconv") from exc
CNN_TOTAL_STRIDE = (8, 8, 2)


class SparseResidualBlock(nn.Module):
    """Apply a residual sparse-convolution block while preserving active-site indexing."""

    def __init__(self, channels: int, indice_key: str) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            channels, channels, kernel_size=3, padding=1, bias=False, indice_key=f"{indice_key}_1"
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = spconv.SubMConv3d(
            channels, channels, kernel_size=3, padding=1, bias=False, indice_key=f"{indice_key}_2"
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()

    def forward(self, sparse: SparseConvTensor) -> SparseConvTensor:
        """Evaluate the configured computation and return predictions and optional loss terms."""
        identity = sparse.features
        output = self.conv1(sparse)
        output = output.replace_feature(self.activation(self.bn1(output.features)))
        output = self.conv2(output)
        output = output.replace_feature(self.activation(self.bn2(output.features) + identity))
        return output


class SparseCNNBackbone(nn.Module):
    """Encode physical voxel features using anisotropic sparse residual convolutions."""

    total_stride = CNN_TOTAL_STRIDE

    def __init__(self, in_channels: int = VOXEL_FEATURE_DIM, d_model: int = 256) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        if in_channels <= 0 or d_model <= 0:
            raise ValueError("in_channels and d_model must be positive")
        self.in_channels = in_channels
        self.d_model = d_model
        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, 32, kernel_size=3, padding=1, bias=False, indice_key="cnn_attn_stem_1"
            ),
            nn.BatchNorm1d(32),
            nn.GELU(),
            spconv.SubMConv3d(
                32, 32, kernel_size=3, padding=1, bias=False, indice_key="cnn_attn_stem_2"
            ),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.down_xy_1 = spconv.SparseSequential(
            spconv.SparseConv3d(
                32,
                64,
                kernel_size=3,
                stride=(2, 2, 1),
                padding=1,
                bias=False,
                indice_key="cnn_attn_down_1",
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.stage_1 = SparseResidualBlock(64, "cnn_attn_res_1")
        self.down_xyz = spconv.SparseSequential(
            spconv.SparseConv3d(
                64,
                128,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                bias=False,
                indice_key="cnn_attn_down_2",
            ),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.stage_2 = nn.ModuleList(
            [
                SparseResidualBlock(128, "cnn_attn_res_2a"),
                SparseResidualBlock(128, "cnn_attn_res_2b"),
            ]
        )
        self.down_xy_2 = spconv.SparseSequential(
            spconv.SparseConv3d(
                128,
                d_model,
                kernel_size=3,
                stride=(2, 2, 1),
                padding=1,
                bias=False,
                indice_key="cnn_attn_down_3",
            ),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.output_block = SparseResidualBlock(d_model, "cnn_attn_output")

    def forward(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: list[int],
        batch_size: int,
    ) -> SparseConvTensor:
        """Evaluate the configured computation and return predictions and optional loss terms."""
        if features.ndim != 2 or features.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected voxel_features (N, {self.in_channels}), got {tuple(features.shape)}"
            )
        if indices.ndim != 2 or indices.shape[1] != 4:
            raise ValueError("voxel_indices must have shape (N, 4)")
        if len(features) != len(indices):
            raise ValueError("voxel_features and voxel_indices row counts disagree")
        if len(features) == 0:
            raise ValueError("Sparse CNN received no active voxels")
        sparse = spconv.SparseConvTensor(
            features=features,
            indices=indices.to(torch.int32),
            spatial_shape=[int(value) for value in spatial_shape],
            batch_size=int(batch_size),
        )
        sparse = self.stem(sparse)
        sparse = self.down_xy_1(sparse)
        sparse = self.stage_1(sparse)
        sparse = self.down_xyz(sparse)
        for block in self.stage_2:
            sparse = block(sparse)
        sparse = self.down_xy_2(sparse)
        sparse = self.output_block(sparse)
        return sparse


class ZoneVoxelRouter(nn.Module):
    """Map zone-owned input voxels onto the sparse CNN output grid."""

    def __init__(self, total_stride: tuple[int, int, int] = CNN_TOTAL_STRIDE) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        if len(total_stride) != 3 or any((value <= 0 for value in total_stride)):
            raise ValueError("total_stride must contain three positive integers")
        self.total_stride = tuple((int(value) for value in total_stride))

    @staticmethod
    def _linear_hash(indices: torch.Tensor, extents: torch.Tensor) -> torch.Tensor:
        batch = indices[:, 0].to(torch.int64)
        xyz = indices[:, 1:].to(torch.int64)
        return ((batch * extents[0] + xyz[:, 0]) * extents[1] + xyz[:, 1]) * extents[2] + xyz[:, 2]

    def forward(
        self,
        input_indices: torch.Tensor,
        output_indices: torch.Tensor,
        zone_voxel_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the configured computation and return predictions and optional loss terms."""
        if zone_voxel_indices.ndim != 1:
            raise ValueError("zone_voxel_indices must be one-dimensional")
        if len(zone_voxel_indices) == 0:
            raise ValueError("Zone-Voxel CSR contains no edges")
        if len(output_indices) == 0:
            raise RuntimeError("Sparse CNN produced no output tokens")
        edge_indices = zone_voxel_indices.to(device=input_indices.device, dtype=torch.long)
        if int(edge_indices.min()) < 0 or int(edge_indices.max()) >= len(input_indices):
            raise IndexError("zone_voxel_indices refer outside the input voxel table")
        inherited = input_indices[edge_indices].to(torch.int64).clone()
        stride = torch.tensor(self.total_stride, dtype=torch.int64, device=inherited.device)
        inherited[:, 1:] = torch.div(inherited[:, 1:], stride, rounding_mode="floor")
        output_long = output_indices.to(device=inherited.device, dtype=torch.int64)
        max_xyz = torch.maximum(inherited[:, 1:].amax(dim=0), output_long[:, 1:].amax(dim=0))
        extents = max_xyz + 1
        inherited_hash = self._linear_hash(inherited, extents)
        output_hash = self._linear_hash(output_long, extents)
        sorted_hash, sorted_rows = torch.sort(output_hash)
        if len(torch.unique_consecutive(sorted_hash)) != len(sorted_hash):
            raise RuntimeError("Sparse CNN output contains duplicate coordinates")
        positions = torch.searchsorted(sorted_hash, inherited_hash)
        safe_positions = positions.clamp(max=len(sorted_hash) - 1)
        matched = (positions < len(sorted_hash)) & (sorted_hash[safe_positions] == inherited_hash)
        if not bool(matched.all()):
            missing = int((~matched).sum())
            raise RuntimeError(
                f"{missing}/{len(matched)} Zone-Voxel edges could not be inherited at CNN stride {self.total_stride}; convolution/grid alignment is inconsistent"
            )
        return sorted_rows[safe_positions]


class MetricDistanceBias(nn.Module):
    """Convert metric query/key distances into a bounded learned attention bias."""

    def __init__(self, init_sigma_m: float = 20.0, bias_clip: float = 12.0) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        if init_sigma_m <= 0.0 or bias_clip <= 0.0:
            raise ValueError("init_sigma_m and bias_clip must be positive")
        self.log_sigma_m = nn.Parameter(torch.tensor(math.log(init_sigma_m), dtype=torch.float32))
        self.bias_clip = float(bias_clip)

    @property
    def sigma_m(self) -> torch.Tensor:
        """Compute sigma m for the current building or model state."""
        return self.log_sigma_m.exp().clamp(min=0.05, max=100.0)

    def forward(self, query_coords_m: torch.Tensor, key_coords_m: torch.Tensor) -> torch.Tensor:
        """Evaluate the configured computation and return predictions and optional loss terms."""
        difference = query_coords_m.float().unsqueeze(1) - key_coords_m.float().unsqueeze(0)
        distance_sq = difference.square().sum(dim=-1)
        bias = -distance_sq / (2.0 * self.sigma_m.square())
        return bias.clamp(min=-self.bias_clip, max=0.0)


class ZoneGlobalCrossAttention(nn.Module):
    """Attend from zone-owned query tokens to same-building global keys and values."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_zone_queries: int = 256,
        max_global_tokens: int = 4096,
        init_sigma_m: float = 20.0,
        distance_bias_clip: float = 12.0,
        distance_chunk_size: int = 64,
    ) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        if d_model <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if max_zone_queries <= 0 or max_global_tokens <= 0:
            raise ValueError("Token limits must be positive")
        if distance_chunk_size <= 0:
            raise ValueError("distance_chunk_size must be positive")
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_zone_queries = max_zone_queries
        self.max_global_tokens = max_global_tokens
        self.distance_chunk_size = distance_chunk_size
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.distance_bias = MetricDistanceBias(
            init_sigma_m=init_sigma_m, bias_clip=distance_bias_clip
        )

    @staticmethod
    def _weighted_group_mean(
        values: torch.Tensor, inverse: torch.Tensor, weights: torch.Tensor, group_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = weights.to(dtype=values.dtype)
        group_weights = values.new_zeros(group_count)
        group_weights.index_add_(0, inverse, weights)
        grouped = values.new_zeros((group_count, values.shape[-1]))
        grouped.index_add_(0, inverse, values * weights.unsqueeze(-1))
        grouped = grouped / group_weights.clamp_min(1e-06).unsqueeze(-1)
        return (grouped, group_weights)

    def _build_queries(
        self,
        owned_features: torch.Tensor,
        owned_coords_m: torch.Tensor,
        owned_grid_indices: torch.Tensor,
        incidence_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_count = len(owned_features)
        weights = incidence_weights.to(dtype=owned_features.dtype)
        if token_count <= self.max_zone_queries:
            return (owned_features, owned_coords_m, weights)
        factor = max(2, int(math.ceil((token_count / self.max_zone_queries) ** (1.0 / 3.0))))
        while True:
            buckets = torch.div(owned_grid_indices.to(torch.int64), factor, rounding_mode="floor")
            _, inverse = torch.unique(buckets, dim=0, sorted=True, return_inverse=True)
            group_count = int(inverse.max()) + 1
            if group_count <= self.max_zone_queries:
                break
            factor += 1
        query_features, query_weights = self._weighted_group_mean(
            owned_features, inverse, weights, group_count
        )
        query_coords_m, _ = self._weighted_group_mean(owned_coords_m, inverse, weights, group_count)
        return (query_features, query_coords_m, query_weights)

    def _select_global_tokens(
        self, query_coords_m: torch.Tensor, global_coords_m: torch.Tensor
    ) -> torch.Tensor:
        global_count = len(global_coords_m)
        if global_count <= self.max_global_tokens:
            return torch.arange(global_count, device=global_coords_m.device)
        with torch.no_grad():
            nearest_distance_sq = torch.full(
                (global_count,), torch.inf, dtype=torch.float32, device=global_coords_m.device
            )
            global_coords = global_coords_m.float()
            for start in range(0, len(query_coords_m), self.distance_chunk_size):
                query_chunk = query_coords_m[start : start + self.distance_chunk_size].float()
                difference = query_chunk[:, None, :] - global_coords[None, :, :]
                chunk_min = difference.square().sum(dim=-1).amin(dim=0)
                nearest_distance_sq = torch.minimum(nearest_distance_sq, chunk_min)
            return torch.topk(
                nearest_distance_sq, k=self.max_global_tokens, largest=False, sorted=False
            ).indices

    def forward_zone(
        self,
        owned_features: torch.Tensor,
        owned_coords_m: torch.Tensor,
        owned_grid_indices: torch.Tensor,
        incidence_weights: torch.Tensor,
        global_features: torch.Tensor,
        global_coords_m: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Use zone-owned tokens as queries and same-building global tokens as keys/values, apply metric bias, and incidence-pool a zone embedding with token counts."""
        query_features, query_coords_m, query_weights = self._build_queries(
            owned_features, owned_coords_m, owned_grid_indices, incidence_weights
        )
        selected = self._select_global_tokens(query_coords_m, global_coords_m)
        key_value_features = global_features[selected]
        key_value_coords_m = global_coords_m[selected]
        query = self.norm_q(query_features).unsqueeze(0)
        key_value = self.norm_kv(key_value_features).unsqueeze(0)
        distance_bias = self.distance_bias(query_coords_m, key_value_coords_m)
        attention_mask = (
            distance_bias.to(dtype=query.dtype)
            .unsqueeze(0)
            .expand(self.n_heads, -1, -1)
            .contiguous()
        )
        attended, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=attention_mask,
            need_weights=False,
        )
        tokens = query_features + self.attention_dropout(attended.squeeze(0))
        tokens = tokens + self.ffn(self.norm_ffn(tokens))
        normalized_weights = query_weights / query_weights.sum().clamp_min(1e-06)
        zone_vector = (tokens * normalized_weights.unsqueeze(-1)).sum(dim=0)
        return (self.output_norm(zone_vector), len(query_features), len(key_value_features))


class BuildingCNNAttentionEncoder(nn.Module):
    """Encode sparse building voxels and condition each zone on global building tokens."""

    required_batch_keys = {
        "voxel_features",
        "voxel_indices",
        "spatial_shape",
        "batch_size",
        "grid_origins_m",
        "voxel_sizes_m",
        "zone_counts",
        "zone_mask",
        "zone_padding_mask",
        "zone_batch_indices",
        "zone_local_indices",
        "zone_voxel_ptr",
        "zone_voxel_indices",
        "zone_voxel_roles",
    }

    def __init__(
        self,
        in_channels: int = VOXEL_FEATURE_DIM,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_zone_queries: int = 256,
        max_global_tokens: int = 4096,
        init_sigma_m: float = 20.0,
        distance_bias_clip: float = 12.0,
        distance_chunk_size: int = 64,
    ) -> None:
        """Initialize and validate the supplied model or physical parameters."""
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.cnn = SparseCNNBackbone(in_channels=in_channels, d_model=d_model)
        self.router = ZoneVoxelRouter(total_stride=self.cnn.total_stride)
        self.attention = ZoneGlobalCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_zone_queries=max_zone_queries,
            max_global_tokens=max_global_tokens,
            init_sigma_m=init_sigma_m,
            distance_bias_clip=distance_bias_clip,
            distance_chunk_size=distance_chunk_size,
        )
        self._initialize_sparse_weights()

    def _initialize_sparse_weights(self) -> None:
        for module in self.cnn.modules():
            if isinstance(module, (spconv.SubMConv3d, spconv.SparseConv3d)):
                nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_batch(self, batch: dict[str, Any]) -> None:
        missing = sorted(self.required_batch_keys.difference(batch))
        if missing:
            raise KeyError(f"cnn_attention batch is missing keys: {missing}")
        if int(batch["batch_size"]) <= 0:
            raise ValueError("batch_size must be positive")
        features = batch["voxel_features"]
        indices = batch["voxel_indices"]
        if features.ndim != 2 or features.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected voxel_features (N, {self.in_channels}), got {tuple(features.shape)}"
            )
        if len(features) != len(indices):
            raise ValueError("voxel_features and voxel_indices row counts disagree")
        if len(batch["zone_voxel_ptr"]) != len(batch["zone_batch_indices"]) + 1:
            raise ValueError("zone_voxel_ptr length must equal total zones + 1")
        if len(batch["zone_voxel_indices"]) != len(batch["zone_voxel_roles"]):
            raise ValueError("Zone-Voxel indices and roles must have equal length")

    @staticmethod
    def _output_metric_coords(
        output_indices: torch.Tensor,
        grid_origins_m: torch.Tensor,
        voxel_sizes_m: torch.Tensor,
        total_stride: tuple[int, int, int],
    ) -> torch.Tensor:
        batch_indices = output_indices[:, 0].long()
        output_ijk = output_indices[:, 1:].float()
        stride = torch.tensor(total_stride, dtype=output_ijk.dtype, device=output_ijk.device)
        origins = grid_origins_m.to(output_ijk.device)[batch_indices]
        voxel_size = voxel_sizes_m.to(output_ijk.device)[batch_indices].unsqueeze(-1)
        return origins + (output_ijk + 0.5) * stride * voxel_size

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Evaluate the configured computation and return predictions and optional loss terms."""
        self._validate_batch(batch)
        sparse_output = self.cnn(
            features=batch["voxel_features"],
            indices=batch["voxel_indices"],
            spatial_shape=batch["spatial_shape"],
            batch_size=int(batch["batch_size"]),
        )
        global_features = sparse_output.features
        global_indices = sparse_output.indices
        if not torch.isfinite(global_features).all():
            raise FloatingPointError("Sparse CNN produced NaN or infinite features")
        global_batch_indices = global_indices[:, 0].long()
        global_coords_m = self._output_metric_coords(
            global_indices, batch["grid_origins_m"], batch["voxel_sizes_m"], self.cnn.total_stride
        )
        edge_to_global_token = self.router(
            input_indices=batch["voxel_indices"],
            output_indices=global_indices,
            zone_voxel_indices=batch["zone_voxel_indices"],
        )
        zone_ptr = batch["zone_voxel_ptr"].detach().cpu().tolist()
        zone_batch_indices = batch["zone_batch_indices"].to(
            global_features.device, dtype=torch.long
        )
        zone_local_indices = batch["zone_local_indices"].to(
            global_features.device, dtype=torch.long
        )
        total_zones = len(zone_ptr) - 1
        if total_zones != len(zone_batch_indices):
            raise ValueError("Flattened zone metadata disagrees with Zone-Voxel CSR")
        zone_vectors: list[torch.Tensor] = []
        inherited_token_chunks: list[torch.Tensor] = []
        inherited_weight_chunks: list[torch.Tensor] = []
        inherited_ptr = [0]
        query_counts: list[int] = []
        key_value_counts: list[int] = []
        for zone_index in range(total_zones):
            edge_start = int(zone_ptr[zone_index])
            edge_end = int(zone_ptr[zone_index + 1])
            mapped = edge_to_global_token[edge_start:edge_end]
            inherited_tokens, incidence_counts = torch.unique(
                mapped, sorted=True, return_counts=True
            )
            if len(inherited_tokens) == 0:
                raise RuntimeError(f"Zone {zone_index} inherited no CNN output tokens")
            building_index = zone_batch_indices[zone_index]
            if not bool((global_batch_indices[inherited_tokens] == building_index).all()):
                raise RuntimeError("A zone inherited tokens from another building")
            building_global_rows = torch.nonzero(
                global_batch_indices == building_index, as_tuple=False
            ).squeeze(-1)
            zone_vector, query_count, key_value_count = self.attention.forward_zone(
                owned_features=global_features[inherited_tokens],
                owned_coords_m=global_coords_m[inherited_tokens],
                owned_grid_indices=global_indices[inherited_tokens, 1:],
                incidence_weights=incidence_counts,
                global_features=global_features[building_global_rows],
                global_coords_m=global_coords_m[building_global_rows],
            )
            zone_vectors.append(zone_vector)
            inherited_token_chunks.append(inherited_tokens)
            inherited_weight_chunks.append(incidence_counts)
            inherited_ptr.append(inherited_ptr[-1] + len(inherited_tokens))
            query_counts.append(query_count)
            key_value_counts.append(key_value_count)
        zone_features_flat = torch.stack(zone_vectors, dim=0)
        batch_size = int(batch["batch_size"])
        max_zones = int(batch["zone_mask"].shape[1])
        flat_positions = zone_batch_indices * max_zones + zone_local_indices
        padded_flat = zone_features_flat.new_zeros((batch_size * max_zones, self.d_model))
        padded_flat = padded_flat.index_copy(0, flat_positions, zone_features_flat)
        zone_features = padded_flat.view(batch_size, max_zones, self.d_model)
        padding_mask = batch["zone_padding_mask"].to(
            device=global_features.device, dtype=torch.bool
        )
        return {
            "zone_features": zone_features,
            "zone_features_flat": zone_features_flat,
            "padding_mask": padding_mask,
            "global_features": global_features,
            "global_indices": global_indices,
            "global_coords_m": global_coords_m,
            "global_batch_indices": global_batch_indices,
            "zone_output_token_ptr": torch.tensor(
                inherited_ptr, dtype=torch.int64, device=global_features.device
            ),
            "zone_output_token_indices": torch.cat(inherited_token_chunks),
            "zone_output_token_weights": torch.cat(inherited_weight_chunks),
            "zone_query_counts": torch.tensor(
                query_counts, dtype=torch.int64, device=global_features.device
            ),
            "zone_key_value_counts": torch.tensor(
                key_value_counts, dtype=torch.int64, device=global_features.device
            ),
        }


__all__ = [
    "BuildingCNNAttentionEncoder",
    "CNN_TOTAL_STRIDE",
    "MetricDistanceBias",
    "SparseCNNBackbone",
    "SparseResidualBlock",
    "ZoneGlobalCrossAttention",
    "ZoneVoxelRouter",
]
