"""
Coordinate-wise Trimmed Mean aggregator with automatic FedAvg fallback.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

from hive.core.tensors import unflatten
from hive.host.aggregation.base import AggregationResult, Aggregator, ClientUpdate, compute_per_client_stats
from hive.host.aggregation.fedavg import FedAvgAggregator


class TrimmedMeanAggregator(Aggregator):
    """
    Coordinate-wise Trimmed Mean:
    Sorts values along each coordinate across clients and removes the top and bottom
    k = floor(n * trim_ratio) updates.
    Requires n >= 3 and floor(n * trim_ratio) >= 1; otherwise falls back to FedAvg.
    """

    def __init__(self, trim_ratio: float = 0.2, server_clip_norm: float | None = None) -> None:
        self.trim_ratio = trim_ratio
        self.server_clip_norm = server_clip_norm

    def aggregate(
        self,
        updates: list[ClientUpdate],
        template: Mapping[str, torch.Tensor],
    ) -> AggregationResult:
        n = len(updates)
        if n == 0:
            raise ValueError("Cannot aggregate empty list of updates")

        k = math.floor(n * self.trim_ratio)

        # Fallback condition check
        if n < 3 or k < 1:
            fallback = FedAvgAggregator(server_clip_norm=self.server_clip_norm)
            res = fallback.aggregate(updates, template)
            res.aggregator_used = "trimmed-mean"
            res.fallback_note = f"n={n} insufficient for trim_ratio={self.trim_ratio} (k={k}); fell back to fedavg"
            return res

        per_client_stats = compute_per_client_stats(updates)

        stacked = torch.stack([u.delta_vec for u in updates], dim=0)  # (N, D)
        sorted_coords, _ = torch.sort(stacked, dim=0)

        # Slice out lowest k and highest k along coordinate axis
        trimmed = sorted_coords[k : n - k, :]
        agg_vec = trimmed.mean(dim=0)

        agg_dict = unflatten(agg_vec, template)
        return AggregationResult(
            delta=agg_dict,
            delta_vec=agg_vec,
            accepted_ids=[u.node_id for u in updates],
            rejected={},
            per_client_stats=per_client_stats,
            aggregator_used="trimmed-mean",
            fallback_note=None,
        )