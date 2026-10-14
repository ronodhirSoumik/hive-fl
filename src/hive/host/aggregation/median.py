"""
Coordinate-wise Median aggregator with automatic FedAvg fallback.
"""

from __future__ import annotations

from typing import Mapping

import torch

from hive.core.tensors import unflatten
from hive.host.aggregation.base import AggregationResult, Aggregator, ClientUpdate, compute_per_client_stats
from hive.host.aggregation.fedavg import FedAvgAggregator


class MedianAggregator(Aggregator):
    """
    Coordinate-wise Median: takes the median value along each model weight dimension.
    Requires n >= 3; otherwise falls back to FedAvg with a note.
    """

    def __init__(self, server_clip_norm: float | None = None) -> None:
        self.server_clip_norm = server_clip_norm

    def aggregate(
        self,
        updates: list[ClientUpdate],
        template: Mapping[str, torch.Tensor],
    ) -> AggregationResult:
        n = len(updates)
        if n == 0:
            raise ValueError("Cannot aggregate empty list of updates")

        if n < 3:
            fallback = FedAvgAggregator(server_clip_norm=self.server_clip_norm)
            res = fallback.aggregate(updates, template)
            res.aggregator_used = "median"
            res.fallback_note = f"n={n} insufficient for coordinate-wise median (requires >= 3); fell back to fedavg"
            return res

        per_client_stats = compute_per_client_stats(updates)

        stacked = torch.stack([u.delta_vec for u in updates], dim=0)  # (N, D)
        agg_vec = torch.median(stacked, dim=0).values

        agg_dict = unflatten(agg_vec, template)
        return AggregationResult(
            delta=agg_dict,
            delta_vec=agg_vec,
            accepted_ids=[u.node_id for u in updates],
            rejected={},
            per_client_stats=per_client_stats,
            aggregator_used="median",
            fallback_note=None,
        )