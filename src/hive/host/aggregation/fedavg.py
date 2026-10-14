"""
Federated Averaging (FedAvg) aggregator with optional server-side L2 clipping.
"""

from __future__ import annotations

from typing import Mapping

import torch

from hive.core.tensors import clip_by_norm, unflatten
from hive.host.aggregation.base import AggregationResult, Aggregator, ClientUpdate, compute_per_client_stats


class FedAvgAggregator(Aggregator):
    """
    Standard Federated Averaging: sample-weighted mean across client updates.
    Optionally applies a server-side L2 clipping bound to each update before averaging.
    """

    def __init__(self, server_clip_norm: float | None = None) -> None:
        self.server_clip_norm = server_clip_norm

    def aggregate(
        self,
        updates: list[ClientUpdate],
        template: Mapping[str, torch.Tensor],
    ) -> AggregationResult:
        if not updates:
            raise ValueError("Cannot aggregate empty list of updates")

        total_samples = sum(u.num_samples for u in updates)
        if total_samples <= 0:
            weights = [1.0 / len(updates)] * len(updates)
        else:
            weights = [u.num_samples / total_samples for u in updates]

        # Compute per-client stats before clipping
        per_client_stats = compute_per_client_stats(updates, weights=weights)

        # Apply server-side L2 clipping if configured
        clipped_vecs: list[torch.Tensor] = []
        for u in updates:
            vec = u.delta_vec
            if self.server_clip_norm is not None and self.server_clip_norm > 0:
                vec, _ = clip_by_norm(vec, self.server_clip_norm)
            clipped_vecs.append(vec)

        # Compute weighted sum
        stacked = torch.stack(clipped_vecs, dim=0)  # Shape: (N, D)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=stacked.device).unsqueeze(1)
        agg_vec = (stacked * weight_tensor).sum(dim=0)

        agg_dict = unflatten(agg_vec, template)
        return AggregationResult(
            delta=agg_dict,
            delta_vec=agg_vec,
            accepted_ids=[u.node_id for u in updates],
            rejected={},
            per_client_stats=per_client_stats,
            aggregator_used="fedavg",
            fallback_note=None,
        )