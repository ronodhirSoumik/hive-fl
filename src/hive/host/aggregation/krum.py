"""
Krum and Multi-Krum Byzantine-robust aggregation algorithms with automatic fallbacks.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

from hive.core.tensors import unflatten
from hive.host.aggregation.base import AggregationResult, Aggregator, ClientUpdate, compute_per_client_stats
from hive.host.aggregation.fedavg import FedAvgAggregator
from hive.host.aggregation.trimmed_mean import TrimmedMeanAggregator


class KrumAggregator(Aggregator):
    """
    Krum Byzantine-robust aggregator:
    Selects the candidate update that is closest to its n - f - 2 nearest neighbors.
    Requires n >= 2f + 3 where f = max_byzantine; otherwise falls back to trimmed-mean or FedAvg.
    """

    def __init__(
        self,
        max_byzantine: int | None = None,
        multi_krum: bool = False,
        server_clip_norm: float | None = None,
    ) -> None:
        self.max_byzantine = max_byzantine
        self.multi_krum = multi_krum
        self.server_clip_norm = server_clip_norm

    def aggregate(
        self,
        updates: list[ClientUpdate],
        template: Mapping[str, torch.Tensor],
    ) -> AggregationResult:
        n = len(updates)
        if n == 0:
            raise ValueError("Cannot aggregate empty list of updates")

        # Determine number of assumed Byzantine clients f
        f = self.max_byzantine if self.max_byzantine is not None else max(0, (n - 3) // 2)

        # Check Krum condition: n >= 2f + 3
        if n < 2 * f + 3 or n < 3:
            # Fall back to TrimmedMean or FedAvg
            fallback: Aggregator
            if n >= 3:
                fallback = TrimmedMeanAggregator(server_clip_norm=self.server_clip_norm)
                fallback_name = "trimmed-mean"
            else:
                fallback = FedAvgAggregator(server_clip_norm=self.server_clip_norm)
                fallback_name = "fedavg"

            res = fallback.aggregate(updates, template)
            res.aggregator_used = "multi-krum" if self.multi_krum else "krum"
            res.fallback_note = f"n={n} insufficient for Krum with f={f} (requires n >= 2f+3); fell back to {fallback_name}"
            return res

        per_client_stats = compute_per_client_stats(updates)

        # Pairwise distance matrix computation
        stacked = torch.stack([u.delta_vec for u in updates], dim=0)  # (N, D)
        # diff[i, j] = stacked[i] - stacked[j]
        diff = stacked.unsqueeze(1) - stacked.unsqueeze(0)  # (N, N, D)
        dists = torch.sum(diff**2, dim=-1)  # (N, N)

        # For each client i, take sum of distances to its (n - f - 2) nearest neighbors (excluding self)
        num_neighbors = n - f - 2
        scores: list[float] = []
        for i in range(n):
            # Sort row i distances
            sorted_dists, _ = torch.sort(dists[i])
            # sorted_dists[0] is distance to self (0.0), take indices 1 to num_neighbors inclusive
            score = torch.sum(sorted_dists[1 : num_neighbors + 1]).item()
            scores.append(score)

        if not self.multi_krum:
            # Single Krum: pick index with minimal score
            best_idx = int(torch.tensor(scores).argmin().item())
            agg_vec = stacked[best_idx].clone()
            accepted_ids = [updates[best_idx].node_id]
        else:
            # Multi-Krum: pick top m = n - f candidates and average them
            m = max(1, n - f)
            sorted_indices = torch.tensor(scores).argsort()[:m].tolist()
            agg_vec = stacked[sorted_indices].mean(dim=0)
            accepted_ids = [updates[idx].node_id for idx in sorted_indices]

        agg_dict = unflatten(agg_vec, template)
        return AggregationResult(
            delta=agg_dict,
            delta_vec=agg_vec,
            accepted_ids=accepted_ids,
            rejected={u.node_id: "not_selected_by_krum" for u in updates if u.node_id not in accepted_ids},
            per_client_stats=per_client_stats,
            aggregator_used="multi-krum" if self.multi_krum else "krum",
            fallback_note=None,
        )