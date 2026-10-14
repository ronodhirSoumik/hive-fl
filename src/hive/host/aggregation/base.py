"""
Base abstractions and data structures for federated aggregation algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from hive.core.schemas import UpdateMeta
from hive.core.tensors import cosine_similarity, l2_norm, per_layer_norms, unflatten


@dataclass
class ClientUpdate:
    """Represents a validated weight update submitted by a contributor node."""
    node_id: str
    delta: dict[str, torch.Tensor]
    delta_vec: torch.Tensor
    num_samples: int
    meta: UpdateMeta


@dataclass
class AggregationResult:
    """Outcome of an aggregation round across contributor updates."""
    delta: dict[str, torch.Tensor]
    delta_vec: torch.Tensor
    accepted_ids: list[str]
    rejected: dict[str, str]
    per_client_stats: dict[str, dict[str, Any]]
    aggregator_used: str
    fallback_note: str | None = None


def compute_per_client_stats(
    updates: list[ClientUpdate],
    weights: list[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Computes per-client statistics:
    - L2 update norm
    - Cosine similarity to the mean of all other updates
    - Per-layer norms
    - Outlier flag (norm > median + 3*MAD or cosine < -0.2)
    - Relative contribution weight
    """
    n = len(updates)
    if n == 0:
        return {}

    norms = [l2_norm(u.delta_vec) for u in updates]
    sorted_norms = sorted(norms)
    median_norm = sorted_norms[n // 2]
    abs_deviations = sorted([abs(x - median_norm) for x in norms])
    mad = abs_deviations[n // 2] if n > 0 else 0.0

    stats: dict[str, dict[str, Any]] = {}
    for i, u in enumerate(updates):
        # Calculate mean of all OTHER updates
        other_vecs = [updates[j].delta_vec for j in range(n) if j != i]
        if other_vecs:
            mean_others = torch.stack(other_vecs, dim=0).mean(dim=0)
            cos_sim = cosine_similarity(u.delta_vec, mean_others)
        else:
            cos_sim = 1.0

        u_norm = norms[i]
        # Outlier rule: norm exceeds median + 3*MAD or strongly negative cosine similarity
        is_outlier = (u_norm > (median_norm + (3.0 * mad)) and mad > 1e-6) or (cos_sim < -0.2)

        stats[u.node_id] = {
            "norm": round(u_norm, 5),
            "cosine_similarity": round(cos_sim, 5),
            "layer_norms": {k: round(v, 5) for k, v in per_layer_norms(u.delta).items()},
            "is_outlier": bool(is_outlier),
            "num_samples": u.num_samples,
            "weight": round(weights[i], 5) if weights and i < len(weights) else None,
        }

    return stats


class Aggregator(ABC):
    """Abstract base class for federated weight aggregation algorithms."""

    @abstractmethod
    def aggregate(
        self,
        updates: list[ClientUpdate],
        template: Mapping[str, torch.Tensor],
    ) -> AggregationResult:
        """
        Aggregates a collection of client updates into a global model delta.
        """
        pass