"""
Factory for creating aggregation engine instances from configuration.
"""

from __future__ import annotations

from hive.core.schemas import AggregationConfig, AggregationName
from hive.host.aggregation.base import Aggregator
from hive.host.aggregation.fedavg import FedAvgAggregator
from hive.host.aggregation.krum import KrumAggregator
from hive.host.aggregation.median import MedianAggregator
from hive.host.aggregation.trimmed_mean import TrimmedMeanAggregator


def make_aggregator(
    name: str | AggregationName = AggregationName.FEDAVG,
    config: AggregationConfig | None = None,
) -> Aggregator:
    """
    Instantiates an Aggregator instance configured with the specified strategy.
    """
    if config is None:
        config = AggregationConfig()

    strategy = str(name).lower().replace("_", "-")
    if strategy.startswith("aggregationname."):
        strategy = strategy.split(".", 1)[1]

    if strategy in ("fedavg", "mean"):
        return FedAvgAggregator(server_clip_norm=config.server_clip_norm)
    elif strategy in ("trimmed-mean", "trimmedmean"):
        return TrimmedMeanAggregator(
            trim_ratio=config.trim_ratio,
            server_clip_norm=config.server_clip_norm,
        )
    elif strategy == "median":
        return MedianAggregator(server_clip_norm=config.server_clip_norm)
    elif strategy == "krum":
        return KrumAggregator(
            max_byzantine=config.max_byzantine,
            multi_krum=False,
            server_clip_norm=config.server_clip_norm,
        )
    elif strategy in ("multi-krum", "multikrum"):
        return KrumAggregator(
            max_byzantine=config.max_byzantine,
            multi_krum=True,
            server_clip_norm=config.server_clip_norm,
        )
    else:
        raise ValueError(f"Unknown aggregation strategy: '{name}'. Choose from fedavg, trimmed-mean, median, krum, multi-krum.")