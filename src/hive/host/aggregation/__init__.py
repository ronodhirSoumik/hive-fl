"""
Hive Host Aggregation Engine.
Provides FedAvg, Trimmed Mean, Median, Krum, and Multi-Krum strategies.
"""

from hive.host.aggregation.base import (
    AggregationResult,
    Aggregator,
    ClientUpdate,
    compute_per_client_stats,
)
from hive.host.aggregation.factory import make_aggregator
from hive.host.aggregation.fedavg import FedAvgAggregator
from hive.host.aggregation.krum import KrumAggregator
from hive.host.aggregation.median import MedianAggregator
from hive.host.aggregation.trimmed_mean import TrimmedMeanAggregator

__all__ = [
    "ClientUpdate",
    "AggregationResult",
    "Aggregator",
    "compute_per_client_stats",
    "FedAvgAggregator",
    "TrimmedMeanAggregator",
    "MedianAggregator",
    "KrumAggregator",
    "make_aggregator",
]