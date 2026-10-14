"""
Unit tests for Host aggregation strategies: FedAvg, Trimmed Mean, Median, Krum, Multi-Krum.
Tests mathematical correctness, Byzantine outlier resistance, and safety fallbacks.
"""

import pytest
import torch

from hive.core.schemas import AggregationConfig, AggregationName, UpdateMeta
from hive.core.tensors import flatten, l2_norm
from hive.host.aggregation import (
    ClientUpdate,
    FedAvgAggregator,
    KrumAggregator,
    MedianAggregator,
    TrimmedMeanAggregator,
    make_aggregator,
)


@pytest.fixture
def model_template() -> dict[str, torch.Tensor]:
    return {
        "weight": torch.zeros((2, 2), dtype=torch.float32),
        "bias": torch.zeros((2,), dtype=torch.float32),
    }


def make_update(node_id: str, values: list[float], num_samples: int = 100) -> ClientUpdate:
    """Creates a mock ClientUpdate with 6 parameters (4 weights + 2 biases)."""
    t_val = torch.tensor(values, dtype=torch.float32)
    delta = {
        "weight": t_val[:4].reshape(2, 2),
        "bias": t_val[4:].reshape(2),
    }
    meta = UpdateMeta(
        round_id=1,
        base_model_hash="sha256:base",
        num_train_samples=num_samples,
        train_loss=0.5,
        eval_loss=0.4,
    )
    return ClientUpdate(
        node_id=node_id,
        delta=delta,
        delta_vec=t_val,
        num_samples=num_samples,
        meta=meta,
    )


def test_fedavg_weighted_mean(model_template: dict[str, torch.Tensor]) -> None:
    # 2 clients with different sample counts
    u1 = make_update("node_1", [1.0] * 6, num_samples=100)  # weight 1/3
    u2 = make_update("node_2", [4.0] * 6, num_samples=200)  # weight 2/3

    agg = FedAvgAggregator()
    res = agg.aggregate([u1, u2], model_template)

    assert res.aggregator_used == "fedavg"
    assert res.fallback_note is None
    assert len(res.accepted_ids) == 2

    # Expected: 1/3 * 1.0 + 2/3 * 4.0 = 1/3 + 8/3 = 3.0
    expected_vec = torch.tensor([3.0] * 6, dtype=torch.float32)
    assert torch.allclose(res.delta_vec, expected_vec)


def test_fedavg_server_clipping(model_template: dict[str, torch.Tensor]) -> None:
    # Vector with norm = 10.0
    u = make_update("node_1", [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], num_samples=100)

    # Aggregator with server_clip_norm = 4.0
    agg = FedAvgAggregator(server_clip_norm=4.0)
    res = agg.aggregate([u], model_template)

    # Norm must be bounded by 4.0
    assert pytest.approx(l2_norm(res.delta_vec), rel=1e-5) == 4.0
    assert pytest.approx(res.delta_vec[0].item(), rel=1e-5) == 4.0


def test_trimmed_mean_outlier_resistance(model_template: dict[str, torch.Tensor]) -> None:
    # 5 clients: 4 honest around 1.0, 1 poisoned outlier at 100.0
    updates = [
        make_update("h1", [1.0] * 6),
        make_update("h2", [1.1] * 6),
        make_update("h3", [0.9] * 6),
        make_update("h4", [1.05] * 6),
        make_update("poisoned", [100.0] * 6),
    ]

    # Trimmed mean with ratio 0.2 (k = floor(5*0.2) = 1) trims 1 lowest and 1 highest
    agg = TrimmedMeanAggregator(trim_ratio=0.2)
    res = agg.aggregate(updates, model_template)

    # Output should ignore 100.0 and stay close to 1.0
    mean_val = res.delta_vec[0].item()
    assert 0.95 <= mean_val <= 1.15
    assert res.aggregator_used == "trimmed-mean"
    assert res.fallback_note is None


def test_trimmed_mean_fallback_condition(model_template: dict[str, torch.Tensor]) -> None:
    # Only 2 clients -> cannot trim, must fall back to FedAvg
    u1 = make_update("c1", [2.0] * 6)
    u2 = make_update("c2", [4.0] * 6)

    agg = TrimmedMeanAggregator(trim_ratio=0.2)
    res = agg.aggregate([u1, u2], model_template)

    assert res.fallback_note is not None
    assert "fell back to fedavg" in res.fallback_note
    assert torch.allclose(res.delta_vec, torch.tensor([3.0] * 6))


def test_median_outlier_resistance(model_template: dict[str, torch.Tensor]) -> None:
    updates = [
        make_update("c1", [1.0] * 6),
        make_update("c2", [2.0] * 6),
        make_update("poisoned", [500.0] * 6),
    ]
    agg = MedianAggregator()
    res = agg.aggregate(updates, model_template)

    # Coordinate-wise median of [1, 2, 500] is 2.0
    assert torch.allclose(res.delta_vec, torch.tensor([2.0] * 6))
    assert res.fallback_note is None


def test_krum_byzantine_resistance(model_template: dict[str, torch.Tensor]) -> None:
    # 5 clients: 4 honest around 1.0, 1 severe Byzantine poisoner
    updates = [
        make_update("c1", [1.0] * 6),
        make_update("c2", [1.02] * 6),
        make_update("c3", [0.98] * 6),
        make_update("c4", [1.01] * 6),
        make_update("byzantine", [50.0] * 6),
    ]

    # n=5, f=1: 2f + 3 = 5 <= 5. Valid Krum condition!
    agg = KrumAggregator(max_byzantine=1)
    res = agg.aggregate(updates, model_template)

    # Must select an honest update, rejecting the byzantine node
    assert "byzantine" not in res.accepted_ids
    assert len(res.accepted_ids) == 1
    assert 0.95 <= res.delta_vec[0].item() <= 1.05


def test_krum_fallback_when_insufficient_clients(model_template: dict[str, torch.Tensor]) -> None:
    # n=2, cannot run Krum
    updates = [make_update("c1", [1.0] * 6), make_update("c2", [3.0] * 6)]
    agg = KrumAggregator()
    res = agg.aggregate(updates, model_template)

    assert res.fallback_note is not None
    assert "fell back to fedavg" in res.fallback_note


def test_make_aggregator_factory() -> None:
    assert isinstance(make_aggregator("fedavg"), FedAvgAggregator)
    assert isinstance(make_aggregator("trimmed-mean"), TrimmedMeanAggregator)
    assert isinstance(make_aggregator("median"), MedianAggregator)
    assert isinstance(make_aggregator("krum"), KrumAggregator)
    assert isinstance(make_aggregator(AggregationName.MULTI_KRUM), KrumAggregator)

    with pytest.raises(ValueError, match="Unknown aggregation"):
        make_aggregator("nonexistent_algo")