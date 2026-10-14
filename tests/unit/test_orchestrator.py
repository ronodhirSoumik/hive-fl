"""
Unit tests for Host Orchestrator round state machine, claims, updates, and aggregation.
"""

from pathlib import Path
import pytest
import torch

from hive.core.config import HostConfig
from hive.core.schemas import Capabilities, UpdateMeta
from hive.core.tensors import state_dict_to_bytes
from hive.host.db import HostDatabase
from hive.host.orchestrator import Orchestrator


@pytest.fixture
def test_setup(tmp_path: Path) -> tuple[Orchestrator, HostDatabase]:
    cfg = HostConfig(
        total_rounds=2,
        min_clients=2,
        max_clients_per_round=10,
        round_timeout_s=30,
        workdir=tmp_path / "host_workdir",
    )
    db = HostDatabase(tmp_path / "host_workdir" / "hive.db")
    initial_weights = {
        "linear.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "linear.bias": torch.tensor([0.5], dtype=torch.float32),
    }
    orchestrator = Orchestrator(
        config=cfg,
        db=db,
        package_digest="sha256:pkg123",
        host_key_fingerprint="ed25519:hostfp",
        initial_weights=initial_weights,
    )
    return orchestrator, db


@pytest.mark.asyncio
async def test_orchestrator_full_round_lifecycle(test_setup: tuple[Orchestrator, HostDatabase]) -> None:
    orchestrator, db = test_setup

    # 1. Register 2 nodes
    cap = Capabilities(os="linux", cores=4, has_docker=True, has_gpu=False)
    n1_id, n1_sec, _ = orchestrator.register_node("Node-1", cap)
    n2_id, n2_sec, _ = orchestrator.register_node("Node-2", cap)

    # 2. Authentication check
    assert orchestrator.authenticate_node(n1_id, n1_sec) is True
    assert orchestrator.authenticate_node(n1_id, "wrong_secret") is False

    # 3. Mark nodes ready with local samples
    orchestrator.mark_node_ready(n1_id, num_samples=100)
    orchestrator.mark_node_ready(n2_id, num_samples=200)

    # 4. Start orchestrator
    await orchestrator.start()
    assert orchestrator.state.status.value == "running"

    # Initial round 0 checkpoint should exist
    assert (orchestrator.checkpoints_dir / "round_0.safetensors").exists()

    # 5. Trigger tick: eligible count is 2 (>= min_clients 2) -> Opens Round 1
    await orchestrator._tick()
    latest_r = db.get_latest_round()
    assert latest_r is not None
    assert latest_r["round_id"] == 1
    assert latest_r["status"] == "open"
    base_hash = latest_r["model_hash_in"]

    # 6. Nodes claim model weights
    model_bytes_1 = orchestrator.claim_model(n1_id, round_id=1)
    model_bytes_2 = orchestrator.claim_model(n2_id, round_id=1)
    assert len(model_bytes_1) > 0
    assert len(model_bytes_2) > 0

    # 7. Nodes submit updates
    delta_1 = {
        "linear.weight": torch.tensor([[0.1, 0.1]], dtype=torch.float32),
        "linear.bias": torch.tensor([0.1], dtype=torch.float32),
    }
    meta_1 = UpdateMeta(
        round_id=1,
        base_model_hash=base_hash,
        num_train_samples=100,
        train_loss=0.5,
        eval_loss=0.4,
        eval_metrics={"accuracy": 0.85},
    )
    orchestrator.submit_update(n1_id, round_id=1, meta=meta_1, delta_bytes=state_dict_to_bytes(delta_1))

    delta_2 = {
        "linear.weight": torch.tensor([[0.2, 0.2]], dtype=torch.float32),
        "linear.bias": torch.tensor([0.2], dtype=torch.float32),
    }
    meta_2 = UpdateMeta(
        round_id=1,
        base_model_hash=base_hash,
        num_train_samples=200,
        train_loss=0.4,
        eval_loss=0.35,
        eval_metrics={"accuracy": 0.90},
    )
    orchestrator.submit_update(n2_id, round_id=1, meta=meta_2, delta_bytes=state_dict_to_bytes(delta_2))

    # 8. Trigger tick: both claims submitted -> Aggregation executes and closes Round 1
    await orchestrator._tick()
    closed_r = db.get_round(1)
    assert closed_r["status"] == "closed"
    assert closed_r["model_hash_out"] is not None
    assert closed_r["model_hash_out"] != base_hash

    # Checkpoint for round 1 must exist on disk
    assert (orchestrator.checkpoints_dir / "round_1.safetensors").exists()

    # Metrics recorded
    metrics = db.list_metrics()
    assert len(metrics) == 1
    assert metrics[0]["round_id"] == 1
    assert metrics[0]["participants"] == 2
    assert 0.85 <= metrics[0]["eval_accuracy"] <= 0.90

    await orchestrator.stop()