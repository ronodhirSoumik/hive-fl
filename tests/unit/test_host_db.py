"""
Unit tests for Host SQLite database persistence layer.
"""

from pathlib import Path
from hive.core.schemas import Capabilities, NodeState
from hive.host.db import HostDatabase


def test_host_db_lifecycle(tmp_path: Path) -> None:
    db_file = tmp_path / "test_hive.db"
    db = HostDatabase(db_file)
    assert db_file.exists()

    # 1. Session operations
    sess = db.create_or_load_session(
        session_id="sess_123",
        name="Test FL",
        description="Testing persistence",
        data_requirements="Data spec",
        package_digest="sha256:pkg",
        host_key_fingerprint="ed25519:fp",
        total_rounds=5,
        join_token="tok_abc",
    )
    assert sess["session_id"] == "sess_123"
    assert sess["current_round"] == 0

    db.update_session(current_round=1, status="running")
    updated_sess = db.get_session()
    assert updated_sess["current_round"] == 1
    assert updated_sess["status"] == "running"

    # 2. Node operations
    cap = Capabilities(os="linux", cores=4, has_docker=True, has_gpu=False)
    db.add_node("node_1", "AlphaNode", "secret_hash_1", cap, approved=True)
    db.add_node("node_2", "BetaNode", "secret_hash_2", cap, approved=False)

    nodes = db.list_nodes()
    assert len(nodes) == 2
    assert nodes[0]["nickname"] == "AlphaNode"
    assert nodes[1]["approved"] == 0

    db.update_node("node_2", approved=1, state=NodeState.ELIGIBLE.value)
    node_2 = db.get_node("node_2")
    assert node_2["approved"] == 1
    assert node_2["state"] == NodeState.ELIGIBLE.value

    # 3. Round operations
    db.create_round(1, deadline=1700000000.0, model_hash_in="sha256:in")
    r1 = db.get_round(1)
    assert r1["status"] == "open"
    assert r1["model_hash_in"] == "sha256:in"

    # 4. Claims
    claimed_1 = db.add_claim(1, "node_1")
    assert claimed_1 is True
    duplicate_claim = db.add_claim(1, "node_1")
    assert duplicate_claim is False
    assert db.list_claims(1) == ["node_1"]

    # 5. Updates
    db.add_update(
        round_id=1,
        node_id="node_1",
        meta_json='{"num_train_samples": 50}',
        norm=1.5,
        cosine=0.95,
        accepted=True,
        reject_reason=None,
        byte_count=1024,
    )
    updates = db.list_updates(1)
    assert len(updates) == 1
    assert updates[0]["accepted"] == 1
    assert updates[0]["bytes"] == 1024

    # 6. Close Round & Metrics
    db.close_round(1, status="closed", model_hash_out="sha256:out", aggregator_used="fedavg")
    closed_r = db.get_round(1)
    assert closed_r["status"] == "closed"
    assert closed_r["model_hash_out"] == "sha256:out"

    db.save_metrics(1, train_loss=0.5, eval_loss=0.4, eval_accuracy=0.88, participants=1, duration_s=12.5)
    metrics = db.list_metrics()
    assert len(metrics) == 1
    assert metrics[0]["eval_accuracy"] == 0.88

    # 7. Events
    db.add_event("INFO", "Round completed")
    events = db.list_events()
    assert len(events) == 1
    assert events[0]["message"] == "Round completed"

    # 8. Byte accounting
    db.record_bytes("node_1", "model_download", bytes_in=0, bytes_out=5000)
    db.record_bytes("node_1", "update_upload", bytes_in=1000, bytes_out=0)
    bytes_summary = db.get_byte_accounting_summary()
    assert len(bytes_summary) == 2