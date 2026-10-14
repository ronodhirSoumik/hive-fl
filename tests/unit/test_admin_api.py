"""
Unit tests for Admin API (/admin/api), privacy indicators, and model weight verification.
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient
import torch

from hive.core.config import HostConfig
from hive.core.hashing import model_hash
from hive.core.tensors import state_dict_to_bytes
from hive.host.app import create_host_apps
from hive.host.db import HostDatabase
from hive.host.orchestrator import Orchestrator


@pytest.fixture
def admin_setup(tmp_path: Path) -> tuple[TestClient, Orchestrator, str]:
    cfg = HostConfig(workdir=tmp_path / "host")
    db = HostDatabase(tmp_path / "host" / "hive.db")
    initial_weights = {"fc.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}

    orchestrator = Orchestrator(
        config=cfg,
        db=db,
        package_digest="sha256:pkg",
        host_key_fingerprint="ed25519:fp",
        initial_weights=initial_weights,
    )
    admin_token = "secret_admin_token_xyz"
    orchestrator.state.admin_token = admin_token
    orchestrator.db.create_or_load_session(
        session_id="sess_admin_test",
        name="Admin Test Session",
        description="Testing Admin Endpoints",
        data_requirements="Data",
        package_digest="sha256:pkg",
        host_key_fingerprint="ed25519:fp",
        total_rounds=3,
        join_token="join_tok",
    )

    _, admin_app = create_host_apps(cfg, orchestrator, package_path=None)
    client = TestClient(admin_app)
    return client, orchestrator, admin_token


def test_admin_auth_and_summary(admin_setup: tuple[TestClient, Orchestrator, str]) -> None:
    client, orchestrator, admin_token = admin_setup

    # 1. Access without token -> 401
    res_unauth = client.get("/admin/api/summary")
    assert res_unauth.status_code == 401

    auth_header = {"X-Admin-Token": admin_token}

    # 2. Access with token -> 200
    res = client.get("/admin/api/summary", headers=auth_header)
    assert res.status_code == 200
    summary = res.json()
    assert summary["session"]["session_id"] == "sess_admin_test"
    assert "active_nodes" in summary


def test_admin_privacy_indicators(admin_setup: tuple[TestClient, Orchestrator, str]) -> None:
    client, orchestrator, admin_token = admin_setup
    auth_header = {"X-Admin-Token": admin_token}

    res = client.get("/admin/api/privacy", headers=auth_header)
    assert res.status_code == 200
    privacy = res.json()

    # All 7 indicators present
    assert "transport" in privacy
    assert "raw_data_guard" in privacy
    assert "update_only_ratio" in privacy
    assert "differential_privacy" in privacy
    assert "anonymity_floor" in privacy
    assert "poisoning_defenses" in privacy
    assert "limitations" in privacy
    assert "Differential Privacy epsilon is an estimated upper-bound" in privacy["limitations"]


def test_admin_nodes_metrics_no_ips(admin_setup: tuple[TestClient, Orchestrator, str]) -> None:
    client, orchestrator, admin_token = admin_setup
    auth_header = {"X-Admin-Token": admin_token}

    # Register sample node
    orchestrator.db.add_node(
        "node_sample",
        nickname="Bob",
        secret_hash="hash",
        capabilities={"os": "linux", "cores": 4, "has_docker": True, "has_gpu": False},
    )

    res = client.get("/admin/api/nodes", headers=auth_header)
    assert res.status_code == 200
    nodes_list = res.json()
    assert len(nodes_list) >= 1

    for n in nodes_list:
        assert "nickname" in n
        assert "rounds_contributed" in n
        # Must NOT leak or contain IP addresses
        assert "ip" not in n
        assert "ip_address" not in n


def test_admin_session_controls(admin_setup: tuple[TestClient, Orchestrator, str]) -> None:
    client, orchestrator, admin_token = admin_setup
    auth_header = {"X-Admin-Token": admin_token}

    # Pause
    res_pause = client.post("/admin/api/session/pause", headers=auth_header)
    assert res_pause.status_code == 200
    assert res_pause.json()["status"] == "paused"
    assert orchestrator.state.status.value == "paused"

    # Resume
    res_resume = client.post("/admin/api/session/resume", headers=auth_header)
    assert res_resume.status_code == 200
    assert res_resume.json()["status"] == "running"
    assert orchestrator.state.status.value == "running"

    # Rotate join token
    old_tok = orchestrator.state.join_token
    res_rotate = client.post("/admin/api/session/rotate-token", headers=auth_header)
    assert res_rotate.status_code == 200
    new_tok = res_rotate.json()["join_token"]
    assert new_tok != old_tok


def test_model_weight_verification_tamper_detection(admin_setup: tuple[TestClient, Orchestrator, str], tmp_path: Path) -> None:
    client, orchestrator, admin_token = admin_setup
    auth_header = {"X-Admin-Token": admin_token}

    # Create dummy checkpoint on disk and record in DB
    sd = {"w": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    expected_hash = model_hash(sd)
    ckpt_file = orchestrator.checkpoints_dir / "round_1.safetensors"
    ckpt_file.write_bytes(state_dict_to_bytes(sd))

    orchestrator.db.create_round(1, deadline=1700000000.0, model_hash_in="sha256:prev")
    orchestrator.db.close_round(1, status="closed", model_hash_out=expected_hash)

    # 1. Verification of untampered file -> PASS
    res = client.post("/model/verify?round_id=1", headers=auth_header)
    assert res.status_code == 200
    ver_data = res.json()
    assert ver_data["verified"] is True
    assert ver_data["match"] is True
    assert ver_data["tampered"] is False
    assert ver_data["computed_model_hash"] == expected_hash

    # 2. Tamper with checkpoint file on disk (alter weights)
    sd_tampered = {"w": torch.tensor([1.0, 2.0001], dtype=torch.float32)}
    ckpt_file.write_bytes(state_dict_to_bytes(sd_tampered))

    # 3. Verification of tampered file -> DETECTED & FAILS!
    res_tampered = client.post("/model/verify?round_id=1", headers=auth_header)
    assert res_tampered.status_code == 200
    ver_tampered = res_tampered.json()
    assert ver_tampered["verified"] is True
    assert ver_tampered["match"] is False
    assert ver_tampered["tampered"] is True