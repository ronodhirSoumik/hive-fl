"""
Unit tests for Public FL API (/v1) endpoints using FastAPI TestClient.
"""

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient
import torch

from hive.core.config import HostConfig
from hive.core.tensors import state_dict_to_bytes
from hive.host.app import create_host_apps
from hive.host.db import HostDatabase
from hive.host.orchestrator import Orchestrator


@pytest.fixture
def client_setup(tmp_path: Path) -> tuple[TestClient, Orchestrator, str]:
    cfg = HostConfig(workdir=tmp_path / "host")
    db = HostDatabase(tmp_path / "host" / "hive.db")
    initial_weights = {"w": torch.tensor([1.0, 2.0], dtype=torch.float32)}

    dummy_pkg = tmp_path / "dummy.hivepkg"
    dummy_pkg.write_bytes(b"dummy_pkg_data")

    orchestrator = Orchestrator(
        config=cfg,
        db=db,
        package_digest="sha256:dummy",
        host_key_fingerprint="ed25519:dummy",
        initial_weights=initial_weights,
    )
    # Initialize session in DB
    orchestrator.db.create_or_load_session(
        session_id="sess_test",
        name="Test Session",
        description="Public API Tests",
        data_requirements="None",
        package_digest="sha256:dummy",
        host_key_fingerprint="ed25519:dummy",
        total_rounds=2,
        join_token="secret_join_token_123",
    )
    orchestrator.state.join_token = "secret_join_token_123"

    public_app, _ = create_host_apps(cfg, orchestrator, package_path=dummy_pkg)
    client = TestClient(public_app)
    return client, orchestrator, "secret_join_token_123"


def test_public_api_session_and_join(client_setup: tuple[TestClient, Orchestrator, str]) -> None:
    client, orchestrator, join_token = client_setup

    # 1. Health check
    res = client.get("/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # 2. GET /v1/session without join token -> 401
    res_no_token = client.get("/v1/session")
    assert res_no_token.status_code == 401

    # 3. GET /v1/session with join token -> 200
    res_sess = client.get("/v1/session", headers={"X-Join-Token": join_token})
    assert res_sess.status_code == 200
    sess_info = res_sess.json()
    assert sess_info["session_id"] == "sess_test"
    assert sess_info["package_digest"] == "sha256:dummy"

    # 4. POST /v1/join with join token -> 200
    join_payload = {
        "nickname": "TesterNode",
        "capabilities": {
            "os": "windows",
            "cores": 4,
            "has_docker": True,
            "has_gpu": False,
        },
    }
    res_join = client.post("/v1/join", json=join_payload, headers={"X-Join-Token": join_token})
    assert res_join.status_code == 200
    join_data = res_join.json()
    assert "node_id" in join_data
    assert "node_secret" in join_data
    assert join_data["approval"] == "approved"

    node_id = join_data["node_id"]
    secret = join_data["node_secret"]
    auth_header = {"Authorization": f"Bearer {node_id}.{secret}"}

    # 5. GET /v1/package -> 200
    res_pkg = client.get("/v1/package", headers=auth_header)
    assert res_pkg.status_code == 200
    assert res_pkg.content == b"dummy_pkg_data"

    # 6. POST /v1/ready -> 200
    res_ready = client.post("/v1/ready", json={"data_ok": True, "num_samples": 120}, headers=auth_header)
    assert res_ready.status_code == 200
    assert res_ready.json()["status"] == "eligible"

    # 7. POST /v1/heartbeat -> 200
    res_hb = client.post("/v1/heartbeat", json={"state": "waiting"}, headers=auth_header)
    assert res_hb.status_code == 200

    # 8. POST /v1/leave -> 200
    res_leave = client.post("/v1/leave", headers=auth_header)
    assert res_leave.status_code == 200
    assert res_leave.json()["left"] is True


def test_public_api_multipart_validation(client_setup: tuple[TestClient, Orchestrator, str]) -> None:
    client, orchestrator, join_token = client_setup

    # Register node
    join_res = client.post(
        "/v1/join",
        json={"nickname": "NodeA", "capabilities": {"os": "linux", "cores": 2, "has_docker": False, "has_gpu": False}},
        headers={"X-Join-Token": join_token},
    ).json()
    auth = {"Authorization": f"Bearer {join_res['node_id']}.{join_res['node_secret']}"}

    # Attempt multipart upload with extra disallowed parts -> must be rejected (400)
    files = {
        "meta": (None, json.dumps({"round_id": 1, "base_model_hash": "sha256:abc", "num_train_samples": 10, "train_loss": 0.5, "eval_loss": 0.5})),
        "delta": ("update.safetensors", b"dummy_safetensors_data", "application/octet-stream"),
        "raw_audio_leak.wav": ("audio.wav", b"RIFF....", "audio/wav"),  # Forbidden!
    }
    res = client.post("/v1/round/1/update", files=files, headers=auth)
    assert res.status_code == 400
    assert "extra forbidden parts" in res.json()["detail"]["message"].lower()