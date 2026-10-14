"""
Unit tests for Local Contributor Web API (T13).
Verifies UI token authentication, anti-CSRF, DNS-rebinding security, and local endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
from starlette.testclient import TestClient

from hive.client.agent import ClientAgent
from hive.client.config_store import ClientConfigStore
from hive.client.local_api import create_local_app
from hive.client.resources import ResourceController
from hive.core.schemas import ResourcePolicy


@pytest.fixture
def local_app_setup(tmp_path: Path) -> tuple[TestClient, ClientAgent, str]:
    workdir = tmp_path / "client_app_test"
    store = ClientConfigStore(workdir)
    res_ctrl = ResourceController(config_store=store)
    agent = ClientAgent(work_dir=workdir, config_store=store, resource_controller=res_ctrl)

    ui_token = "secret_ui_token_999"
    app = create_local_app(agent, ui_token=ui_token)
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    return client, agent, ui_token


def test_local_api_auth_required(local_app_setup: tuple[TestClient, ClientAgent, str]) -> None:
    """Missing or invalid token returns 401."""
    client, _, _ = local_app_setup

    # 1. No token
    res = client.get("/local/status")
    assert res.status_code == 401
    assert res.json()["detail"]["code"] == "ERR_UI_AUTH_FAILED"

    # 2. Wrong token
    res = client.get("/local/status", headers={"X-Client-Token": "bad_token"})
    assert res.status_code == 401


def test_local_api_dns_rebinding_defense(local_app_setup: tuple[TestClient, ClientAgent, str]) -> None:
    """Requests with external Host headers are blocked to prevent DNS rebinding."""
    client, _, token = local_app_setup
    headers = {"X-Client-Token": token, "Host": "attacker.com"}
    res = client.get("/local/status", headers=headers)
    assert res.status_code == 403
    assert res.json()["detail"]["code"] == "ERR_DNS_REBINDING"


def test_local_api_csrf_origin_defense(local_app_setup: tuple[TestClient, ClientAgent, str]) -> None:
    """Requests with non-localhost Origin headers are blocked to prevent CSRF."""
    client, _, token = local_app_setup
    headers = {"X-Client-Token": token, "Origin": "http://evil-tracker.org"}
    res = client.get("/local/status", headers=headers)
    assert res.status_code == 403
    assert res.json()["detail"]["code"] == "ERR_INVALID_ORIGIN"


def test_local_api_status_and_resources(local_app_setup: tuple[TestClient, ClientAgent, str]) -> None:
    """Verifies /local/status, /local/resources (GET/PUT), and pause/resume."""
    client, agent, token = local_app_setup
    auth = {"X-Client-Token": token}

    # 1. Status
    res = client.get("/local/status", headers=auth)
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "DISCONNECTED"
    assert "can_train" in data

    # 2. Get Resources
    res_r = client.get("/local/resources", headers=auth)
    assert res_r.status_code == 200
    assert "max_cpu_percent" in res_r.json()

    # 3. Update Resources
    new_pol = {"max_cpu_percent": 80, "only_when_idle": False, "max_rounds": 10}
    res_put = client.put("/local/resources", json=new_pol, headers=auth)
    assert res_put.status_code == 200
    assert res_put.json()["max_cpu_percent"] == 80
    assert agent.resource_controller.policy.max_cpu_percent == 80

    # 4. Pause and Resume
    res_pause = client.post("/local/pause", headers=auth)
    assert res_pause.status_code == 200
    assert agent.resource_controller.is_user_paused is True
    assert agent.state == "PAUSED_LOCAL"

    res_resume = client.post("/local/resume", headers=auth)
    assert res_resume.status_code == 200
    assert agent.resource_controller.is_user_paused is False


def test_local_api_fs_browser(local_app_setup: tuple[TestClient, ClientAgent, str], tmp_path: Path) -> None:
    """Verifies folder browser security (only directories returned, no files)."""
    client, _, token = local_app_setup
    auth = {"X-Client-Token": token}

    # Create test directory structure
    folder_a = tmp_path / "folder_a"
    folder_a.mkdir()
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("classified data")

    res = client.get(f"/local/fs?path={tmp_path}", headers=auth)
    assert res.status_code == 200
    body = res.json()
    dirs = [d["name"] for d in body["directories"]]
    assert "folder_a" in dirs
    assert "secret.txt" not in dirs  # Files must never be listed


def test_local_api_payload_preview(local_app_setup: tuple[TestClient, ClientAgent, str]) -> None:
    """Verifies payload transparency statements and limitations notice."""
    client, agent, token = local_app_setup
    auth = {"X-Client-Token": token}

    agent.payload_preview = {
        "tensor_count": 3,
        "bytes": 1024,
        "sha256": "abc123def456",
        "tensors": [{"name": "layer.weight", "shape": [10, 10], "dtype": "torch.float32"}],
    }

    res = client.get("/local/payload-preview", headers=auth)
    assert res.status_code == 200
    data = res.json()
    assert data["preview"]["tensor_count"] == 3
    assert "Raw data files never leave your device" in data["statement"]
    assert "limitations" in data
