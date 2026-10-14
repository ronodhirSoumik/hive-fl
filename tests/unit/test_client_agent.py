"""
Unit and integration tests for ClientAgent state machine and host communication (T13).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
import torch
import torch.nn as nn

from hive.client.agent import ClientAgent
from hive.client.config_store import ClientConfigStore
from hive.client.resources import ResourceController
from hive.core.config import HostConfig
from hive.core.hashing import model_hash
from hive.core.schemas import DataRequirements, Hyperparameters, Manifest, ResourcePolicy, RoundInfo
from hive.core.uri import build_hive_uri
from hive.host.app import create_host_apps
from hive.host.db import HostDatabase
from hive.host.orchestrator import Orchestrator
from hive.package.builder import build_package
from hive.package.signing import fingerprint, generate_keypair


# Package Model Code
MODEL_SRC = """
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)
"""

# Package Preprocessor Code
PREPROC_SRC = """
import torch
from torch.utils.data import TensorDataset
from hive.core.contracts import BaseDataPreprocessor, ValidationResult

class LinearPreprocessor(BaseDataPreprocessor):
    def validate_local_directory(self, data_dir: str) -> ValidationResult:
        return ValidationResult(ok=True, num_samples=50, message="Validated 50 records")

    def build_dataset(self, data_dir: str, split: str = "train", seed: int = 42) -> TensorDataset:
        x = torch.randn(50, 4)
        y = torch.randint(0, 2, (50,))
        return TensorDataset(x, y)
"""


@pytest.fixture
def test_package_bundle(tmp_path: Path) -> tuple[Path, str, str, dict[str, torch.Tensor]]:
    """Builds and signs a valid .hivepkg for testing client agent interaction."""
    priv_key, pub_key = generate_keypair()
    fp = fingerprint(pub_key)

    manifest = Manifest(
        package_name="test-linear",
        version="1.0.0",
        description="Linear model for agent testing",
        model_entrypoint="model.py:LinearModel",
        preprocessor_entrypoint="preprocess.py:LinearPreprocessor",
        loss="cross_entropy",
        metrics="accuracy",
        hyperparameters=Hyperparameters(local_epochs=1, batch_size=10, learning_rate=0.01),
        data_requirements=DataRequirements(description="Synthetic", min_samples=10),
    )

    pkg_path = tmp_path / "linear.hivepkg"
    build_package(
        manifest=manifest,
        model_code=MODEL_SRC,
        preprocessor_code=PREPROC_SRC,
        requirements_text="torch\n",
        private_key=priv_key,
        out_path=pkg_path,
    )

    from hive.package.reader import inspect_package
    inspected = inspect_package(pkg_path)
    pkg_digest = inspected["content_digest"]

    initial_weights = {
        "fc.weight": torch.zeros(2, 4, dtype=torch.float32),
        "fc.bias": torch.zeros(2, dtype=torch.float32),
    }

    return pkg_path, pkg_digest, fp, initial_weights


@pytest.mark.asyncio
async def test_client_agent_full_lifecycle(tmp_path: Path, test_package_bundle: tuple[Path, str, str, dict[str, torch.Tensor]]) -> None:
    """
    Tests complete client lifecycle:
    connect -> await consent -> give consent -> validate data -> execute round -> upload -> leave.
    """
    pkg_path, pkg_digest, host_fp, initial_weights = test_package_bundle

    # 1. Setup Host Server
    host_workdir = tmp_path / "host_workdir"
    host_workdir.mkdir(parents=True)
    cfg = HostConfig(workdir=host_workdir, min_clients=1, max_clients_per_round=2, round_timeout_s=10.0)
    db = HostDatabase(host_workdir / "hive.db")

    orchestrator = Orchestrator(
        config=cfg,
        db=db,
        package_digest=pkg_digest,
        host_key_fingerprint=host_fp,
        initial_weights=initial_weights,
    )

    join_token = "secret_agent_token"
    orchestrator.db.create_or_load_session(
        session_id="sess_agent_test",
        name="Agent Lifecycle Test",
        description="Testing ClientAgent",
        data_requirements="Synthetic",
        package_digest=pkg_digest,
        host_key_fingerprint=host_fp,
        total_rounds=2,
        join_token=join_token,
    )
    orchestrator.state.join_token = join_token
    await orchestrator.start()

    public_app, _ = create_host_apps(cfg, orchestrator, package_path=pkg_path)

    # 2. Setup Client Agent with in-memory ASGI transport
    transport = httpx.ASGITransport(app=public_app)
    async_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")

    client_workdir = tmp_path / "client_workdir"
    config_store = ClientConfigStore(client_workdir)
    res_ctrl = ResourceController(
        policy=ResourcePolicy(only_when_idle=False, pause_on_battery=False),
        config_store=config_store,
    )

    agent = ClientAgent(
        work_dir=client_workdir,
        config_store=config_store,
        resource_controller=res_ctrl,
        sandbox_mode="process",
        http_client=async_client,
    )

    assert agent.state == "DISCONNECTED"

    # 3. Connect to Host
    join_uri = build_hive_uri(
        authority="testserver",
        session_id="sess_agent_test",
        join_token=join_token,
        fingerprint=host_fp,
        tls=False,
    )

    await agent.connect(join_uri, nickname="TestWorkerNode")

    # Agent should transition to AWAITING_CONSENT because digest is not yet approved
    assert agent.state == "AWAITING_CONSENT"
    assert agent.node_id is not None
    assert agent.package_manifest is not None
    assert agent.package_manifest.content_digest == pkg_digest

    # 4. Give User Consent
    data_dir = tmp_path / "local_data"
    data_dir.mkdir(parents=True)
    agent.data_dir = data_dir

    agent.give_consent()
    assert config_store.is_digest_consented(pkg_digest) is True

    # 5. Validate Local Data Directory
    val_res = await agent.validate_data(data_dir)
    assert val_res["ok"] is True
    assert val_res["num_samples"] == 50
    assert agent.state == "READY"

    # 6. Trigger round opening on Host
    await orchestrator._tick()
    latest_r = orchestrator.db.get_latest_round()
    assert latest_r is not None
    assert latest_r["status"] == "open"
    assert latest_r["round_id"] == 1

    # 7. Execute Round
    r_resp = await async_client.get(
        "/v1/round/current",
        headers={"Authorization": f"Bearer {agent.node_id}.{agent.node_secret}"},
    )
    assert r_resp.status_code == 200
    round_info = RoundInfo.model_validate(r_resp.json())
    assert round_info.status == "open"

    await agent._execute_round(round_info)

    # Check update was uploaded and accepted
    assert agent.bytes_uploaded > 0
    assert len(agent.history) == 1
    assert agent.history[0]["round_id"] == 1
    assert agent.history[0]["accepted"] is True
    assert agent.payload_preview is not None
    assert agent.payload_preview["tensor_count"] == 2
    assert agent.state == "READY"

    # Trigger orchestrator tick to close round
    await orchestrator._tick()
    closed_r = orchestrator.db.get_round(1)
    assert closed_r["status"] == "closed"

    # 8. Gracefully Leave Session
    await agent.leave()
    assert agent.state == "LEFT"
    assert config_store.load_node_credentials() is None
    assert not agent.job_dir.exists()
    assert not agent.extracted_dir.exists()

    await async_client.aclose()
    await orchestrator.stop()
