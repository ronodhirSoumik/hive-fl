"""
Integration test verifying end-to-end multi-node federated learning flow:
1 Host + 2 Contributors executing federated rounds.
"""

import asyncio
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from hive.client.agent import ClientAgent
from hive.client.config_store import ClientConfigStore
from hive.client.resources import ResourceController
from hive.core.config import HostConfig
from hive.core.contracts import BaseDataPreprocessor, ValidationResult
from hive.core.schemas import DataRequirements, Hyperparameters, Manifest, ResourcePolicy, RoundInfo
from hive.core.uri import build_hive_uri
from hive.host.app import create_host_apps
from hive.host.db import HostDatabase
from hive.host.orchestrator import Orchestrator
from hive.package.builder import build_package
from hive.package.signing import fingerprint, generate_keypair
import httpx


MODEL_CODE = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)
"""

PREPROCESS_CODE = """
import torch
from pathlib import Path
from torch.utils.data import TensorDataset
from hive.core.contracts import BaseDataPreprocessor, ValidationResult

class Preprocessor(BaseDataPreprocessor):
    def validate_local_directory(self, data_dir: str) -> ValidationResult:
        p = Path(data_dir)
        if not p.exists():
            return ValidationResult(ok=False, num_samples=0, message="Directory does not exist")
        # Count synthetic samples from a file or dummy logic
        count_file = p / "samples.txt"
        count = int(count_file.read_text().strip()) if count_file.exists() else 20
        return ValidationResult(ok=True, num_samples=count, message=f"Found {count} samples")

    def build_dataset(self, data_dir: str, split: str = "train", seed: int = 42) -> TensorDataset:
        torch.manual_seed(seed)
        x = torch.randn(20, 4)
        y = torch.randint(0, 2, (20,))
        return TensorDataset(x, y)
"""


@pytest.mark.asyncio
async def test_two_contributors_full_flow(tmp_path: Path) -> None:
    # 1. Build and sign package
    priv_key, pub_key = generate_keypair()
    host_fp = fingerprint(pub_key)
    pkg_path = tmp_path / "model.hivepkg"

    manifest = Manifest(
        package_name="test-fl-model",
        version="0.1.0",
        description="End-to-end integration test package",
        model_entrypoint="model.py:Model",
        preprocessor_entrypoint="preprocess.py:Preprocessor",
        loss="cross_entropy",
        metrics="accuracy",
        hyperparameters=Hyperparameters(local_epochs=1, batch_size=5, learning_rate=0.01),
        data_requirements=DataRequirements(description="Synthetic datasets", min_samples=10),
    )

    build_package(
        manifest=manifest,
        model_code=MODEL_CODE,
        preprocessor_code=PREPROCESS_CODE,
        requirements_text="torch\nnumpy",
        private_key=priv_key,
        out_path=pkg_path,
    )

    from hive.package.reader import inspect_package
    info = inspect_package(pkg_path)
    pkg_digest = info["content_digest"]

    # Initial weights matching Model
    initial_weights = {
        "fc.weight": torch.zeros(2, 4, dtype=torch.float32),
        "fc.bias": torch.zeros(2, dtype=torch.float32),
    }

    # 2. Host setup
    host_workdir = tmp_path / "host_workdir"
    host_workdir.mkdir(parents=True)
    cfg = HostConfig(
        workdir=host_workdir,
        min_clients=2,
        total_rounds=2,
        round_timeout_s=30,
        poll_interval_s=1,
    )
    db = HostDatabase(host_workdir / "hive.db")

    orchestrator = Orchestrator(
        config=cfg,
        db=db,
        package_digest=pkg_digest,
        host_key_fingerprint=host_fp,
        initial_weights=initial_weights,
    )

    join_token = "secret_e2e_token"
    orchestrator.db.create_or_load_session(
        session_id="sess_e2e_test",
        name="E2E Integration Test Session",
        description="Testing 2 contributors",
        data_requirements="Synthetic",
        package_digest=pkg_digest,
        host_key_fingerprint=host_fp,
        total_rounds=2,
        join_token=join_token,
    )
    orchestrator.state.join_token = join_token
    await orchestrator.start()

    public_app, _ = create_host_apps(cfg, orchestrator, package_path=pkg_path)

    # 3. Create 2 Contributors with separate workdirs and data
    transport = httpx.ASGITransport(app=public_app)

    async def setup_contributor(node_idx: int) -> tuple[ClientAgent, Path]:
        client_dir = tmp_path / f"client_{node_idx}"
        data_dir = tmp_path / f"data_{node_idx}"
        client_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        (data_dir / "samples.txt").write_text(f"{30 + node_idx * 10}")

        store = ClientConfigStore(client_dir)
        policy = ResourcePolicy(only_when_idle=False, pause_on_battery=False)
        res_ctrl = ResourceController(
            policy=policy,
            config_store=store,
            battery_provider=lambda: False,
        )

        http_c = httpx.AsyncClient(transport=transport, base_url="http://testserver")
        agent = ClientAgent(
            work_dir=client_dir,
            config_store=store,
            resource_controller=res_ctrl,
            sandbox_mode="process",
            http_client=http_c,
        )
        return agent, data_dir

    agent1, data_dir1 = await setup_contributor(1)
    agent2, data_dir2 = await setup_contributor(2)

    join_uri = build_hive_uri(
        authority="testserver",
        session_id="sess_e2e_test",
        join_token=join_token,
        fingerprint=host_fp,
        tls=False,
    )

    # Connect both
    agent1.start()
    agent2.start()

    await agent1.connect(join_uri, nickname="Contributor-Alpha")
    await agent2.connect(join_uri, nickname="Contributor-Beta")

    assert agent1.state == "AWAITING_CONSENT"
    assert agent2.state == "AWAITING_CONSENT"

    agent1.give_consent()
    agent2.give_consent()

    val1 = await agent1.validate_data(data_dir1)
    val2 = await agent2.validate_data(data_dir2)

    assert val1["ok"] is True
    assert val2["ok"] is True
    assert agent1.state == "READY"
    assert agent2.state == "READY"

    # Wait for the background loop to complete both rounds
    for _ in range(60):
        await asyncio.sleep(0.5)
        latest_r = orchestrator.db.get_latest_round()
        if latest_r and latest_r["round_id"] == 2 and latest_r["status"] == "closed":
            break

    # Check that both rounds executed and closed
    r1 = orchestrator.db.get_round(1)
    r2 = orchestrator.db.get_round(2)
    assert r1 is not None and r1["status"] == "closed"
    assert r2 is not None and r2["status"] == "closed"

    # Verify model checkpoints saved
    assert (orchestrator.checkpoints_dir / "round_1.safetensors").exists()
    assert (orchestrator.checkpoints_dir / "round_2.safetensors").exists()

    # Verify both contributors participated
    assert len(agent1.history) >= 1
    assert len(agent2.history) >= 1

    await agent1.leave()
    await agent2.leave()
    await orchestrator.stop()
