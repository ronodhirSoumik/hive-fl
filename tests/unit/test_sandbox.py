"""
Unit tests for Client Sandbox Executors and Job-Directory Protocol (T10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pytest
import torch

from hive.client.sandbox.base import MAX_UPDATE_SIZE_BYTES, SandboxExecutor
from hive.client.sandbox.docker_exec import DockerExecutor
from hive.client.sandbox.process_exec import ProcessExecutor
from hive.core.errors import SandboxError
from hive.core.schemas import WorkerConfig, WorkerResult
from hive.core.tensors import state_dict_to_bytes


class DummyExecutor(SandboxExecutor):
    """Concrete executor for testing base Job-Directory protocol operations."""

    def __init__(self, job_dir: Path | str) -> None:
        super().__init__(job_dir, isolation="test")
        self.suspended = False
        self.resumed = False
        self.killed = False

    def run(self, timeout_s: float = 600.0) -> WorkerResult:
        res = WorkerResult(status="ok", num_train_samples=50, duration_s=1.2)
        (self.out_dir / "result.json").write_text(json.dumps(res.model_dump()), encoding="utf-8")
        return res

    def suspend(self) -> None:
        self.suspended = True

    def resume(self) -> None:
        self.resumed = True

    def kill(self) -> None:
        self.killed = True


def test_job_directory_layout(tmp_path: Path) -> None:
    """Verifies that prepare() builds the canonical job directory structure."""
    job_dir = tmp_path / "job"
    pkg_dir = tmp_path / "source_pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "manifest.json").write_text('{"package_name": "test"}', encoding="utf-8")
    (pkg_dir / "model.py").write_text("class M: pass", encoding="utf-8")

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    weights = {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}

    cfg = WorkerConfig(task="train", hyperparameters={"lr": 0.01})
    executor = DummyExecutor(job_dir)
    executor.prepare(cfg, pkg_source_dir=pkg_dir, data_dir=data_dir, global_weights=weights)

    # Check structure
    assert (job_dir / "in" / "config.json").exists()
    assert (job_dir / "in" / "global.safetensors").exists()
    assert (job_dir / "in" / "pkg" / "manifest.json").exists()
    assert (job_dir / "in" / "pkg" / "model.py").exists()
    assert (job_dir / "out").exists()

    cfg_read = json.loads((job_dir / "in" / "config.json").read_text(encoding="utf-8"))
    assert cfg_read["task"] == "train"
    assert cfg_read["data_dir"] == str(data_dir.resolve())


def test_progress_stream(tmp_path: Path) -> None:
    """Verifies incremental streaming of worker progress records."""
    executor = DummyExecutor(tmp_path / "job")
    executor.out_dir.mkdir(parents=True, exist_ok=True)
    prog_file = executor.out_dir / "progress.jsonl"

    # Initially empty
    assert executor.progress_stream() == []

    # Write first record
    prog_file.write_text(json.dumps({"epoch": 1, "loss": 0.5}) + "\n", encoding="utf-8")
    r1 = executor.progress_stream()
    assert len(r1) == 1
    assert r1[0]["epoch"] == 1

    # Second read without new data yields nothing
    assert executor.progress_stream() == []

    # Append second record
    with open(prog_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": 2, "loss": 0.3}) + "\n")

    r2 = executor.progress_stream()
    assert len(r2) == 1
    assert r2[0]["epoch"] == 2


def test_parse_result_missing_result_json(tmp_path: Path) -> None:
    """Missing result.json must raise SandboxError."""
    executor = DummyExecutor(tmp_path / "job")
    executor.out_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(SandboxError) as exc:
        executor.parse_result()
    assert exc.value.code == "ERR_WORKER_NO_RESULT"


def test_parse_result_malformed_result_json(tmp_path: Path) -> None:
    """Corrupted JSON in result.json must raise SandboxError."""
    executor = DummyExecutor(tmp_path / "job")
    executor.out_dir.mkdir(parents=True, exist_ok=True)
    (executor.out_dir / "result.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SandboxError) as exc:
        executor.parse_result()
    assert exc.value.code == "ERR_WORKER_RESULT_MALFORMED"


def test_parse_result_nan_inf_rejection(tmp_path: Path) -> None:
    """Updates with NaN or Inf values must be rejected."""
    executor = DummyExecutor(tmp_path / "job")
    executor.out_dir.mkdir(parents=True, exist_ok=True)

    res = WorkerResult(status="ok", num_train_samples=10)
    (executor.out_dir / "result.json").write_text(json.dumps(res.model_dump()), encoding="utf-8")

    nan_weights = {"w": torch.tensor([float("nan"), 1.0], dtype=torch.float32)}
    (executor.out_dir / "update.safetensors").write_bytes(state_dict_to_bytes(nan_weights))

    with pytest.raises(SandboxError) as exc:
        executor.parse_result()
    assert exc.value.code == "ERR_UPDATE_NAN_INF"


def test_parse_result_template_validation(tmp_path: Path) -> None:
    """Updates that don't match global template shapes must be rejected."""
    executor = DummyExecutor(tmp_path / "job")
    executor.out_dir.mkdir(parents=True, exist_ok=True)

    res = WorkerResult(status="ok", num_train_samples=10)
    (executor.out_dir / "result.json").write_text(json.dumps(res.model_dump()), encoding="utf-8")

    delta_weights = {"w": torch.tensor([0.1, 0.2], dtype=torch.float32)}
    (executor.out_dir / "update.safetensors").write_bytes(state_dict_to_bytes(delta_weights))

    # Template with different shape
    template = {"w": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)}
    with pytest.raises(SandboxError):
        executor.parse_result(template=template)

    # Template with matching shape succeeds
    good_template = {"w": torch.tensor([0.0, 0.0], dtype=torch.float32)}
    r, delta_dict, delta_bytes = executor.parse_result(template=good_template)
    assert r.status == "ok"
    assert delta_dict is not None
    assert "w" in delta_dict
    assert delta_bytes is not None


def test_docker_executor_unavailable_handling(tmp_path: Path) -> None:
    """DockerExecutor handles unreachable daemon or missing SDK gracefully."""
    exec_docker = DockerExecutor(tmp_path / "job")
    # If Docker daemon is not running, run() must raise SandboxError
    try:
        import docker
        client = docker.from_env()
        client.ping()
    except Exception:
        with pytest.raises(SandboxError):
            exec_docker.run(timeout_s=5.0)
