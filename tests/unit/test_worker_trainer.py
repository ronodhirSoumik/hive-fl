"""
Unit tests for Training Worker and Sandbox Runtime (T11).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

from hive.client.worker.trainer import run_worker_task
from hive.core.contracts import BaseDataPreprocessor, ValidationResult
from hive.core.schemas import DPPolicy, Manifest, WorkerConfig, WorkerResult
from hive.core.tensors import bytes_to_state_dict, state_dict_to_bytes


# Dummy Model for In-Worker Training
DUMMY_MODEL_CODE = """
import torch
import torch.nn as nn

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)
"""

# Dummy Preprocessor for In-Worker Training
DUMMY_PREPROCESSOR_CODE = """
import torch
from torch.utils.data import TensorDataset
from hive.core.contracts import BaseDataPreprocessor, ValidationResult

class DummyPreprocessor(BaseDataPreprocessor):
    def validate_local_directory(self, data_dir: str) -> ValidationResult:
        if "invalid" in data_dir:
            return ValidationResult(ok=False, num_samples=0, message="Directory marked invalid")
        return ValidationResult(ok=True, num_samples=40, message="Validated 40 samples successfully")

    def build_dataset(self, data_dir: str, split: str = "train", seed: int = 42) -> TensorDataset:
        x = torch.randn(40, 4)
        y = torch.randint(0, 2, (40,))
        return TensorDataset(x, y)
"""


@pytest.fixture
def mock_package_dir(tmp_path: Path) -> Path:
    """Creates an extracted mock package directory with model and preprocessor."""
    pkg = tmp_path / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "model.py").write_text(DUMMY_MODEL_CODE, encoding="utf-8")
    (pkg / "preprocess.py").write_text(DUMMY_PREPROCESSOR_CODE, encoding="utf-8")

    manifest = {
        "package_name": "dummy_pkg",
        "version": "0.1.0",
        "description": "Dummy package for worker testing",
        "model_entrypoint": "model.py:TinyClassifier",
        "preprocessor_entrypoint": "preprocess.py:DummyPreprocessor",
        "loss": "cross_entropy",
        "metrics": "accuracy",
        "hyperparameters": {
            "local_epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.01,
            "optimizer": "sgd",
            "eval_split": 0.2,
        },
        "data_requirements": {"description": "mock data", "min_samples": 5},
    }
    (pkg / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pkg


def test_worker_validate_success(tmp_path: Path, mock_package_dir: Path) -> None:
    """Worker validate task reports sample count, writes local_message.txt, and omits free text from result.json."""
    job_dir = tmp_path / "job_val"
    in_dir = job_dir / "in"
    in_dir.mkdir(parents=True)

    # Copy pkg
    import shutil
    shutil.copytree(mock_package_dir, in_dir / "pkg")

    cfg = {
        "task": "validate",
        "data_dir": str(tmp_path / "valid_data"),
        "max_cpu_percent": 100,
    }
    (in_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    run_worker_task(job_dir)

    res_file = job_dir / "out" / "result.json"
    assert res_file.exists()
    res = WorkerResult.model_validate_json(res_file.read_text(encoding="utf-8"))
    assert res.status == "ok"
    assert res.num_train_samples == 40
    assert res.error_message is None

    # Verify local-only message file
    local_msg_file = job_dir / "out" / "local_message.txt"
    assert local_msg_file.exists()
    assert "40 samples" in local_msg_file.read_text(encoding="utf-8")


def test_worker_validate_failure(tmp_path: Path, mock_package_dir: Path) -> None:
    """Worker validate failure writes error code without leaking free-text error message into result.json."""
    job_dir = tmp_path / "job_val_fail"
    in_dir = job_dir / "in"
    in_dir.mkdir(parents=True)

    import shutil
    shutil.copytree(mock_package_dir, in_dir / "pkg")

    cfg = {
        "task": "validate",
        "data_dir": str(tmp_path / "invalid_data"),
        "max_cpu_percent": 100,
    }
    (in_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    run_worker_task(job_dir)

    res_file = job_dir / "out" / "result.json"
    assert res_file.exists()
    res = WorkerResult.model_validate_json(res_file.read_text(encoding="utf-8"))
    assert res.status == "error"
    assert res.error_code == "ERR_DATA_VALIDATION_FAILED"
    assert res.error_message is None

    local_msg = (job_dir / "out" / "local_message.txt").read_text(encoding="utf-8")
    assert "invalid" in local_msg


def test_worker_train_task(tmp_path: Path, mock_package_dir: Path) -> None:
    """Worker train task trains model, outputs safetensors delta, progress records, and eval metrics."""
    job_dir = tmp_path / "job_train"
    in_dir = job_dir / "in"
    in_dir.mkdir(parents=True)

    import shutil
    shutil.copytree(mock_package_dir, in_dir / "pkg")

    # Initial global weights
    initial_weights = {
        "fc.weight": torch.zeros(2, 4, dtype=torch.float32),
        "fc.bias": torch.zeros(2, dtype=torch.float32),
    }
    (in_dir / "global.safetensors").write_bytes(state_dict_to_bytes(initial_weights))

    cfg = {
        "task": "train",
        "data_dir": str(tmp_path / "valid_data"),
        "max_cpu_percent": 100,
        "seed": 123,
    }
    (in_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    run_worker_task(job_dir)

    res_file = job_dir / "out" / "result.json"
    assert res_file.exists()
    res = WorkerResult.model_validate_json(res_file.read_text(encoding="utf-8"))
    assert res.status == "ok"
    assert res.num_train_samples > 0
    assert "accuracy" in res.eval_metrics

    # Check update delta exists and has non-zero weights
    update_file = job_dir / "out" / "update.safetensors"
    assert update_file.exists()
    delta = bytes_to_state_dict(update_file.read_bytes())
    assert "fc.weight" in delta
    assert delta["fc.weight"].shape == (2, 4)
    assert not torch.allclose(delta["fc.weight"], torch.zeros_like(delta["fc.weight"]))

    # Check progress records
    prog_file = job_dir / "out" / "progress.jsonl"
    assert prog_file.exists()
    lines = [json.loads(l) for l in prog_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) > 0
    assert lines[0]["phase"] == "train"


def test_worker_train_with_dp(tmp_path: Path, mock_package_dir: Path) -> None:
    """Worker train task with DP enabled clips delta and adds Gaussian noise."""
    job_dir = tmp_path / "job_dp"
    in_dir = job_dir / "in"
    in_dir.mkdir(parents=True)

    import shutil
    shutil.copytree(mock_package_dir, in_dir / "pkg")

    initial_weights = {
        "fc.weight": torch.zeros(2, 4, dtype=torch.float32),
        "fc.bias": torch.zeros(2, dtype=torch.float32),
    }
    (in_dir / "global.safetensors").write_bytes(state_dict_to_bytes(initial_weights))

    cfg = {
        "task": "train",
        "data_dir": str(tmp_path / "valid_data"),
        "max_cpu_percent": 100,
        "dp": {
            "enabled": True,
            "clip_norm": 0.5,
            "noise_multiplier": 0.1,
        },
        "seed": 42,
    }
    (in_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    run_worker_task(job_dir)

    res = WorkerResult.model_validate_json((job_dir / "out" / "result.json").read_text(encoding="utf-8"))
    assert res.status == "ok"

    delta = bytes_to_state_dict((job_dir / "out" / "update.safetensors").read_bytes())
    assert "fc.weight" in delta


def test_worker_exception_handling(tmp_path: Path) -> None:
    """Worker crashes write trace to worker.log and produce clean error code in result.json."""
    job_dir = tmp_path / "job_crash"
    in_dir = job_dir / "in"
    in_dir.mkdir(parents=True)

    # Missing manifest.json triggers exception
    cfg = {"task": "train", "data_dir": "xyz"}
    (in_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    run_worker_task(job_dir)

    res_file = job_dir / "out" / "result.json"
    assert res_file.exists()
    res = WorkerResult.model_validate_json(res_file.read_text(encoding="utf-8"))
    assert res.status == "error"
    assert res.error_code == "ERR_WORKER_EXECUTION"
    assert res.error_message is None

    log_file = job_dir / "out" / "worker.log"
    assert log_file.exists()
    assert "FileNotFoundError" in log_file.read_text(encoding="utf-8")
