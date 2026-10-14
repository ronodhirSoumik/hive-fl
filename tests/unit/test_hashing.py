"""
Unit tests for deterministic model state hashing and file hashing.
"""

from pathlib import Path
import torch
from hive.core.hashing import bytes_sha256, file_sha256, model_hash


def test_model_hash_determinism() -> None:
    t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    t2 = torch.tensor([0.5, -0.5], dtype=torch.float32)

    sd1 = {"layer1.weight": t1, "layer2.bias": t2}
    sd2 = {"layer2.bias": t2.clone(), "layer1.weight": t1.clone()}

    h1 = model_hash(sd1)
    h2 = model_hash(sd2)

    assert h1.startswith("sha256:")
    assert len(h1) == 7 + 64
    assert h1 == h2  # Key insertion order does not affect hash


def test_model_hash_sensitivity() -> None:
    t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    sd = {"layer1.weight": t1}
    base_hash = model_hash(sd)

    # 1. Alter single float value
    t1_altered = torch.tensor([[1.0, 2.0], [3.0, 4.0001]], dtype=torch.float32)
    assert model_hash({"layer1.weight": t1_altered}) != base_hash

    # 2. Alter key name
    assert model_hash({"layer2.weight": t1}) != base_hash

    # 3. Alter tensor shape
    t1_reshaped = t1.reshape(-1)
    assert model_hash({"layer1.weight": t1_reshaped}) != base_hash


def test_file_and_bytes_hashing(tmp_path: Path) -> None:
    data = b"Hello Hive Federated Learning"
    expected = bytes_sha256(data)

    test_file = tmp_path / "test.bin"
    test_file.write_bytes(data)
    file_h = file_sha256(test_file)

    assert expected == file_h
    assert expected.startswith("sha256:")