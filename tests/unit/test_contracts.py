"""
Unit tests for core contracts and abstract base classes.
"""

from pathlib import Path
from typing import Any, Literal
import pytest
from hive.core.contracts import (
    BaseDataPreprocessor,
    ValidationResult,
    validate_model_contract,
)


class DummyPreprocessor(BaseDataPreprocessor):
    def validate_local_directory(self, data_dir: Path | str) -> ValidationResult:
        p = Path(data_dir)
        if p.exists():
            return ValidationResult(ok=True, num_samples=100, message="OK")
        return ValidationResult(ok=False, num_samples=0, message="Missing directory")

    def build_dataset(
        self,
        data_dir: Path | str,
        split: Literal["train", "eval"],
        seed: int = 42,
    ) -> Any:
        return [(split, seed)]


def test_validation_result() -> None:
    res = ValidationResult(ok=True, num_samples=42, message="All good")
    assert res.ok is True
    assert res.num_samples == 42
    assert res.message == "All good"


def test_preprocessor_contract(tmp_path: Path) -> None:
    prep = DummyPreprocessor()
    assert prep.validate_local_directory(tmp_path).ok is True
    assert prep.validate_local_directory(tmp_path / "nonexistent").ok is False
    ds = prep.build_dataset(tmp_path, split="train", seed=123)
    assert ds == [("train", 123)]


def test_model_contract_validation() -> None:
    # Valid configurations
    validate_model_contract(loss="cross_entropy", metrics="accuracy")
    validate_model_contract(loss="mse", metrics="none")
    validate_model_contract(loss="bce_logits", metrics="accuracy")

    # Invalid loss
    with pytest.raises(ValueError, match="Unsupported loss"):
        validate_model_contract(loss="hinge_loss", metrics="accuracy")

    # Invalid metric
    with pytest.raises(ValueError, match="Unsupported metric"):
        validate_model_contract(loss="cross_entropy", metrics="f1_score")