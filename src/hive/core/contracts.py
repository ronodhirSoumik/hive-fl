"""
Core contracts and abstract interfaces for Hive.
Defines interfaces for data preprocessing, validation and model contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Set

# Allowed losses and metrics per specification
ALLOWED_LOSSES: Set[str] = {"cross_entropy", "mse", "bce_logits"}
ALLOWED_METRICS: Set[str] = {"accuracy", "none"}


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a local contributor dataset."""
    ok: bool
    num_samples: int
    message: str = ""


class BaseDataPreprocessor(ABC):
    """
    Abstract base class for package data preprocessors.
    Package authors implement this class inside preprocess.py.
    """

    @abstractmethod
    def validate_local_directory(self, data_dir: Path | str) -> ValidationResult:
        """
        Inspects the local dataset folder without loading all data into memory.
        Returns a ValidationResult indicating whether the data meets requirements
        and the number of detected usable samples.
        """
        pass

    @abstractmethod
    def build_dataset(
        self,
        data_dir: Path | str,
        split: Literal["train", "eval"],
        seed: int = 42,
    ) -> Any:
        """
        Constructs a torch.utils.data.Dataset yielding (input_tensor, target_tensor).
        The split should be deterministically partitioned using the provided seed.
        """
        pass


def validate_model_contract(loss: str, metrics: str) -> None:
    """Verifies loss and metric contracts against allowed specifications."""
    if loss not in ALLOWED_LOSSES:
        raise ValueError(f"Unsupported loss '{loss}'. Must be one of {sorted(ALLOWED_LOSSES)}")
    if metrics not in ALLOWED_METRICS:
        raise ValueError(f"Unsupported metric '{metrics}'. Must be one of {sorted(ALLOWED_METRICS)}")