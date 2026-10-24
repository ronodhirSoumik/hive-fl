from pathlib import Path
from typing import Literal
import torch
from torch.utils.data import TensorDataset
from hive.core.contracts import BaseDataPreprocessor, ValidationResult

class Preprocessor(BaseDataPreprocessor):
    def validate_local_directory(self, data_dir: Path | str) -> ValidationResult:
        p = Path(data_dir)
        if not p.exists() or not p.is_dir():
            return ValidationResult(ok=False, num_samples=0, message=f"Directory {data_dir} not found")
        data_file = p / "data.pt"
        if not data_file.exists():
            return ValidationResult(ok=False, num_samples=0, message="data.pt file missing")
        try:
            d = torch.load(data_file)
            n = len(d["x"])
            return ValidationResult(ok=True, num_samples=n, message=f"Validated {n} local samples successfully")
        except Exception as e:
            return ValidationResult(ok=False, num_samples=0, message=f"Error reading data.pt: {e}")

    def build_dataset(
        self,
        data_dir: Path | str,
        split: Literal["train", "eval"],
        seed: int = 42,
    ) -> TensorDataset:
        d = torch.load(Path(data_dir) / "data.pt")
        x, y = d["x"], d["y"]
        total = len(x)
        split_idx = int(total * 0.8)
        torch.manual_seed(seed)
        perm = torch.randperm(total)
        x, y = x[perm], y[perm]
        if split == "train":
            return TensorDataset(x[:split_idx], y[:split_idx])
        else:
            return TensorDataset(x[split_idx:], y[split_idx:])
