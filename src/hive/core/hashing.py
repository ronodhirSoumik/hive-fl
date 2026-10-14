"""
Cryptographic and model state hashing utilities for Hive.
Provides deterministic SHA-256 calculation over PyTorch state dicts, files, and bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch


def file_sha256(path: Path | str) -> str:
    """Computes hex SHA-256 of a file on disk prefixed with 'sha256:'."""
    p = Path(path)
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def bytes_sha256(data: bytes) -> str:
    """Computes hex SHA-256 of in-memory bytes prefixed with 'sha256:'."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def model_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    """
    Computes a deterministic, platform-stable SHA-256 hash of a PyTorch state dict.
    Sorts tensor keys alphabetically and hashes key name, shape, dtype, and contiguous bytes.
    Prefix: 'sha256:<64-char-hex>'
    """
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        # Ensure float32 representation for determinism
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        # Hash name, shape, dtype
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))

        # Hash raw tensor bytes using numpy buffer view
        raw_bytes = tensor.numpy().tobytes()
        hasher.update(raw_bytes)

    return f"sha256:{hasher.hexdigest()}"