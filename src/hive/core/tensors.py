"""
Safe tensor serialization, update validation, vector flattening, norm clipping,
and differential privacy noise addition for Hive.
Strictly relies on safetensors to guarantee pickle-free deserialization.
"""

from __future__ import annotations

import io
from typing import Mapping, Sequence

import safetensors.torch
import torch

from hive.core.errors import ProtocolError

DEFAULT_MAX_UPDATE_BYTES = 50 * 1024 * 1024  # 50 MB


def state_dict_to_bytes(state_dict: Mapping[str, torch.Tensor]) -> bytes:
    """
    Serializes a PyTorch state dict to bytes using safetensors only.
    Guarantees float32 tensors and contiguous memory layouts.
    """
    clean_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for key '{k}', got {type(v)}")
        t = v.detach().cpu()
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
        if not t.is_contiguous():
            t = t.contiguous()
        clean_dict[k] = t
    return safetensors.torch.save(clean_dict)


def bytes_to_state_dict(data: bytes) -> dict[str, torch.Tensor]:
    """
    Deserializes safetensors binary bytes back into a PyTorch state dict.
    Safe against arbitrary code execution (no pickle).
    """
    try:
        tensors = safetensors.torch.load(data)
    except Exception as e:
        raise ProtocolError(f"Failed to deserialize safetensors payload: {e}", code="ERR_TENSOR_DESERIALIZATION")
    return {k: v.to(torch.float32) for k, v in tensors.items()}


def compute_delta(
    local_weights: Mapping[str, torch.Tensor],
    global_weights: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Computes model weight delta: delta = local - global."""
    delta: dict[str, torch.Tensor] = {}
    for k in global_weights.keys():
        if k not in local_weights:
            raise ProtocolError(f"Local state dict missing required key: '{k}'", code="ERR_KEY_MISMATCH")
        delta[k] = local_weights[k].detach().cpu().to(torch.float32) - global_weights[k].detach().cpu().to(torch.float32)
    return delta


def apply_delta(
    global_weights: Mapping[str, torch.Tensor],
    delta: Mapping[str, torch.Tensor],
    server_lr: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Applies aggregated update delta to global weights: new_global = global + server_lr * delta."""
    new_global: dict[str, torch.Tensor] = {}
    for k in global_weights.keys():
        if k not in delta:
            raise ProtocolError(f"Delta missing parameter key: '{k}'", code="ERR_KEY_MISMATCH")
        g = global_weights[k].detach().cpu().to(torch.float32)
        d = delta[k].detach().cpu().to(torch.float32)
        new_global[k] = g + (server_lr * d)
    return new_global


def validate_update(
    delta: Mapping[str, torch.Tensor],
    template: Mapping[str, torch.Tensor],
    max_bytes: int = DEFAULT_MAX_UPDATE_BYTES,
) -> None:
    """
    Strictly validates a client update delta against the expected global model template:
    1. Exact key match (no missing or extra keys).
    2. Exact shape match for every tensor.
    3. Float32 data type.
    4. Finiteness (no NaN, Inf, or -Inf values).
    5. Serialized size within max_bytes.
    Raises ProtocolError on any violation.
    """
    # 1. Key validation
    delta_keys = set(delta.keys())
    template_keys = set(template.keys())
    if delta_keys != template_keys:
        missing = sorted(template_keys - delta_keys)
        extra = sorted(delta_keys - template_keys)
        msg_parts = []
        if missing:
            msg_parts.append(f"missing keys: {missing}")
        if extra:
            msg_parts.append(f"extra unexpected keys: {extra}")
        raise ProtocolError(f"Update keys mismatch template ({', '.join(msg_parts)})", code="ERR_TENSOR_KEY_MISMATCH")

    # 2. Shape, dtype, and finiteness validation
    total_elements = 0
    for k, expected_tensor in template.items():
        t = delta[k]
        if not isinstance(t, torch.Tensor):
            raise ProtocolError(f"Value for '{k}' is not a torch.Tensor", code="ERR_TENSOR_TYPE_INVALID")

        if t.shape != expected_tensor.shape:
            raise ProtocolError(
                f"Shape mismatch for parameter '{k}': expected {expected_tensor.shape}, got {t.shape}",
                code="ERR_TENSOR_SHAPE_MISMATCH",
            )

        if t.dtype != torch.float32:
            raise ProtocolError(
                f"Invalid dtype for parameter '{k}': expected torch.float32, got {t.dtype}",
                code="ERR_TENSOR_DTYPE_INVALID",
            )

        if not torch.isfinite(t).all():
            raise ProtocolError(
                f"Non-finite values (NaN or Inf) detected in parameter '{k}'",
                code="ERR_TENSOR_NON_FINITE",
            )
        total_elements += t.numel()

    # 3. Payload size bound check (4 bytes per float32 plus header allowance)
    approx_bytes = total_elements * 4
    if approx_bytes > max_bytes:
        raise ProtocolError(
            f"Update payload exceeds maximum allowed size ({approx_bytes} > {max_bytes} bytes)",
            code="ERR_TENSOR_OVERSIZED",
        )


def flatten(state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Concatenates all tensors sorted by key into a single 1D float32 vector."""
    vectors: list[torch.Tensor] = []
    for k in sorted(state_dict.keys()):
        vectors.append(state_dict[k].detach().cpu().reshape(-1).to(torch.float32))
    if not vectors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(vectors, dim=0)


def unflatten(
    vec: torch.Tensor,
    template: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reconstructs a state dict from a 1D vector using parameter shapes from template."""
    state_dict: dict[str, torch.Tensor] = {}
    offset = 0
    flat = vec.detach().cpu().to(torch.float32)

    for k in sorted(template.keys()):
        ref = template[k]
        numel = ref.numel()
        slice_vec = flat[offset : offset + numel]
        if slice_vec.numel() != numel:
            raise ValueError(f"Vector underflow while unflattening parameter '{k}'")
        state_dict[k] = slice_vec.reshape(ref.shape).clone()
        offset += numel

    if offset != flat.numel():
        raise ValueError(f"Vector length mismatch: used {offset} elements, but vector has {flat.numel()}")
    return state_dict


def l2_norm(state_dict_or_vec: Mapping[str, torch.Tensor] | torch.Tensor) -> float:
    """Computes the overall Euclidean (L2) norm of a vector or state dict."""
    if isinstance(state_dict_or_vec, torch.Tensor):
        return float(torch.norm(state_dict_or_vec.detach().cpu().to(torch.float32), p=2).item())
    total_sq = 0.0
    for t in state_dict_or_vec.values():
        val = torch.norm(t.detach().cpu().to(torch.float32), p=2).item()
        total_sq += val * val
    return float(total_sq**0.5)


def per_layer_norms(state_dict: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Computes L2 norm for each layer parameter in state dict."""
    return {
        k: float(torch.norm(v.detach().cpu().to(torch.float32), p=2).item())
        for k, v in state_dict.items()
    }


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """Computes cosine similarity between two 1D vectors."""
    v1 = vec1.detach().cpu().reshape(-1).to(torch.float32)
    v2 = vec2.detach().cpu().reshape(-1).to(torch.float32)
    n1 = torch.norm(v1, p=2)
    n2 = torch.norm(v2, p=2)
    if n1.item() == 0.0 or n2.item() == 0.0:
        return 0.0
    dot = torch.dot(v1, v2)
    sim = dot / (n1 * n2)
    return float(sim.item())


def clip_by_norm(
    delta: Mapping[str, torch.Tensor] | torch.Tensor,
    max_norm: float,
) -> tuple[dict[str, torch.Tensor] | torch.Tensor, float]:
    """
    Clips a delta (state dict or 1D tensor) to a maximum L2 norm:
    If norm > max_norm, scales down by (max_norm / norm).
    Returns (clipped_delta, original_norm).
    """
    orig_norm = l2_norm(delta)
    if orig_norm <= max_norm or orig_norm == 0.0:
        if isinstance(delta, torch.Tensor):
            return delta.clone(), orig_norm
        return {k: v.clone() for k, v in delta.items()}, orig_norm

    scale = max_norm / orig_norm
    if isinstance(delta, torch.Tensor):
        return delta * scale, orig_norm
    clipped_dict = {k: v * scale for k, v in delta.items()}
    return clipped_dict, orig_norm


def add_gaussian_noise(
    delta: Mapping[str, torch.Tensor] | torch.Tensor,
    clip_norm: float,
    noise_multiplier: float,
    rng: torch.Generator | None = None,
) -> dict[str, torch.Tensor] | torch.Tensor:
    """
    Adds Gaussian differential-privacy noise with std = noise_multiplier * clip_norm.
    If noise_multiplier <= 0, returns unperturbed copy.
    """
    if noise_multiplier <= 0.0:
        if isinstance(delta, torch.Tensor):
            return delta.clone()
        return {k: v.clone() for k, v in delta.items()}

    std = float(noise_multiplier * clip_norm)
    if isinstance(delta, torch.Tensor):
        noise = torch.randn(delta.shape, generator=rng, dtype=torch.float32) * std
        return delta + noise

    noisy_dict: dict[str, torch.Tensor] = {}
    for k, v in delta.items():
        noise = torch.randn(v.shape, generator=rng, dtype=torch.float32) * std
        noisy_dict[k] = v + noise
    return noisy_dict