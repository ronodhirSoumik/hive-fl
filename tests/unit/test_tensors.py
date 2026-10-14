"""
Unit tests for safe tensor serialization, delta math, validation, clipping, and noise.
"""

import math
import pytest
import torch

from hive.core.errors import ProtocolError
from hive.core.tensors import (
    add_gaussian_noise,
    apply_delta,
    bytes_to_state_dict,
    clip_by_norm,
    compute_delta,
    cosine_similarity,
    flatten,
    l2_norm,
    per_layer_norms,
    state_dict_to_bytes,
    unflatten,
    validate_update,
)


@pytest.fixture
def sample_state_dict() -> dict[str, torch.Tensor]:
    return {
        "conv.weight": torch.tensor([[[[1.0, -1.0], [0.5, -0.5]]]], dtype=torch.float32),
        "fc.weight": torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
        "fc.bias": torch.tensor([0.05], dtype=torch.float32),
    }


def test_safetensors_serialization_roundtrip(sample_state_dict: dict[str, torch.Tensor]) -> None:
    raw_bytes = state_dict_to_bytes(sample_state_dict)
    assert len(raw_bytes) > 0

    reconstructed = bytes_to_state_dict(raw_bytes)
    assert set(reconstructed.keys()) == set(sample_state_dict.keys())
    for k in sample_state_dict.keys():
        assert torch.allclose(reconstructed[k], sample_state_dict[k])


def test_delta_computation_and_apply(sample_state_dict: dict[str, torch.Tensor]) -> None:
    global_weights = sample_state_dict
    local_weights = {
        "conv.weight": global_weights["conv.weight"] + 0.1,
        "fc.weight": global_weights["fc.weight"] - 0.2,
        "fc.bias": global_weights["fc.bias"] + 0.05,
    }

    delta = compute_delta(local_weights, global_weights)
    assert torch.allclose(delta["conv.weight"], torch.tensor([[[[0.1, 0.1], [0.1, 0.1]]]]))
    assert torch.allclose(delta["fc.weight"], torch.tensor([[-0.2, -0.2, -0.2, -0.2]]))

    # Apply delta with server_lr = 1.0
    new_global = apply_delta(global_weights, delta, server_lr=1.0)
    for k in global_weights.keys():
        assert torch.allclose(new_global[k], local_weights[k])

    # Apply delta with server_lr = 0.5
    new_global_half = apply_delta(global_weights, delta, server_lr=0.5)
    assert torch.allclose(new_global_half["fc.bias"], global_weights["fc.bias"] + 0.025)


def test_validate_update_happy_path(sample_state_dict: dict[str, torch.Tensor]) -> None:
    delta = {k: torch.zeros_like(v) for k, v in sample_state_dict.items()}
    # Should not raise
    validate_update(delta, sample_state_dict)


def test_validate_update_key_mismatch(sample_state_dict: dict[str, torch.Tensor]) -> None:
    # Missing key
    incomplete_delta = {k: v for k, v in sample_state_dict.items() if k != "fc.bias"}
    with pytest.raises(ProtocolError, match="missing keys"):
        validate_update(incomplete_delta, sample_state_dict)

    # Extra key
    extra_delta = dict(sample_state_dict)
    extra_delta["unexpected.layer"] = torch.tensor([1.0])
    with pytest.raises(ProtocolError, match="extra unexpected keys"):
        validate_update(extra_delta, sample_state_dict)


def test_validate_update_shape_mismatch(sample_state_dict: dict[str, torch.Tensor]) -> None:
    bad_shape_delta = dict(sample_state_dict)
    bad_shape_delta["fc.bias"] = torch.tensor([0.1, 0.2])  # Expected shape (1,)
    with pytest.raises(ProtocolError, match="Shape mismatch"):
        validate_update(bad_shape_delta, sample_state_dict)


def test_validate_update_non_finite_values(sample_state_dict: dict[str, torch.Tensor]) -> None:
    # NaN
    nan_delta = dict(sample_state_dict)
    nan_delta["fc.bias"] = torch.tensor([float("nan")])
    with pytest.raises(ProtocolError, match="Non-finite values"):
        validate_update(nan_delta, sample_state_dict)

    # Inf
    inf_delta = dict(sample_state_dict)
    inf_delta["fc.bias"] = torch.tensor([float("inf")])
    with pytest.raises(ProtocolError, match="Non-finite values"):
        validate_update(inf_delta, sample_state_dict)


def test_flatten_and_unflatten_roundtrip(sample_state_dict: dict[str, torch.Tensor]) -> None:
    vec = flatten(sample_state_dict)
    expected_total_elements = sum(v.numel() for v in sample_state_dict.values())
    assert vec.numel() == expected_total_elements
    assert vec.dim() == 1

    reconstructed = unflatten(vec, sample_state_dict)
    assert set(reconstructed.keys()) == set(sample_state_dict.keys())
    for k in sample_state_dict.keys():
        assert torch.allclose(reconstructed[k], sample_state_dict[k])


def test_l2_norm_and_per_layer_norms() -> None:
    sd = {
        "l1": torch.tensor([3.0, 4.0]),  # norm = 5.0
        "l2": torch.tensor([12.0]),       # norm = 12.0
    }
    # Total norm = sqrt(5^2 + 12^2) = 13.0
    assert math.isclose(l2_norm(sd), 13.0, rel_tol=1e-5)

    layer_norms = per_layer_norms(sd)
    assert math.isclose(layer_norms["l1"], 5.0, rel_tol=1e-5)
    assert math.isclose(layer_norms["l2"], 12.0, rel_tol=1e-5)


def test_cosine_similarity() -> None:
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([1.0, 0.0, 0.0])
    assert math.isclose(cosine_similarity(v1, v2), 1.0, rel_tol=1e-5)

    v_opp = torch.tensor([-1.0, 0.0, 0.0])
    assert math.isclose(cosine_similarity(v1, v_opp), -1.0, rel_tol=1e-5)

    v_orth = torch.tensor([0.0, 1.0, 0.0])
    assert math.isclose(cosine_similarity(v1, v_orth), 0.0, abs_tol=1e-5)


def test_clip_by_norm() -> None:
    vec = torch.tensor([3.0, 4.0])  # norm = 5.0

    # Clip with max_norm = 10.0 (no clipping should occur)
    unclipped, orig_norm = clip_by_norm(vec, max_norm=10.0)
    assert math.isclose(orig_norm, 5.0, rel_tol=1e-5)
    assert math.isclose(l2_norm(unclipped), 5.0, rel_tol=1e-5)

    # Clip with max_norm = 2.5 (should scale by 0.5)
    clipped, orig_norm = clip_by_norm(vec, max_norm=2.5)
    assert math.isclose(orig_norm, 5.0, rel_tol=1e-5)
    assert math.isclose(l2_norm(clipped), 2.5, rel_tol=1e-5)
    assert torch.allclose(clipped, torch.tensor([1.5, 2.0]))


def test_add_gaussian_noise() -> None:
    torch.manual_seed(42)
    vec = torch.zeros(10000, dtype=torch.float32)
    clip_norm = 2.0
    noise_mult = 0.5
    expected_std = clip_norm * noise_mult  # 1.0

    noisy = add_gaussian_noise(vec, clip_norm=clip_norm, noise_multiplier=noise_mult)
    sample_std = float(torch.std(noisy).item())
    sample_mean = float(torch.mean(noisy).item())

    assert math.isclose(sample_mean, 0.0, abs_tol=0.05)
    assert math.isclose(sample_std, expected_std, rel_tol=0.05)

    # Multiplier <= 0 returns exact copy
    no_noise = add_gaussian_noise(vec, clip_norm=2.0, noise_multiplier=0.0)
    assert torch.equal(no_noise, vec)