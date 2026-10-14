"""
Unit tests for Differential Privacy accounting (RDP Gaussian mechanism).
"""

import math
import pytest
from hive.core.dp import DPAccountant, compute_rdp_gaussian, estimate_epsilon, rdp_to_dp


def test_compute_rdp_gaussian() -> None:
    # RDP(alpha=2, sigma=1.0) = 2 / (2 * 1^2) = 1.0
    assert math.isclose(compute_rdp_gaussian(2.0, 1.0), 1.0)

    # RDP(alpha=4, sigma=2.0) = 4 / (2 * 4) = 0.5
    assert math.isclose(compute_rdp_gaussian(4.0, 2.0), 0.5)

    # Multiplier <= 0 gives inf
    assert math.isinf(compute_rdp_gaussian(2.0, 0.0))
    assert math.isinf(compute_rdp_gaussian(2.0, -1.0))


def test_estimate_epsilon() -> None:
    # Zero noise gives infinite epsilon
    assert math.isinf(estimate_epsilon(noise_multiplier=0.0, rounds=10))

    # Epsilon scales with rounds
    eps_5 = estimate_epsilon(noise_multiplier=1.0, rounds=5, delta=1e-5)
    eps_10 = estimate_epsilon(noise_multiplier=1.0, rounds=10, delta=1e-5)
    assert eps_10 > eps_5 > 0.0

    # Epsilon decreases with higher noise
    eps_high_noise = estimate_epsilon(noise_multiplier=2.0, rounds=10, delta=1e-5)
    assert eps_high_noise < eps_10


def test_dp_accountant_workflow() -> None:
    accountant = DPAccountant()
    assert accountant.total_rounds == 0
    assert math.isinf(accountant.get_cumulative_epsilon())

    # Add 3 rounds with noise multiplier 1.0
    for _ in range(3):
        accountant.add_round(noise_multiplier=1.0, clip_norm=1.0)

    assert accountant.total_rounds == 3
    eps = accountant.get_cumulative_epsilon(delta=1e-5)
    assert not math.isinf(eps)
    assert eps > 0.0

    summary = accountant.summary(delta=1e-5)
    assert summary["rounds_tracked"] == 3
    assert summary["is_infinite"] is False
    assert summary["target_delta"] == 1e-5
    assert summary["cumulative_epsilon"] == round(eps, 4)
    assert summary["is_conservative_estimate"] is True


def test_dp_accountant_unnoised_round() -> None:
    accountant = DPAccountant()
    accountant.add_round(noise_multiplier=1.0, clip_norm=1.0)
    assert not math.isinf(accountant.get_cumulative_epsilon())

    # Add round with 0 noise -> breaks DP guarantee
    accountant.add_round(noise_multiplier=0.0, clip_norm=1.0)
    assert math.isinf(accountant.get_cumulative_epsilon())
    assert accountant.summary()["is_infinite"] is True