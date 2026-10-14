"""
Differential Privacy accounting using Renyi Differential Privacy (RDP).
Provides conservative, unamplified privacy budget tracking for the Gaussian mechanism.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

# Canonical alpha orders evaluated for RDP to (epsilon, delta) conversion
RDP_ORDERS = [
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 48.0, 64.0, 128.0, 256.0,
]


def compute_rdp_gaussian(alpha: float, noise_multiplier: float) -> float:
    """
    Computes Renyi Differential Privacy (RDP) of order alpha for a single Gaussian mechanism query:
    RDP(alpha) = alpha / (2 * noise_multiplier^2).
    If noise_multiplier <= 0, returns infinity.
    """
    if noise_multiplier <= 0.0:
        return float("inf")
    return alpha / (2.0 * (noise_multiplier**2))


def rdp_to_dp(rdp_values: Sequence[float], delta: float, orders: Sequence[float] = RDP_ORDERS) -> float:
    """
    Converts a sequence of cumulative RDP values across orders to an (epsilon, delta) guarantee:
    epsilon = min_alpha [ RDP(alpha) + ln(1/delta) / (alpha - 1) ].
    """
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"Target delta must be in (0, 1), got {delta}")

    min_eps = float("inf")
    log_inv_delta = math.log(1.0 / delta)

    for rdp, alpha in zip(rdp_values, orders):
        if math.isinf(rdp):
            return float("inf")
        eps = rdp + (log_inv_delta / (alpha - 1.0))
        if eps < min_eps:
            min_eps = eps

    return min_eps


def estimate_epsilon(
    noise_multiplier: float,
    rounds: int,
    delta: float = 1e-5,
    orders: Sequence[float] = RDP_ORDERS,
) -> float:
    """
    Estimates the cumulative privacy loss epsilon over T independent rounds
    with a constant noise multiplier and target delta.
    Conservative: assumes no subsampling amplification.
    """
    if noise_multiplier <= 0.0 or rounds <= 0:
        return float("inf")

    cumulative_rdps = [rounds * compute_rdp_gaussian(alpha, noise_multiplier) for alpha in orders]
    return rdp_to_dp(cumulative_rdps, delta=delta, orders=orders)


class DPAccountant:
    """
    Tracks privacy budget consumption round-by-round for a contributor node.
    """

    def __init__(self, orders: Sequence[float] = RDP_ORDERS) -> None:
        self.orders = list(orders)
        self.history: list[dict[str, float]] = []
        self._cumulative_rdp = [0.0 for _ in self.orders]
        self._has_infinite = False

    def add_round(self, noise_multiplier: float, clip_norm: float = 1.0) -> None:
        """Records participation in a round with the specified DP parameters."""
        self.history.append({
            "noise_multiplier": noise_multiplier,
            "clip_norm": clip_norm,
        })
        if noise_multiplier <= 0.0:
            self._has_infinite = True
        else:
            for i, alpha in enumerate(self.orders):
                self._cumulative_rdp[i] += compute_rdp_gaussian(alpha, noise_multiplier)

    @property
    def total_rounds(self) -> int:
        return len(self.history)

    def get_cumulative_epsilon(self, delta: float = 1e-5) -> float:
        """Returns the current total privacy parameter epsilon for the target delta."""
        if not self.history or self._has_infinite:
            return float("inf")
        return rdp_to_dp(self._cumulative_rdp, delta=delta, orders=self.orders)

    def summary(self, delta: float = 1e-5) -> dict[str, Any]:
        """Produces a JSON-serializable privacy summary."""
        eps = self.get_cumulative_epsilon(delta=delta)
        clip_norms = [h["clip_norm"] for h in self.history]
        noise_mults = [h["noise_multiplier"] for h in self.history]
        return {
            "rounds_tracked": self.total_rounds,
            "target_delta": delta,
            "cumulative_epsilon": None if math.isinf(eps) else round(eps, 4),
            "is_infinite": math.isinf(eps),
            "mean_clip_norm": sum(clip_norms) / len(clip_norms) if clip_norms else 0.0,
            "max_noise_multiplier": max(noise_mults) if noise_mults else 0.0,
            "is_conservative_estimate": True,
        }