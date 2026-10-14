"""
Privacy indicators generator and threat model disclosure for the Hive Host.
Computes live metrics for the 7 privacy indicators and outputs the documented limitations.
"""

from __future__ import annotations

import json
from typing import Any

from hive.core.dp import estimate_epsilon
from hive.host.db import HostDatabase
from hive.host.state import HostState

LIMITATIONS_TEXT = (
    "Differential Privacy epsilon is an estimated upper-bound based on client-reported parameters "
    "without subsampling amplification. Secure aggregation (SecAgg+) is not currently enabled in v1; "
    "the Host server observes individual weight updates. Robust aggregators (Trimmed Mean, Median, Krum) "
    "mitigate Byzantine poisoning up to theoretical thresholds but do not guarantee 100% elimination. "
    "Weight-encoded exfiltration by a malicious package is mitigated by author signing, consent reviews, "
    "and differential privacy, but cannot be mathematically ruled out without formal verification."
)


def get_privacy_indicators(db: HostDatabase, state: HostState, min_clients: int, model_size_bytes: int) -> dict[str, Any]:
    """
    Computes data for the 7 privacy indicator cards shown on the Host Web UI.
    """
    byte_summary = db.get_byte_accounting_summary()
    all_updates = db.list_all_updates()
    all_rounds = db.list_rounds()

    # 1. Transport indicator
    transport_status = "green" if state.is_tls else "red"
    transport_label = "TLS Encrypted (HTTPS)" if state.is_tls else "Unencrypted Transport (HTTP)"

    # 2. Raw-data guard indicator
    bytes_by_cat: dict[str, int] = {}
    for b in byte_summary:
        cat = b["category"]
        bytes_by_cat[cat] = bytes_by_cat.get(cat, 0) + b["bytes_in"]

    rejected_uploads = sum(1 for u in all_updates if u["accepted"] == 0)
    raw_data_status = "green" if rejected_uploads == 0 else "amber"

    # 3. Update-only ratio indicator
    total_update_bytes = bytes_by_cat.get("update_upload", 0)
    closed_rounds_count = sum(1 for r in all_rounds if r["status"] == "closed")
    accepted_updates_count = sum(1 for u in all_updates if u["accepted"] == 1)
    expected_bytes = max(1, accepted_updates_count * model_size_bytes)
    ratio = total_update_bytes / expected_bytes if expected_bytes > 0 else 1.0

    # 4. Differential Privacy indicator
    dp_updates_count = 0
    clip_norms: list[float] = []
    noise_mults: list[float] = []
    for u in all_updates:
        if u["accepted"] == 1:
            try:
                m = json.loads(u["meta_json"])
                dp_dict = m.get("dp", {})
                if dp_dict.get("enabled", False):
                    dp_updates_count += 1
                    if "clip_norm" in dp_dict:
                        clip_norms.append(float(dp_dict["clip_norm"]))
                    if "noise_multiplier" in dp_dict:
                        noise_mults.append(float(dp_dict["noise_multiplier"]))
            except Exception:
                pass

    dp_pct = (dp_updates_count / accepted_updates_count * 100.0) if accepted_updates_count > 0 else 0.0
    est_eps = estimate_epsilon(max(noise_mults), closed_rounds_count) if (noise_mults and closed_rounds_count > 0) else None

    # 5. Aggregation anonymity floor indicator
    participant_counts = [len(db.list_updates(r["round_id"])) for r in all_rounds if r["status"] == "closed"]
    min_observed = min(participant_counts) if participant_counts else 0

    # 6. Poisoning defenses indicator
    latest_round = db.get_latest_round()
    aggregator_name = latest_round["aggregator_used"] if latest_round else "fedavg"
    outliers_flagged = sum(1 for u in all_updates if u["accepted"] == 1 and u["norm"] > 10.0)

    return {
        "transport": {
            "status": transport_status,
            "label": transport_label,
            "is_tls": state.is_tls,
            "url": state.public_url,
        },
        "raw_data_guard": {
            "status": raw_data_status,
            "bytes_received_by_category": bytes_by_cat,
            "rejected_non_conforming_uploads": rejected_uploads,
            "statement": "Host endpoints accept strictly 'meta' JSON + 'delta' safetensors. No audio or raw data received.",
        },
        "update_only_ratio": {
            "status": "green" if 0.8 <= ratio <= 1.3 else "amber",
            "ratio": round(ratio, 2),
            "total_update_bytes": total_update_bytes,
            "expected_bytes": expected_bytes,
        },
        "differential_privacy": {
            "status": "green" if dp_pct > 0 else "neutral",
            "percent_dp_enabled": round(dp_pct, 1),
            "mean_clip_norm": round(sum(clip_norms) / len(clip_norms), 4) if clip_norms else None,
            "max_noise_multiplier": max(noise_mults) if noise_mults else None,
            "estimated_cumulative_epsilon": round(est_eps, 2) if est_eps is not None else None,
            "note": "Client-reported conservative estimate (no subsampling amplification).",
        },
        "anonymity_floor": {
            "status": "green" if (min_observed >= min_clients or min_observed == 0) else "amber",
            "configured_min_clients": min_clients,
            "smallest_observed_participants": min_observed,
        },
        "poisoning_defenses": {
            "aggregator_in_use": aggregator_name,
            "outliers_flagged_count": outliers_flagged,
            "fallback_used": latest_round.get("fallback_note") if latest_round else None,
        },
        "limitations": LIMITATIONS_TEXT,
    }