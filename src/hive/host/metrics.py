"""
Metrics aggregation and contributor accounting for the Hive Host.
Computes round progress, contribution shares, update norms, and contributor metrics
without tracking or storing IP addresses.
"""

from __future__ import annotations

import json
from typing import Any

from hive.host.db import HostDatabase


def get_contribution_metrics(db: HostDatabase) -> list[dict[str, Any]]:
    """
    Computes per-node contribution metrics:
    rounds contributed, missed claims, sample counts, update norms,
    cosine similarities, outlier flags, and aggregate weights.
    Pseudonymous only (nicknames and node_id; zero IP addresses).
    """
    nodes = db.list_nodes()
    all_updates = db.list_all_updates()
    all_rounds = db.list_rounds()
    byte_summary = db.get_byte_accounting_summary()

    # Map byte accounting
    bytes_by_node: dict[str, dict[str, int]] = {}
    for b in byte_summary:
        nid = b["node_id"]
        if nid not in bytes_by_node:
            bytes_by_node[nid] = {"bytes_in": 0, "bytes_out": 0}
        bytes_by_node[nid]["bytes_in"] += b["bytes_in"]
        bytes_by_node[nid]["bytes_out"] += b["bytes_out"]

    total_accepted_all_nodes = sum(1 for u in all_updates if u["accepted"] == 1)

    result: list[dict[str, Any]] = []
    for node in nodes:
        nid = node["node_id"]
        node_updates = [u for u in all_updates if u["node_id"] == nid]
        accepted_updates = [u for u in node_updates if u["accepted"] == 1]
        rejected_updates = [u for u in node_updates if u["accepted"] == 0]

        # Count claims across all rounds
        claims_count = 0
        for r in all_rounds:
            if nid in db.list_claims(r["round_id"]):
                claims_count += 1

        rounds_contributed = len(accepted_updates)
        rounds_missed = max(0, claims_count - rounds_contributed)

        # Compute sample sums
        samples_reported = 0
        norms: list[float] = []
        cosines: list[float] = []
        dp_enabled_count = 0
        for u in accepted_updates:
            norms.append(u["norm"])
            cosines.append(u["cosine"])
            try:
                m = json.loads(u["meta_json"])
                samples_reported += m.get("num_train_samples", 0)
                if m.get("dp", {}).get("enabled", False):
                    dp_enabled_count += 1
            except Exception:
                pass

        mean_norm = (sum(norms) / len(norms)) if norms else 0.0
        mean_cos = (sum(cosines) / len(cosines)) if cosines else 1.0
        share_weight = (rounds_contributed / total_accepted_all_nodes * 100.0) if total_accepted_all_nodes > 0 else 0.0

        b_info = bytes_by_node.get(nid, {"bytes_in": 0, "bytes_out": 0})

        result.append({
            "node_id": nid,
            "nickname": node["nickname"],
            "state": node["state"],
            "approved": bool(node["approved"]),
            "rounds_contributed": rounds_contributed,
            "rounds_missed": rounds_missed,
            "samples_reported": samples_reported,
            "accepted_count": len(accepted_updates),
            "rejected_count": len(rejected_updates),
            "mean_update_norm": round(mean_norm, 4),
            "mean_cosine_similarity": round(mean_cos, 4),
            "share_of_aggregate_weight_pct": round(share_weight, 2),
            "dp_participated_rounds": dp_enabled_count,
            "bytes_in": b_info["bytes_in"],
            "bytes_out": b_info["bytes_out"],
            "last_seen": node["last_seen"],
        })

    return result