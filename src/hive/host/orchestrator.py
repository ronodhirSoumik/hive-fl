"""
Federated Learning Round Orchestrator for Hive Host.
Manages session lifecycle, round transitions, client claims, update reception,
Byzantine-robust aggregation, persistence, and checkpointing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from hive.core.config import HostConfig
from hive.core.errors import ProtocolError
from hive.core.hashing import model_hash
from hive.core.logging import get_logger
from hive.core.schemas import (
    AggregationConfig,
    Capabilities,
    NodeState,
    SessionState,
    UpdateMeta,
)
from hive.core.tensors import (
    apply_delta,
    bytes_to_state_dict,
    flatten,
    l2_norm,
    state_dict_to_bytes,
    validate_update,
)
from hive.host.aggregation.base import AggregationResult, Aggregator, ClientUpdate
from hive.host.aggregation.factory import make_aggregator
from hive.host.db import HostDatabase
from hive.host.state import HostState

logger = get_logger("hive.host.orchestrator")


class Orchestrator:
    """
    Asynchronous state machine governing federated learning rounds.
    """

    def __init__(
        self,
        config: HostConfig,
        db: HostDatabase,
        package_digest: str,
        host_key_fingerprint: str,
        initial_weights: dict[str, torch.Tensor],
        aggregator: Aggregator | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.workdir = Path(config.workdir)
        self.package_digest = package_digest
        self.host_key_fingerprint = host_key_fingerprint

        self.aggregator = aggregator or make_aggregator(
            config.aggregation,
            AggregationConfig(
                name=config.aggregation,
                server_clip_norm=config.server_clip_norm,
                trim_ratio=config.trim_ratio,
            ),
        )

        self.state = HostState()
        self.state.global_weights = {k: v.clone() for k, v in initial_weights.items()}
        self.state.model_hash = model_hash(self.state.global_weights)

        self.checkpoints_dir = self.workdir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._loop_task: asyncio.Task[None] | None = None
        self._round_retries = 0
        self._pending_updates: dict[str, ClientUpdate] = {}

    async def start(self, resume: bool = False) -> None:
        """Initializes storage, restores checkpoints if resuming, and starts background loop."""
        async with self.state.lock:
            # Check or save session in DB
            session = self.db.get_session()
            if not session:
                join_tok = secrets.token_urlsafe(16)
                session = self.db.create_or_load_session(
                    session_id=f"sess_{secrets.token_hex(6)}",
                    name="Hive FL Session",
                    description="Federated learning round orchestration",
                    data_requirements="Configured package dataset requirements",
                    package_digest=self.package_digest,
                    host_key_fingerprint=self.host_key_fingerprint,
                    total_rounds=self.config.total_rounds,
                    join_token=join_tok,
                )
            self.state.join_token = session["join_token"]

            if resume:
                latest_rnd = self.db.get_latest_round()
                if latest_rnd and latest_rnd["status"] == "closed" and latest_rnd["model_hash_out"]:
                    ckpt_file = self.checkpoints_dir / f"round_{latest_rnd['round_id']}.safetensors"
                    if ckpt_file.exists():
                        self.state.global_weights = bytes_to_state_dict(ckpt_file.read_bytes())
                        self.state.model_hash = latest_rnd["model_hash_out"]
                        self.state.current_round_id = latest_rnd["round_id"]
                        logger.info(f"Resumed from round {latest_rnd['round_id']} checkpoint: {ckpt_file}")
            else:
                # Save initial round 0 weights
                round_0_file = self.checkpoints_dir / "round_0.safetensors"
                if not round_0_file.exists():
                    round_0_file.write_bytes(state_dict_to_bytes(self.state.global_weights))

            self.state.status = SessionState.RUNNING
            self.db.update_session(status=SessionState.RUNNING.value)
            self._running = True
            self.db.add_event("INFO", f"Host orchestrator started for session {session['session_id']}")

        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Gracefully halts the orchestrator loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self.state.status = SessionState.PAUSED
        self.db.update_session(status=SessionState.PAUSED.value)
        self.db.add_event("INFO", "Host orchestrator stopped")

    # ---------------- State Machine Tick Loop ----------------

    async def _run_loop(self) -> None:
        """Main periodic supervision loop."""
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"Error in orchestrator tick: {e}", exc_info=True)
            await asyncio.sleep(1.0)

    async def _tick(self) -> None:
        now = time.time()
        async with self.state.lock:
            if self.state.status != SessionState.RUNNING:
                return

            # 1. Heartbeat check: detect offline nodes
            for node in self.db.list_nodes():
                if node["state"] not in (NodeState.OFFLINE.value, NodeState.LEFT.value):
                    if now - node["last_seen"] > self.config.offline_after_s:
                        self.db.update_node(node["node_id"], state=NodeState.OFFLINE.value)
                        self.db.add_event("WARNING", f"Node '{node['nickname']}' marked offline (no heartbeat)")

            current_round = self.db.get_latest_round()

            # 2. Check if we should open a new round
            if current_round is None or current_round["status"] == "closed":
                next_round_id = (current_round["round_id"] + 1) if current_round else 1
                if next_round_id > self.config.total_rounds:
                    # Session finished
                    self.state.status = SessionState.COMPLETED
                    self.db.update_session(status=SessionState.COMPLETED.value, closed_at=now)
                    final_path = self.workdir / "final_model.safetensors"
                    final_path.write_bytes(state_dict_to_bytes(self.state.global_weights))
                    self.db.add_event("INFO", f"All {self.config.total_rounds} rounds completed! Final model saved.")
                    return

                # Check eligible participants count
                eligible_nodes = [
                    n for n in self.db.list_nodes()
                    if n["approved"] == 1 and n["state"] in (NodeState.ELIGIBLE.value, "waiting", NodeState.TRAINING.value)
                ]
                if len(eligible_nodes) >= self.config.min_clients:
                    # Open new round
                    deadline = now + self.config.round_timeout_s
                    self.db.create_round(next_round_id, deadline, self.state.model_hash)
                    self.state.current_round_id = next_round_id
                    self.state.active_claims.clear()
                    self._pending_updates.clear()
                    self._round_retries = 0
                    self.db.update_session(current_round=next_round_id)
                    self.db.add_event("INFO", f"Round {next_round_id}/{self.config.total_rounds} opened (deadline in {self.config.round_timeout_s}s)")
                return

            # 3. Handle active open round
            if current_round["status"] == "open":
                r_id = current_round["round_id"]
                deadline = current_round["deadline"]
                claims = self.db.list_claims(r_id)
                updates = self.db.list_updates(r_id)

                time_expired = now >= deadline
                all_claims_submitted = len(claims) >= self.config.min_clients and len(updates) >= len(claims)
                max_clients_reached = len(updates) >= self.config.max_clients_per_round

                if time_expired or all_claims_submitted or max_clients_reached:
                    # Close and evaluate round
                    accepted_updates = [u for u in self._pending_updates.values() if u.meta.round_id == r_id]
                    if len(accepted_updates) >= self.config.min_clients:
                        # Success: Aggregate updates!
                        agg_res = self.aggregator.aggregate(accepted_updates, self.state.global_weights)
                        self.state.global_weights = apply_delta(self.state.global_weights, agg_res.delta, server_lr=1.0)
                        new_hash = model_hash(self.state.global_weights)
                        self.state.model_hash = new_hash

                        # Save checkpoint
                        ckpt_path = self.checkpoints_dir / f"round_{r_id}.safetensors"
                        ckpt_path.write_bytes(state_dict_to_bytes(self.state.global_weights))

                        # Calculate round weighted metrics
                        tot_eval_samples = sum(u.meta.num_eval_samples for u in accepted_updates)
                        tot_train_samples = sum(u.meta.num_train_samples for u in accepted_updates)
                        w_train_loss = (
                            sum(u.meta.train_loss * u.meta.num_train_samples for u in accepted_updates) / max(1, tot_train_samples)
                        )
                        w_eval_loss = (
                            sum(u.meta.eval_loss * u.meta.num_eval_samples for u in accepted_updates) / max(1, tot_eval_samples)
                        )
                        w_eval_acc = (
                            sum(u.meta.eval_metrics.get("accuracy", 0.0) * u.meta.num_eval_samples for u in accepted_updates)
                            / max(1, tot_eval_samples)
                        )
                        duration = now - current_round["opened_at"]

                        self.db.save_metrics(r_id, w_train_loss, w_eval_loss, w_eval_acc, len(accepted_updates), duration)
                        self.db.close_round(
                            r_id,
                            status="closed",
                            model_hash_out=new_hash,
                            aggregator_used=agg_res.aggregator_used,
                            fallback_note=agg_res.fallback_note,
                        )
                        self.db.add_event(
                            "INFO",
                            f"Round {r_id} completed via {agg_res.aggregator_used} ({len(accepted_updates)} updates, loss: {w_eval_loss:.4f}, acc: {w_eval_acc:.4f})",
                        )
                    else:
                        # Failed round due to lack of quorum
                        self._round_retries += 1
                        if self._round_retries >= 3:
                            self.state.status = SessionState.PAUSED
                            self.db.update_session(status=SessionState.PAUSED.value)
                            self.db.close_round(r_id, status="closed", fallback_note="Aborted: insufficient updates")
                            self.db.add_event("ERROR", f"Session paused: Round {r_id} failed quorum 3 times")
                        else:
                            # Extend deadline
                            new_deadline = now + self.config.round_timeout_s
                            with self.db._get_connection() as conn:
                                conn.execute("UPDATE rounds SET deadline = ? WHERE round_id = ?", (new_deadline, r_id))
                            self.db.add_event(
                                "WARNING",
                                f"Round {r_id} quorum not met ({len(accepted_updates)}/{self.config.min_clients}); retrying (attempt {self._round_retries}/3)",
                            )

    # ---------------- Contributor Node API Actions ----------------

    def register_node(self, nickname: str, capabilities: Capabilities) -> tuple[str, str, str]:
        """Registers a new node, returns (node_id, secret, approval_status)."""
        node_id = f"node_{secrets.token_hex(4)}"
        secret = secrets.token_hex(16)
        secret_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        approved = not self.config.require_approval
        self.db.add_node(node_id, nickname, secret_hash, capabilities, approved=approved)
        status = "approved" if approved else "pending"
        self.db.add_event("INFO", f"Node '{nickname}' ({node_id}) registered with status '{status}'")
        return node_id, secret, status

    def authenticate_node(self, node_id: str, secret: str) -> bool:
        """Verifies bearer token credentials with constant-time hash comparison."""
        node = self.db.get_node(node_id)
        if not node:
            return False
        expected_hash = node["secret_hash"]
        actual_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        return secrets.compare_digest(expected_hash, actual_hash)

    def mark_node_ready(self, node_id: str, num_samples: int) -> None:
        node = self.db.get_node(node_id)
        if not node:
            raise ProtocolError(f"Node {node_id} not found", code="ERR_NODE_NOT_FOUND")
        if node["approved"] != 1:
            raise ProtocolError("Node is pending administrator approval", code="ERR_NODE_PENDING_APPROVAL")
        self.db.update_node(node_id, state=NodeState.ELIGIBLE.value, num_samples=num_samples, last_seen=time.time())

    def claim_model(self, node_id: str, round_id: int) -> bytes:
        """Allows an eligible node to claim and download current global weights for training."""
        round_info = self.db.get_round(round_id)
        if not round_info or round_info["status"] != "open":
            raise ProtocolError(f"Round {round_id} is not open for claims", code="ERR_ROUND_CLOSED")

        claims = self.db.list_claims(round_id)
        if node_id not in claims and len(claims) >= self.config.max_clients_per_round:
            raise ProtocolError(f"Round {round_id} has reached maximum client capacity", code="ERR_ROUND_FULL")

        self.db.add_claim(round_id, node_id)
        self.db.update_node(node_id, state=NodeState.TRAINING.value, last_seen=time.time())
        self.db.record_bytes(node_id, "model_download", bytes_in=0, bytes_out=len(state_dict_to_bytes(self.state.global_weights)))
        return state_dict_to_bytes(self.state.global_weights)

    def submit_update(
        self,
        node_id: str,
        round_id: int,
        meta: UpdateMeta,
        delta_bytes: bytes,
    ) -> None:
        """Receives and strictly validates an uploaded model update delta."""
        now = time.time()
        round_info = self.db.get_round(round_id)
        if not round_info or round_info["status"] != "open":
            raise ProtocolError(f"Round {round_id} is closed", code="ERR_ROUND_CLOSED")

        if now > round_info["deadline"]:
            raise ProtocolError(f"Submission deadline for round {round_id} has passed", code="ERR_ROUND_LATE")

        claims = self.db.list_claims(round_id)
        if node_id not in claims:
            raise ProtocolError(f"Node {node_id} has not claimed round {round_id}", code="ERR_NOT_CLAIMED")

        # Check base model hash
        if meta.base_model_hash != round_info["model_hash_in"]:
            raise ProtocolError("Base model hash mismatch. Delta trained on outdated weights.", code="ERR_BASE_HASH_MISMATCH")

        # Prevent duplicate submissions in same round
        if node_id in self._pending_updates:
            raise ProtocolError("Duplicate update submission in the same round", code="ERR_DUPLICATE_SUBMISSION")

        # Deserialize and validate update
        delta = bytes_to_state_dict(delta_bytes)
        validate_update(delta, self.state.global_weights)

        delta_vec = flatten(delta)
        norm = l2_norm(delta_vec)

        # Record update in DB
        meta_json = json.dumps(meta.model_dump())
        self.db.add_update(
            round_id=round_id,
            node_id=node_id,
            meta_json=meta_json,
            norm=norm,
            cosine=1.0,
            accepted=True,
            reject_reason=None,
            byte_count=len(delta_bytes),
        )

        client_update = ClientUpdate(
            node_id=node_id,
            delta=delta,
            delta_vec=delta_vec,
            num_samples=meta.num_train_samples,
            meta=meta,
        )
        self._pending_updates[node_id] = client_update
        self.db.update_node(node_id, state="waiting", last_seen=now)
        self.db.record_bytes(node_id, "update_upload", bytes_in=len(delta_bytes), bytes_out=0)

    def process_heartbeat(self, node_id: str, state: str, reason: str | None, progress: float | None) -> None:
        now = time.time()
        self.db.update_node(node_id, state=state, last_seen=now)
        self.db.record_bytes(node_id, "control", bytes_in=64, bytes_out=32)

    def leave_node(self, node_id: str) -> None:
        self.db.update_node(node_id, state=NodeState.LEFT.value, last_seen=time.time())
        self.db.add_event("INFO", f"Node {node_id} left the session")