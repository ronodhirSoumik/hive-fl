"""
Client Agent: State machine, host communication, and training orchestration brain.
Manages the join flow, consent verification, local dataset validation, round polling,
and weight delta submission.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import httpx
import psutil
import torch

from hive.client.config_store import ClientConfigStore, get_client_dir
from hive.client.resources import ResourceController
from hive.client.sandbox import get_sandbox_executor
from hive.client.sandbox.base import SandboxExecutor
from hive.core.errors import ConsentRequired, HiveError, PackageError, ProtocolError, SandboxError
from hive.core.hashing import model_hash
from hive.core.logging import get_logger
from hive.core.schemas import (
    Capabilities,
    Heartbeat,
    JoinRequest,
    JoinResponse,
    Manifest,
    NodeState,
    RoundInfo,
    SessionInfo,
    UpdateMeta,
)
from hive.core.tensors import bytes_to_state_dict, state_dict_to_bytes
from hive.core.uri import HiveURI, parse_hive_uri
from hive.package.lint import lint_directory
from hive.package.reader import extract_package

logger = get_logger("hive.client.agent")


def detect_capabilities() -> Capabilities:
    """Discovers local hardware configuration and software features."""
    cores = psutil.cpu_count(logical=True) or 2
    mem = psutil.virtual_memory()
    ram_mb = int(mem.total / (1024 * 1024))

    has_docker = False
    try:
        import docker
        client = docker.from_env()
        client.ping()
        has_docker = True
    except Exception:
        has_docker = False

    has_gpu = False
    try:
        has_gpu = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        has_gpu = False

    return Capabilities(
        os=sys.platform,
        cores=cores,
        has_docker=has_docker,
        has_gpu=has_gpu,
        ram_mb=ram_mb,
    )


class ClientAgent:
    """
    Main federated learning contributor agent.
    Coordinates server handshakes, consent gating, sandbox execution, and round life-cycle.
    """

    def __init__(
        self,
        work_dir: Path | str | None = None,
        config_store: ClientConfigStore | None = None,
        resource_controller: ResourceController | None = None,
        sandbox_mode: str = "auto",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.work_dir = Path(work_dir).resolve() if work_dir else get_client_dir()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.pkg_cache_dir = self.work_dir / "pkg_cache"
        self.pkg_cache_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir = self.work_dir / "extracted"
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.job_dir = self.work_dir / "job"

        self.config_store = config_store or ClientConfigStore(self.work_dir)
        self.resource_controller = resource_controller or ResourceController(config_store=self.config_store)
        self.sandbox_mode = sandbox_mode
        self._custom_client = http_client

        # Client runtime state
        self.state: str = "DISCONNECTED"
        self.state_reason: str = "Not connected"
        self.session_info: SessionInfo | None = None
        self.join_uri: str | None = None
        self.host_base_url: str | None = None
        self.pinned_fingerprint: str | None = None
        self.nickname: str = "Anonymous"
        self.node_id: str | None = None
        self.node_secret: str | None = None

        # Package state
        self.package_path: Path | None = None
        self.extracted_pkg_dir: Path | None = None
        self.package_manifest: Manifest | None = None
        self.lint_findings: list[dict[str, Any]] = []

        # Data state
        self.data_dir: Path | None = None
        self.num_samples: int = 0

        # Privacy and history
        self.bytes_uploaded: int = 0
        self.payload_preview: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self._completed_rounds: set[int] = set()

        # Logs
        self.log_records: list[dict[str, Any]] = []

        # Execution flags
        self._running: bool = False
        self._poll_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None

    def set_state(self, new_state: str, reason: str = "") -> None:
        """Transitions agent state and logs event."""
        old = self.state
        self.state = new_state
        self.state_reason = reason
        rec = {
            "ts": time.time(),
            "from_state": old,
            "to_state": new_state,
            "reason": reason,
        }
        self.log_records.append(rec)
        if len(self.log_records) > 200:
            self.log_records.pop(0)
        logger.info(f"Agent state transition: from {old} to {new_state} (reason: {reason})")

    def _get_http_client(self) -> httpx.AsyncClient:
        """Returns shared or new httpx AsyncClient."""
        if self._custom_client is not None:
            return self._custom_client
        return httpx.AsyncClient(timeout=30.0)

    def _auth_headers(self) -> dict[str, str]:
        """Constructs Bearer authentication header."""
        if self.node_id and self.node_secret:
            return {"Authorization": f"Bearer {self.node_id}.{self.node_secret}"}
        return {}

    # ---------------- Connection & Join Flow ----------------

    async def connect(self, uri: str, nickname: str = "Anonymous") -> None:
        """
        Connects to a Hive Host session using the canonical join URI.
        Downloads package, verifies crypto, lints code, and awaits consent.
        """
        self.join_uri = uri.strip()
        self.nickname = nickname.strip() or "Anonymous"
        parsed = parse_hive_uri(self.join_uri)

        self.host_base_url = parsed.base_http_url
        self.pinned_fingerprint = parsed.fingerprint
        self.set_state("CONNECTING", f"Connecting to {self.host_base_url}")

        client = self._get_http_client()
        try:
            # 1. Fetch public session info using X-Join-Token header
            sess_headers = {"X-Join-Token": parsed.join_token}
            sess_res = await client.get(f"{self.host_base_url}/v1/session", headers=sess_headers)
            if sess_res.status_code != 200:
                raise ProtocolError(f"Host returned status {sess_res.status_code} for session info", code="ERR_SESSION_FETCH")
            self.session_info = SessionInfo.model_validate(sess_res.json())

            # 2. Join session or reuse stored credentials
            stored_creds = self.config_store.load_node_credentials()
            if stored_creds and stored_creds.get("uri") == self.join_uri and stored_creds.get("node_id"):
                self.node_id = stored_creds["node_id"]
                self.node_secret = stored_creds.get("node_secret")
            else:
                join_req = JoinRequest(nickname=self.nickname, capabilities=detect_capabilities())
                headers = {"X-Join-Token": parsed.join_token}
                join_res = await client.post(f"{self.host_base_url}/v1/join", json=join_req.model_dump(), headers=headers)
                if join_res.status_code != 200:
                    raise ProtocolError(f"Join rejected by host: {join_res.text}", code="ERR_JOIN_REJECTED")

                join_data = join_res.json()
                self.node_id = join_data["node_id"]
                self.node_secret = join_data["node_secret"]
                self.config_store.save_node_credentials({
                    "uri": self.join_uri,
                    "node_id": self.node_id,
                    "node_secret": self.node_secret,
                })

            # 3. Download package archive
            pkg_res = await client.get(f"{self.host_base_url}/v1/package", headers=self._auth_headers())
            if pkg_res.status_code != 200:
                raise ProtocolError(f"Failed to download package: {pkg_res.status_code}", code="ERR_PACKAGE_DOWNLOAD")

            pkg_bytes = pkg_res.content
            digest_prefix = hashlib.sha256(pkg_bytes).hexdigest()[:12]
            self.package_path = self.pkg_cache_dir / f"{digest_prefix}.hivepkg"
            self.package_path.write_bytes(pkg_bytes)

            # 4. Safely verify digest, signature, and extract package
            target_extract_dir = self.extracted_dir / digest_prefix
            self.package_manifest = extract_package(
                self.package_path,
                target_extract_dir,
                expected_fingerprint=self.pinned_fingerprint,
            )
            self.extracted_pkg_dir = target_extract_dir

            # 5. Run static lint
            raw_findings = lint_directory(self.extracted_pkg_dir)
            self.lint_findings = [
                {"severity": f.severity, "file": f.file, "line": f.line, "message": f.message}
                for f in raw_findings
            ]

            # 6. Check user consent
            manifest_digest = self.package_manifest.content_digest
            if not self.config_store.is_digest_consented(manifest_digest):
                self.set_state("AWAITING_CONSENT", "Package downloaded. Awaiting user consent approval.")
                return

            # Already consented -> check data dir
            if self.data_dir and self.data_dir.exists():
                await self.validate_data(self.data_dir)
            else:
                self.set_state("AWAITING_DATA", "Package approved. Please select your local dataset directory.")

        except Exception as e:
            err_code = getattr(e, "code", "ERR_CLIENT_CONNECT")
            self.set_state("ERROR", f"Connection failed: {e}")
            logger.error(f"Client connect failed: {e} (code={err_code})")
            raise
        finally:
            if self._custom_client is None:
                await client.aclose()

    def give_consent(self, digest: str | None = None) -> None:
        """Records user consent approval for the current package digest."""
        if not self.package_manifest:
            raise ConsentRequired("No active package loaded to consent to.")

        target_digest = digest or self.package_manifest.content_digest
        self.config_store.save_consent(target_digest, {"approved": True, "ts": time.time()})
        logger.info(f"User granted explicit consent for package digest {target_digest}")

        if self.data_dir and self.data_dir.exists():
            asyncio.create_task(self.validate_data(self.data_dir))
        else:
            self.set_state("AWAITING_DATA", "Consent recorded. Select data folder to proceed.")

    async def validate_data(self, data_path: Path | str) -> dict[str, Any]:
        """
        Validates the local dataset directory against package preprocessor requirements.
        Communicates with the sandbox worker via the Job-Directory Protocol.
        """
        p = Path(data_path).resolve()
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Selected data directory does not exist: {p}")

        if not self.package_manifest or not self.extracted_pkg_dir:
            raise HiveError("Package must be downloaded and approved before data validation.")

        self.data_dir = p
        self.set_state("VALIDATING_DATA", f"Validating dataset at {p}")

        # Build validate task config
        worker_cfg = self.resource_controller.build_worker_config(task="validate")
        executor = get_sandbox_executor(mode=self.sandbox_mode, job_dir=self.job_dir)
        executor.prepare(
            worker_config=worker_cfg,
            pkg_source_dir=self.extracted_pkg_dir,
            data_dir=self.data_dir,
        )

        try:
            res = executor.run(timeout_s=60.0)
            local_msg_file = executor.out_dir / "local_message.txt"
            local_message = local_msg_file.read_text(encoding="utf-8") if local_msg_file.exists() else ""

            if res.status != "ok" or res.num_train_samples <= 0:
                self.set_state("ERROR", f"Data validation failed: {local_message}")
                return {"ok": False, "num_samples": 0, "message": local_message}

            self.num_samples = res.num_train_samples

            # Inform Host that contributor is ready and eligible
            client = self._get_http_client()
            try:
                ready_payload = {"data_ok": True, "num_samples": self.num_samples}
                resp = await client.post(
                    f"{self.host_base_url}/v1/ready",
                    json=ready_payload,
                    headers=self._auth_headers(),
                )
                if resp.status_code != 200:
                    raise ProtocolError(f"Host rejected ready declaration: {resp.text}", code="ERR_HOST_READY_REJECTED")
            finally:
                if self._custom_client is None:
                    await client.aclose()

            self.set_state("READY", f"Ready with {self.num_samples} local samples.")
            return {"ok": True, "num_samples": self.num_samples, "message": local_message}

        finally:
            executor.cleanup()

    # ---------------- Round Execution & Submission ----------------

    async def _execute_round(self, round_info: RoundInfo) -> None:
        """
        Downloads model, runs local training inside sandbox, computes weight delta,
        and uploads update to the Host.
        """
        can, reason = self.resource_controller.can_train()
        if not can:
            self.set_state("PAUSED_LOCAL", reason)
            return

        client = self._get_http_client()
        r_id = round_info.round_id
        try:
            # 1. Claim model weights by downloading
            model_res = await client.get(
                f"{self.host_base_url}/v1/round/{r_id}/model",
                headers=self._auth_headers(),
            )
            if model_res.status_code == 409:
                logger.info(f"Round {r_id} is full or closed; waiting for next round")
                return
            if model_res.status_code != 200:
                raise ProtocolError(f"Failed to claim/download model: {model_res.status_code}", code="ERR_MODEL_CLAIM")

            global_bytes = model_res.content
            global_weights = bytes_to_state_dict(global_bytes)

            # Verify announced model hash
            computed_hash = model_hash(global_weights)
            if round_info.model_hash and computed_hash != round_info.model_hash:
                raise ProtocolError(
                    f"Announced model hash ({round_info.model_hash}) does not match downloaded weights ({computed_hash})",
                    code="ERR_MODEL_HASH_MISMATCH",
                )

            # 2. Prepare Sandbox Execution
            self.set_state("TRAINING", f"Training round {r_id}")
            worker_cfg = self.resource_controller.build_worker_config(
                task="train",
                hyperparameters=round_info.hyperparameters,
                dp=round_info.dp_policy,
                seed=42 + r_id,
            )

            executor = get_sandbox_executor(
                mode=self.sandbox_mode,
                job_dir=self.job_dir,
                cpus=max(1.0, float(self.resource_controller.policy.max_cpu_percent) / 50.0),
                memory_mb=self.resource_controller.policy.max_ram_mb,
            )
            self.resource_controller.set_active_executor(executor)

            executor.prepare(
                worker_config=worker_cfg,
                pkg_source_dir=self.extracted_pkg_dir or "",
                data_dir=self.data_dir or "",
                global_weights=global_weights,
            )

            # 3. Execute Worker Task
            res = executor.run(timeout_s=600.0)
            if res.status != "ok":
                logger.error(f"Worker training failed: round={r_id}, code={res.error_code}")
                self.set_state("READY", f"Training failed with code: {res.error_code}")
                return

            # 4. Parse result and strictly validate update
            res, delta_dict, delta_bytes = executor.parse_result(template=global_weights)
            if not delta_bytes or delta_dict is None:
                raise SandboxError("Worker succeeded but produced no update delta", code="ERR_NO_DELTA")

            delta_sha = hashlib.sha256(delta_bytes).hexdigest()

            # Store payload preview for UI transparency
            first_tensors = [
                {"name": k, "shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in list(delta_dict.items())[:5]
            ]
            self.payload_preview = {
                "tensor_count": len(delta_dict),
                "bytes": len(delta_bytes),
                "sha256": delta_sha,
                "tensors": first_tensors,
                "dp": round_info.dp_policy.model_dump() if round_info.dp_policy else {},
            }

            # 5. Upload Multipart Update
            self.set_state("UPLOADING", f"Uploading update for round {r_id}")
            meta = UpdateMeta(
                round_id=r_id,
                base_model_hash=computed_hash,
                num_train_samples=res.num_train_samples,
                num_eval_samples=res.num_eval_samples,
                train_loss=res.train_loss,
                eval_loss=res.eval_loss,
                eval_metrics=res.eval_metrics,
                local_epochs=round_info.hyperparameters.get("local_epochs", 1),
                duration_s=res.duration_s,
                delta_sha256=delta_sha,
                dp=round_info.dp_policy.model_dump() if round_info.dp_policy else {},
            )

            data = {"meta": meta.model_dump_json()}
            files = {
                "delta": ("update.safetensors", delta_bytes, "application/octet-stream"),
            }
            up_res = await client.post(
                f"{self.host_base_url}/v1/round/{r_id}/update",
                data=data,
                files=files,
                headers=self._auth_headers(),
            )

            if up_res.status_code == 200:
                self.bytes_uploaded += len(delta_bytes)
                self._completed_rounds.add(r_id)
                self.resource_controller.record_round_completed()
                self.history.append({
                    "round_id": r_id,
                    "train_loss": res.train_loss,
                    "eval_loss": res.eval_loss,
                    "eval_metrics": res.eval_metrics,
                    "duration_s": res.duration_s,
                    "bytes_uploaded": len(delta_bytes),
                    "accepted": True,
                })
                logger.info(f"Successfully submitted round {r_id} update ({len(delta_bytes)} bytes)")
            elif up_res.status_code == 409:
                logger.warning(f"Update rejected (late / round closed) for round {r_id}")
            else:
                logger.error(f"Upload error: {up_res.status_code} {up_res.text}")

            # Check max contributed rounds
            if (
                self.resource_controller.policy.max_rounds > 0
                and self.resource_controller.rounds_contributed >= self.resource_controller.policy.max_rounds
            ):
                logger.info("Reached maximum configured contributed rounds. Leaving session.")
                await self.leave()
                return

            self.set_state("READY", "Round completed. Waiting for next round.")

        finally:
            if "executor" in locals():
                executor.cleanup()
            self.resource_controller.set_active_executor(None)
            if self._custom_client is None:
                await client.aclose()

    # ---------------- Background Loops & Lifecycle ----------------

    async def _heartbeat_loop(self) -> None:
        """Sends periodic heartbeats to Host server."""
        while self._running:
            if self.host_base_url and self.node_id and self.state not in ("DISCONNECTED", "LEFT", "ERROR"):
                try:
                    client = self._get_http_client()
                    hb_state = "waiting"
                    if self.state == "TRAINING":
                        hb_state = "training"
                    elif self.state == "UPLOADING":
                        hb_state = "uploading"
                    elif self.state == "PAUSED_LOCAL":
                        hb_state = "paused_local"

                    hb = Heartbeat(state=hb_state, reason=self.state_reason)  # type: ignore[arg-type]
                    await client.post(
                        f"{self.host_base_url}/v1/heartbeat",
                        json=hb.model_dump(),
                        headers=self._auth_headers(),
                    )
                    if self._custom_client is None:
                        await client.aclose()
                except Exception:
                    pass
            await asyncio.sleep(10.0)

    async def _round_poll_loop(self) -> None:
        """Polls Host /v1/round/current and triggers training when a new round is open."""
        backoff_s = 2.0
        while self._running:
            if self.host_base_url and self.state == "READY":
                try:
                    client = self._get_http_client()
                    resp = await client.get(
                        f"{self.host_base_url}/v1/round/current",
                        headers=self._auth_headers(),
                    )
                    if resp.status_code == 200:
                        round_info = RoundInfo.model_validate(resp.json())
                        if round_info.status == "open" and round_info.round_id not in self._completed_rounds:
                            await self._execute_round(round_info)
                        backoff_s = 2.0
                    if self._custom_client is None:
                        await client.aclose()
                except Exception as e:
                    logger.debug("Round poll encountered error; will retry", error=str(e))
                    backoff_s = min(30.0, backoff_s * 1.5)

            await asyncio.sleep(backoff_s)

    def start(self) -> None:
        """Starts background polling and resource governance monitors."""
        if self._running:
            return
        self._running = True
        self.resource_controller.start_monitor()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._poll_task = asyncio.create_task(self._round_poll_loop())
        logger.info("Client agent started")

    async def stop(self) -> None:
        """Stops background tasks."""
        self._running = False
        self.resource_controller.stop_monitor()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._poll_task:
            self._poll_task.cancel()
        logger.info("Client agent stopped")

    def pause(self) -> None:
        """Pauses training."""
        self.resource_controller.pause()
        self.set_state("PAUSED_LOCAL", "Paused by user")

    def resume(self) -> None:
        """Resumes training."""
        self.resource_controller.resume()
        if self.state == "PAUSED_LOCAL":
            self.set_state("READY", "Resumed by user")

    async def leave(self) -> None:
        """Leaves federated session gracefully, cleans cache, and disconnects."""
        if self.host_base_url and self.node_id:
            try:
                client = self._get_http_client()
                await client.post(f"{self.host_base_url}/v1/leave", headers=self._auth_headers())
                if self._custom_client is None:
                    await client.aclose()
            except Exception:
                pass

        await self.stop()
        self.config_store.clear_node_credentials()

        # Clean volatile directories
        if self.job_dir.exists():
            shutil.rmtree(self.job_dir, ignore_errors=True)
        if self.extracted_dir.exists():
            shutil.rmtree(self.extracted_dir, ignore_errors=True)

        self.set_state("LEFT", "Gracefully departed federated session")
