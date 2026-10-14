"""
Public FastAPI endpoints (/v1) for contributor node interaction with the Hive Host.
Strictly separates public federated protocol endpoints from local admin management.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, Response, StreamingResponse

from hive.core.errors import HiveError, ProtocolError
from hive.core.schemas import (
    Heartbeat,
    JoinRequest,
    JoinResponse,
    RoundInfo,
    SessionInfo,
    SessionState,
    UpdateMeta,
)
from hive.host.orchestrator import Orchestrator


class RateLimiter:
    """In-memory token bucket rate limiter per node ID."""

    def __init__(self, rate: float = 60.0, per_seconds: float = 60.0) -> None:
        self.capacity = rate
        self.refill_rate = rate / per_seconds
        self.buckets: dict[str, tuple[float, float]] = {}

    def is_allowed(self, key: str) -> tuple[bool, float]:
        now = time.time()
        tokens, last_update = self.buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last_update) * self.refill_rate)
        if tokens >= 1.0:
            self.buckets[key] = (tokens - 1.0, now)
            return True, 0.0
        retry_after = (1.0 - tokens) / self.refill_rate
        self.buckets[key] = (tokens, now)
        return False, retry_after


def create_public_api_router(orchestrator: Orchestrator, package_path: Path | str | None) -> APIRouter:
    router = APIRouter(prefix="/v1")
    limiter = RateLimiter(rate=120, per_seconds=60)

    # ---------------- Auth Dependency ----------------

    def authenticate_node(
        authorization: Annotated[str | None, Header()] = None,
    ) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "ERR_AUTH_MISSING", "message": "Missing Bearer authorization header"},
            )
        token = authorization[len("Bearer ") :].strip()
        if "." not in token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "ERR_AUTH_MALFORMED", "message": "Token must be formatted as node_id.node_secret"},
            )
        node_id, secret = token.split(".", 1)
        if not orchestrator.authenticate_node(node_id, secret):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "ERR_AUTH_FAILED", "message": "Invalid node credentials"},
            )

        # Rate limit check per node
        allowed, retry_after = limiter.is_allowed(node_id)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"code": "ERR_RATE_LIMIT", "message": "Rate limit exceeded"},
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
        return node_id

    # ---------------- Endpoints ----------------

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "session_status": orchestrator.state.status.value}

    @router.get("/session", response_model=SessionInfo)
    async def get_session(
        x_join_token: Annotated[str | None, Header()] = None,
    ) -> SessionInfo:
        if not x_join_token or x_join_token != orchestrator.state.join_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "ERR_INVALID_JOIN_TOKEN", "message": "Missing or invalid X-Join-Token header"},
            )
        db_sess = orchestrator.db.get_session()
        if not db_sess:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not initialized")
        return SessionInfo(
            session_id=db_sess["session_id"],
            name=db_sess["name"],
            description=db_sess["description"],
            data_requirements=db_sess["data_requirements"],
            package_digest=db_sess["package_digest"],
            host_key_fingerprint=db_sess["host_key_fingerprint"],
            status=orchestrator.state.status,
            total_rounds=db_sess["total_rounds"],
            current_round=db_sess["current_round"],
        )

    @router.post("/join", response_model=JoinResponse)
    async def join_session(
        request: JoinRequest,
        x_join_token: Annotated[str | None, Header()] = None,
    ) -> JoinResponse:
        if not x_join_token or x_join_token != orchestrator.state.join_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "ERR_INVALID_JOIN_TOKEN", "message": "Invalid join token"},
            )
        node_id, secret, approval = orchestrator.register_node(request.nickname, request.capabilities)
        return JoinResponse(node_id=node_id, node_secret=secret, approval=approval)

    @router.get("/package")
    async def download_package(node_id: str = Depends(authenticate_node)) -> Response:
        if not package_path or not Path(package_path).exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model package file not found")
        p = Path(package_path)
        data = p.read_bytes()
        orchestrator.db.record_bytes(node_id, "package_download", bytes_in=0, bytes_out=len(data))
        return Response(content=data, media_type="application/gzip")

    @router.post("/ready")
    async def declare_ready(
        data: dict[str, Any],
        node_id: str = Depends(authenticate_node),
    ) -> dict[str, str]:
        ok = data.get("data_ok", False)
        num_samples = int(data.get("num_samples", 0))
        if not ok or num_samples <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "ERR_INVALID_READY_STATUS", "message": "Contributor data validation must be successful with num_samples > 0"},
            )
        try:
            orchestrator.mark_node_ready(node_id, num_samples)
        except ProtocolError as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=e.to_dict())
        return {"status": "eligible"}

    @router.get("/round/current", response_model=RoundInfo)
    async def get_current_round(node_id: str = Depends(authenticate_node)) -> RoundInfo:
        rnd = orchestrator.db.get_latest_round()
        if not rnd or rnd["status"] != "open":
            return RoundInfo(status="none", round_id=0, retry_after_s=orchestrator.config.poll_interval_s)
        return RoundInfo(
            status="open",
            round_id=rnd["round_id"],
            deadline_ts=rnd["deadline"],
            model_hash=rnd["model_hash_in"],
            retry_after_s=orchestrator.config.poll_interval_s,
        )

    @router.get("/round/{round_id}/model")
    async def download_round_model(
        round_id: int,
        node_id: str = Depends(authenticate_node),
    ) -> Response:
        try:
            model_bytes = orchestrator.claim_model(node_id, round_id)
            return Response(content=model_bytes, media_type="application/octet-stream")
        except ProtocolError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=e.to_dict())

    @router.post("/round/{round_id}/update")
    async def upload_round_update(
        round_id: int,
        request: Request,
        node_id: str = Depends(authenticate_node),
    ) -> dict[str, bool]:
        # Parse multipart form data with strict allowlist
        form = await request.form()
        allowed_keys = {"meta", "delta"}
        extra_keys = set(form.keys()) - allowed_keys
        if extra_keys:
            # Reject non-conforming uploads (raw audio or extra files)
            orchestrator.db.add_update(
                round_id=round_id,
                node_id=node_id,
                meta_json="{}",
                norm=0.0,
                cosine=0.0,
                accepted=False,
                reject_reason=f"Forbidden multipart fields: {list(extra_keys)}",
                byte_count=0,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "ERR_FORBIDDEN_FIELDS", "message": f"Multipart upload contains extra forbidden parts: {list(extra_keys)}"},
            )

        meta_raw = form.get("meta")
        delta_file = form.get("delta")

        if not meta_raw or not delta_file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "ERR_MISSING_PARTS", "message": "Multipart form must contain 'meta' JSON and 'delta' file"},
            )

        try:
            meta_dict = json.loads(str(meta_raw))
            meta = UpdateMeta(**meta_dict)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "ERR_INVALID_META", "message": f"Invalid update metadata: {e}"},
            )

        if isinstance(delta_file, UploadFile):
            delta_bytes = await delta_file.read()
        elif isinstance(delta_file, bytes):
            delta_bytes = delta_file
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid delta file object")

        try:
            orchestrator.submit_update(node_id, round_id, meta, delta_bytes)
        except ProtocolError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=e.to_dict())

        return {"accepted": True}

    @router.post("/heartbeat")
    async def heartbeat(
        hb: Heartbeat,
        node_id: str = Depends(authenticate_node),
    ) -> dict[str, str]:
        orchestrator.process_heartbeat(node_id, hb.state, hb.reason, hb.progress)
        return {"status": "ok"}

    @router.post("/leave")
    async def leave(node_id: str = Depends(authenticate_node)) -> dict[str, bool]:
        orchestrator.leave_node(node_id)
        return {"left": True}

    return router