"""
Admin API (/admin/api) for the Hive Host training dashboard and model verification.
Runs strictly on 127.0.0.1:admin_port with CSRF checks and admin token authentication.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse, JSONResponse

from hive.core.hashing import model_hash
from hive.core.schemas import NodeState, SessionState
from hive.core.tensors import bytes_to_state_dict, state_dict_to_bytes
from hive.host.metrics import get_contribution_metrics
from hive.host.orchestrator import Orchestrator
from hive.host.privacy import get_privacy_indicators


def create_admin_api_router(orchestrator: Orchestrator) -> APIRouter:
    router = APIRouter()

    # ---------------- Admin Auth Dependency ----------------

    def authenticate_admin(
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
        hive_admin_token: Annotated[str | None, Cookie()] = None,
        token: Annotated[str | None, Query()] = None,
    ) -> bool:
        provided = x_admin_token or hive_admin_token or token
        expected = orchestrator.state.admin_token
        if not provided or not secrets.compare_digest(provided, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "ERR_ADMIN_AUTH_FAILED", "message": "Invalid or missing admin authentication token"},
            )

        # CSRF check on state-changing methods (POST, PUT, DELETE)
        if request.method in ("POST", "PUT", "DELETE"):
            origin = request.headers.get("origin")
            host = request.headers.get("host")
            if origin:
                # Origin should match host
                cleaned_origin = origin.replace("http://", "").replace("https://", "")
                if host and cleaned_origin != host:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail={"code": "ERR_CSRF_FAILED", "message": "Origin header mismatch"},
                    )
        return True

    # ---------------- Dashboard & Management Endpoints ----------------

    @router.get("/admin/api/summary")
    async def get_summary(_: bool = Depends(authenticate_admin)) -> dict[str, Any]:
        sess = orchestrator.db.get_session() or {}
        nodes = orchestrator.db.list_nodes()
        latest_rnd = orchestrator.db.get_latest_round()
        metrics = orchestrator.db.list_metrics()

        # Breakdown of nodes by state
        node_counts = {
            "eligible": sum(1 for n in nodes if n["state"] == NodeState.ELIGIBLE.value),
            "training": sum(1 for n in nodes if n["state"] == NodeState.TRAINING.value),
            "waiting": sum(1 for n in nodes if n["state"] in ("waiting", "registered")),
            "offline": sum(1 for n in nodes if n["state"] == NodeState.OFFLINE.value),
            "total": len(nodes),
        }

        latest_metric = metrics[-1] if metrics else None

        return {
            "session": {
                "session_id": sess.get("session_id", ""),
                "name": sess.get("name", ""),
                "status": orchestrator.state.status.value,
                "total_rounds": sess.get("total_rounds", 0),
                "current_round": sess.get("current_round", 0),
                "join_token": orchestrator.state.join_token,
                "public_url": orchestrator.state.public_url,
                "is_tls": orchestrator.state.is_tls,
            },
            "active_nodes": node_counts,
            "current_round_info": {
                "round_id": latest_rnd["round_id"] if latest_rnd else 0,
                "status": latest_rnd["status"] if latest_rnd else "none",
                "deadline": latest_rnd["deadline"] if latest_rnd else 0.0,
                "claims_count": len(orchestrator.db.list_claims(latest_rnd["round_id"])) if latest_rnd else 0,
                "updates_count": len(orchestrator.db.list_updates(latest_rnd["round_id"])) if latest_rnd else 0,
            },
            "latest_metrics": latest_metric,
        }

    @router.get("/admin/api/rounds")
    async def list_rounds(_: bool = Depends(authenticate_admin)) -> list[dict[str, Any]]:
        rounds = orchestrator.db.list_rounds()
        metrics_by_round = {m["round_id"]: m for m in orchestrator.db.list_metrics()}
        result: list[dict[str, Any]] = []
        for r in rounds:
            r_dict = dict(r)
            r_dict["metrics"] = metrics_by_round.get(r["round_id"])
            result.append(r_dict)
        return result

    @router.get("/admin/api/rounds/{round_id}")
    async def get_round_detail(round_id: int, _: bool = Depends(authenticate_admin)) -> dict[str, Any]:
        rnd = orchestrator.db.get_round(round_id)
        if not rnd:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Round not found")
        updates = orchestrator.db.list_updates(round_id)
        claims = orchestrator.db.list_claims(round_id)
        metric = orchestrator.db.get_metrics(round_id) if hasattr(orchestrator.db, "get_metrics") else None

        return {
            "round": rnd,
            "claims": claims,
            "updates": updates,
            "metrics": metric,
        }

    @router.get("/admin/api/nodes")
    async def list_nodes_metrics(_: bool = Depends(authenticate_admin)) -> list[dict[str, Any]]:
        return get_contribution_metrics(orchestrator.db)

    @router.get("/admin/api/privacy")
    async def get_privacy(_: bool = Depends(authenticate_admin)) -> dict[str, Any]:
        model_size = len(state_dict_to_bytes(orchestrator.state.global_weights))
        return get_privacy_indicators(
            orchestrator.db,
            orchestrator.state,
            min_clients=orchestrator.config.min_clients,
            model_size_bytes=model_size,
        )

    @router.get("/admin/api/events")
    async def list_events(_: bool = Depends(authenticate_admin)) -> list[dict[str, Any]]:
        return orchestrator.db.list_events(limit=50)

    # ---------------- Node Administration ----------------

    @router.post("/admin/api/nodes/{node_id}/approve")
    async def approve_node(node_id: str, _: bool = Depends(authenticate_admin)) -> dict[str, bool]:
        node = orchestrator.db.get_node(node_id)
        if not node:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")
        orchestrator.db.update_node(node_id, approved=1, state=NodeState.ELIGIBLE.value)
        orchestrator.db.add_event("INFO", f"Administrator approved node '{node['nickname']}'")
        return {"approved": True}

    @router.post("/admin/api/nodes/{node_id}/kick")
    async def kick_node(node_id: str, _: bool = Depends(authenticate_admin)) -> dict[str, bool]:
        node = orchestrator.db.get_node(node_id)
        if not node:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")
        orchestrator.db.update_node(node_id, state=NodeState.LEFT.value)
        orchestrator.db.add_event("WARNING", f"Administrator kicked node '{node['nickname']}'")
        return {"kicked": True}

    # ---------------- Session Control ----------------

    @router.post("/admin/api/session/pause")
    async def pause_session(_: bool = Depends(authenticate_admin)) -> dict[str, str]:
        orchestrator.state.status = SessionState.PAUSED
        orchestrator.db.update_session(status=SessionState.PAUSED.value)
        orchestrator.db.add_event("INFO", "Session paused by administrator")
        return {"status": "paused"}

    @router.post("/admin/api/session/resume")
    async def resume_session(_: bool = Depends(authenticate_admin)) -> dict[str, str]:
        orchestrator.state.status = SessionState.RUNNING
        orchestrator.db.update_session(status=SessionState.RUNNING.value)
        orchestrator.db.add_event("INFO", "Session resumed by administrator")
        return {"status": "running"}

    @router.post("/admin/api/session/rotate-token")
    async def rotate_token(_: bool = Depends(authenticate_admin)) -> dict[str, str]:
        new_token = secrets.token_urlsafe(16)
        orchestrator.state.join_token = new_token
        orchestrator.db.update_session(join_token=new_token)
        orchestrator.db.add_event("INFO", "Session join token rotated by administrator")
        return {"join_token": new_token}

    # ---------------- Weight Verification & Download ----------------

    @router.post("/model/verify")
    async def verify_checkpoint(
        round_id: int | None = None,
        _: bool = Depends(authenticate_admin),
    ) -> dict[str, Any]:
        """
        Recomputes the SHA-256 hash of a model checkpoint file from disk
        and compares with the database chain to detect tampering or corruption.
        """
        if round_id is not None:
            ckpt_path = orchestrator.checkpoints_dir / f"round_{round_id}.safetensors"
            rnd = orchestrator.db.get_round(round_id)
            expected_hash = rnd["model_hash_out"] if rnd else None
        else:
            ckpt_path = orchestrator.workdir / "final_model.safetensors"
            if not ckpt_path.exists():
                latest = orchestrator.db.get_latest_round()
                if latest:
                    ckpt_path = orchestrator.checkpoints_dir / f"round_{latest['round_id']}.safetensors"
                    expected_hash = latest["model_hash_out"]
                else:
                    expected_hash = orchestrator.state.model_hash
            else:
                expected_hash = orchestrator.state.model_hash

        if not ckpt_path.exists():
            return {
                "verified": False,
                "error": f"Checkpoint file '{ckpt_path.name}' does not exist on disk",
                "match": False,
            }

        try:
            sd = bytes_to_state_dict(ckpt_path.read_bytes())
            computed_hash = model_hash(sd)
        except Exception as e:
            return {"verified": False, "error": f"Failed reading safetensors checkpoint: {e}", "match": False}

        match = (computed_hash == expected_hash) if expected_hash else True
        return {
            "verified": True,
            "target_file": ckpt_path.name,
            "computed_model_hash": computed_hash,
            "expected_model_hash": expected_hash,
            "match": match,
            "tampered": not match,
        }

    @router.get("/model/download")
    async def download_model(_: bool = Depends(authenticate_admin)) -> Response:
        final_path = orchestrator.workdir / "final_model.safetensors"
        if final_path.exists():
            return Response(content=final_path.read_bytes(), media_type="application/octet-stream")
        latest = orchestrator.db.get_latest_round()
        if latest:
            ckpt = orchestrator.checkpoints_dir / f"round_{latest['round_id']}.safetensors"
            if ckpt.exists():
                return Response(content=ckpt.read_bytes(), media_type="application/octet-stream")
        # Return in-memory weights
        return Response(content=state_dict_to_bytes(orchestrator.state.global_weights), media_type="application/octet-stream")

    return router