"""
Local Contributor Web API (127.0.0.1:8765).
Provides secure local endpoints for the Client Web UI with anti-CSRF,
DNS-rebinding protection, UI token authentication, and directory browsing.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Mapping

from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from hive.client.agent import ClientAgent
from hive.core.errors import HiveError
from hive.core.schemas import ResourcePolicy

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver"} | {
    h.strip() for h in os.getenv("HIVE_ALLOWED_HOSTS", "").split(",") if h.strip()
}


def create_local_app(agent: ClientAgent, ui_token: str | None = None) -> FastAPI:
    """
    Creates and configures the local Contributor FastAPI application.
    """
    app = FastAPI(title="Hive Local Client API", version="0.1.0")
    token = ui_token or secrets.token_urlsafe(24)

    # CORS configuration limited to localhost
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8765", "http://localhost:8765", "http://127.0.0.1:8766", "http://localhost:8766"],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------- Security Dependencies ----------------

    async def verify_host_and_origin(request: Request) -> None:
        """Protects against DNS rebinding and cross-site request forgery."""
        host_header = request.headers.get("host", "").split(":")[0].strip()
        if host_header and host_header not in ALLOWED_HOSTS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "ERR_DNS_REBINDING", "message": f"Forbidden host '{host_header}'"},
            )

        origin = request.headers.get("origin")
        if origin:
            clean_origin = origin.split("://")[-1].split(":")[0]
            if clean_origin not in ALLOWED_HOSTS:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"code": "ERR_INVALID_ORIGIN", "message": f"Cross-origin request forbidden from '{origin}'"},
                )

    async def authenticate_ui(
        request: Request,
        x_client_token: str | None = Header(None, alias="X-Client-Token"),
        token_query: str | None = Query(None, alias="token"),
        ui_token_cookie: str | None = Cookie(None, alias="ui_token"),
    ) -> None:
        """Verifies UI authentication token via header, cookie, or query parameter."""
        provided = x_client_token or token_query or ui_token_cookie
        auth_hdr = request.headers.get("authorization", "")
        if not provided and auth_hdr.startswith("Bearer "):
            provided = auth_hdr.replace("Bearer ", "").strip()

        if not provided or not secrets.compare_digest(provided, token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "ERR_UI_AUTH_FAILED", "message": "Missing or invalid UI token"},
            )

    # ---------------- Router Endpoints ----------------

    router = APIRouter(prefix="/local", dependencies=[Depends(verify_host_and_origin), Depends(authenticate_ui)])

    @router.get("/status")
    async def get_status() -> dict[str, Any]:
        """Returns comprehensive status of client agent and resource governor."""
        can_train, gating_reason = agent.resource_controller.can_train()
        return {
            "state": agent.state,
            "state_reason": agent.state_reason,
            "session_info": agent.session_info.model_dump() if agent.session_info else None,
            "node_id": agent.node_id,
            "nickname": agent.nickname,
            "data_dir": str(agent.data_dir) if agent.data_dir else None,
            "num_samples": agent.num_samples,
            "bytes_uploaded": agent.bytes_uploaded,
            "can_train": can_train,
            "gating_reason": gating_reason,
            "sandbox_mode": agent.sandbox_mode,
            "is_user_paused": agent.resource_controller.is_user_paused,
            "rounds_contributed": agent.resource_controller.rounds_contributed,
            "max_rounds": agent.resource_controller.policy.max_rounds,
        }

    @router.post("/connect")
    async def connect_session(payload: dict[str, Any]) -> dict[str, str]:
        """Initiates connection to a Host FL session."""
        uri = payload.get("uri", "").strip()
        nickname = payload.get("nickname", "Anonymous").strip()
        if not uri:
            raise HTTPException(status_code=400, detail={"code": "ERR_BAD_URI", "message": "URI cannot be empty"})
        try:
            await agent.connect(uri, nickname=nickname)
            return {"status": "ok", "state": agent.state}
        except HiveError as e:
            raise HTTPException(status_code=400, detail=e.to_dict())

    @router.get("/package-review")
    async def get_package_review() -> dict[str, Any]:
        """Provides full package inspection data, code sources, and static lint findings."""
        if not agent.package_manifest or not agent.extracted_pkg_dir:
            raise HTTPException(
                status_code=400,
                detail={"code": "ERR_NO_PACKAGE", "message": "No package downloaded yet"},
            )

        sources: dict[str, str] = {}
        for fname in ["model.py", "preprocess.py", "requirements.txt", "README.md"]:
            fpath = agent.extracted_pkg_dir / fname
            if fpath.exists() and fpath.is_file():
                try:
                    sources[fname] = fpath.read_text(encoding="utf-8")
                except Exception:
                    sources[fname] = "<binary or unreadable>"

        return {
            "manifest": agent.package_manifest.model_dump(),
            "content_digest": agent.package_manifest.content_digest,
            "pinned_fingerprint": agent.pinned_fingerprint,
            "lint_findings": agent.lint_findings,
            "sources": sources,
            "data_requirements": agent.package_manifest.data_requirements.model_dump(),
        }

    @router.post("/consent")
    async def grant_consent(payload: dict[str, Any] | None = None) -> dict[str, str]:
        """Grants user approval for current package digest."""
        digest = payload.get("digest") if payload else None
        try:
            agent.give_consent(digest)
            return {"status": "ok", "state": agent.state}
        except Exception as e:
            raise HTTPException(status_code=400, detail={"code": "ERR_CONSENT", "message": str(e)})

    @router.post("/data-dir")
    async def set_data_directory(payload: dict[str, Any]) -> dict[str, Any]:
        """Validates local data folder and declares readiness."""
        path = payload.get("path", "").strip()
        if not path:
            raise HTTPException(status_code=400, detail={"code": "ERR_EMPTY_PATH", "message": "Path is required"})
        try:
            res = await agent.validate_data(path)
            return res
        except Exception as e:
            raise HTTPException(status_code=400, detail={"code": "ERR_DATA_DIR", "message": str(e)})

    @router.get("/fs")
    async def browse_filesystem(path: str = Query("")) -> dict[str, Any]:
        """
        Secure directory browser for the local UI folder picker.
        Only traverses and returns directories (never file contents).
        """
        target = Path(path).resolve() if path else Path.home().resolve()
        if not target.exists() or not target.is_dir():
            target = Path.home().resolve()

        directories = []
        try:
            for item in sorted(target.iterdir()):
                if item.is_dir() and not item.name.startswith("."):
                    directories.append({
                        "name": item.name,
                        "path": str(item.resolve()),
                    })
        except (PermissionError, OSError):
            pass

        parent = str(target.parent.resolve()) if target.parent != target else None
        return {
            "current": str(target),
            "parent": parent,
            "directories": directories,
        }

    @router.get("/resources")
    async def get_resources() -> dict[str, Any]:
        """Returns active ResourcePolicy."""
        return agent.resource_controller.policy.model_dump()

    @router.put("/resources")
    async def update_resources(policy_dict: dict[str, Any]) -> dict[str, Any]:
        """Updates and persists ResourcePolicy live."""
        try:
            new_pol = ResourcePolicy.model_validate(policy_dict)
            agent.resource_controller.update_policy(new_pol)
            return new_pol.model_dump()
        except Exception as e:
            raise HTTPException(status_code=400, detail={"code": "ERR_INVALID_POLICY", "message": str(e)})

    @router.post("/pause")
    async def pause_training() -> dict[str, str]:
        """Pauses training."""
        agent.pause()
        return {"status": "ok", "state": agent.state}

    @router.post("/resume")
    async def resume_training() -> dict[str, str]:
        """Resumes training."""
        agent.resume()
        return {"status": "ok", "state": agent.state}

    @router.post("/leave")
    async def leave_session() -> dict[str, str]:
        """Gracefully leaves federated learning session."""
        await agent.leave()
        return {"status": "ok", "state": agent.state}

    @router.get("/history")
    async def get_history() -> list[dict[str, Any]]:
        """Returns history of contributed rounds and metrics."""
        return agent.history

    @router.get("/payload-preview")
    async def get_payload_preview() -> dict[str, Any]:
        """
        Displays transparency information about what weights leave the device.
        Includes exact tensor shapes, total bytes, SHA-256 digest, and honest limitations text.
        """
        return {
            "preview": agent.payload_preview,
            "statement": "Raw data files never leave your device. Only model parameter weight updates are transmitted.",
            "limitations": "Model weight updates can theoretically leak properties of local datasets. Differential Privacy can be enabled to add bounded noise.",
        }

    @router.get("/logs")
    async def get_logs() -> list[dict[str, Any]]:
        """Returns recent client event and state transition logs."""
        return agent.log_records

    static_dir = Path(__file__).parent / "ui" / "static"
    if static_dir.exists():
        @app.get("/static/vendor/chart.umd.js")
        async def client_chart_alias() -> FileResponse:
            js_file = static_dir / "vendor" / "chart.umd.js"
            if js_file.exists():
                return FileResponse(js_file)
            return FileResponse(static_dir / "vendor" / "chart.umd.min.js")

        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        async def client_index(token: str | None = None) -> FileResponse:
            resp = FileResponse(static_dir / "index.html")
            if token:
                resp.set_cookie(key="ui_token", value=token, httponly=False, samesite="lax")
            return resp

    app.include_router(router)
    return app
