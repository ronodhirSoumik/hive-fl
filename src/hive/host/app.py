"""
Application assembly for the Hive Host.
Builds and serves both the Public FL API and the localized Admin Dashboard API.
"""

from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from hive.core.config import HostConfig
from hive.core.errors import HiveError
from hive.core.logging import get_logger
from hive.core.tensors import bytes_to_state_dict
from hive.host.admin_api import create_admin_api_router
from hive.host.db import HostDatabase
from hive.host.orchestrator import Orchestrator
from hive.host.public_api import create_public_api_router
from hive.host.tunnel import TunnelManager
from hive.package.reader import inspect_package

logger = get_logger("hive.host")


def create_host_apps(
    config: HostConfig,
    orchestrator: Orchestrator,
    package_path: Path | str | None,
) -> tuple[FastAPI, FastAPI]:
    """
    Creates two separate, isolated FastAPI applications:
    1. public_app: Bound to 0.0.0.0:api_port, exposes only /v1/* endpoints.
    2. admin_app: Bound to 127.0.0.1:admin_port, exposes /admin/api/* and dashboard.
    """
    # ---------------- 1. Public App ----------------
    public_app = FastAPI(
        title="Hive Public FL API",
        version="0.1.0",
        docs_url=None,  # No Swagger on public port
        redoc_url=None,
    )

    # Allow client connections
    public_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @public_app.exception_handler(HiveError)
    async def hive_error_handler(_: Request, exc: HiveError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=exc.to_dict(),
        )

    public_router = create_public_api_router(orchestrator, package_path)
    public_app.include_router(public_router)

    # ---------------- 2. Admin App ----------------
    admin_app = FastAPI(
        title="Hive Admin Management API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    static_dir = Path(__file__).parent / "ui" / "static"
    if static_dir.exists():
        @admin_app.get("/static/vendor/chart.umd.js")
        async def chart_alias() -> FileResponse:
            js_file = static_dir / "vendor" / "chart.umd.js"
            if js_file.exists():
                return FileResponse(js_file)
            return FileResponse(static_dir / "vendor" / "chart.umd.min.js")

        admin_app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @admin_app.get("/")
        async def admin_index(token: str | None = None) -> FileResponse:
            resp = FileResponse(static_dir / "index.html")
            if token:
                resp.set_cookie(key="hive_admin_token", value=token, httponly=False, samesite="lax")
            return resp

    admin_router = create_admin_api_router(orchestrator)
    admin_app.include_router(admin_router)

    return public_app, admin_app


async def run_host(
    config: HostConfig,
    package_path: Path | str,
    resume: bool = False,
) -> None:
    """
    Main entrypoint for running the Hive Host server.
    Initializes DB, tunnel, orchestrator, and launches public and admin servers.
    """
    pkg_p = Path(package_path).resolve()
    if not pkg_p.exists():
        raise FileNotFoundError(f"Package file not found: {pkg_p}")

    info = inspect_package(pkg_p)
    package_digest = info["content_digest"]
    host_key_fp = info["author_fingerprint"]

    db_path = Path(config.workdir) / "hive.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = HostDatabase(db_path)

    # Extract initial weights or generate synthetic placeholder for round 0
    # If checkpoints exist, orchestrator start(resume=resume) will load them
    initial_weights = {"placeholder.weight": torch.zeros((1, 1), dtype=torch.float32)}

    orchestrator = Orchestrator(
        config=config,
        db=db,
        package_digest=package_digest,
        host_key_fingerprint=host_key_fp,
        initial_weights=initial_weights,
    )

    # Generate random admin token
    admin_token = secrets.token_urlsafe(16)
    orchestrator.state.admin_token = admin_token

    # Start Tunnel Manager
    tunnel_mgr = TunnelManager(
        mode=config.tunnel,
        api_port=config.api_port,
        manual_public_url=config.public_url,
    )
    public_url = await tunnel_mgr.start()
    orchestrator.state.public_url = public_url
    orchestrator.state.is_tls = tunnel_mgr.is_tls

    # Start Orchestrator
    await orchestrator.start(resume=resume)

    # Build canonical join URI
    session_data = db.get_session() or {}
    session_id = session_data.get("session_id", "sess_default")
    join_token = orchestrator.state.join_token

    join_uri = tunnel_mgr.build_join_uri(
        session_id=session_id,
        join_token=join_token,
        fingerprint=host_key_fp,
    )

    admin_url = f"http://127.0.0.1:{config.admin_port}/?token={admin_token}"

    print("\n" + "=" * 70)
    print("🐝  HIVE FEDERATED LEARNING HOST SERVER STARTED")
    print("=" * 70)
    print(f"Session ID:         {session_id}")
    print(f"Package Name:       {info['package_name']} v{info['version']}")
    print(f"Package Digest:     {package_digest[:24]}...")
    print(f"Author Fingerprint: {host_key_fp}")
    print(f"Transport:          {public_url} ({'HTTPS' if tunnel_mgr.is_tls else 'HTTP'})")
    print("-" * 70)
    print(f"📌 JOIN URI (give to contributors):")
    print(f"   {join_uri}")
    print("-" * 70)
    print(f"📊 ADMIN DASHBOARD (Host UI):")
    print(f"   {admin_url}")
    print("=" * 70 + "\n")

    public_app, admin_app = create_host_apps(config, orchestrator, pkg_p)

    pub_config = uvicorn.Config(public_app, host="0.0.0.0", port=config.api_port, log_level="warning")
    admin_config = uvicorn.Config(admin_app, host="127.0.0.1", port=config.admin_port, log_level="warning")

    pub_server = uvicorn.Server(pub_config)
    admin_server = uvicorn.Server(admin_config)

    try:
        await asyncio.gather(pub_server.serve(), admin_server.serve())
    finally:
        await orchestrator.stop()
        tunnel_mgr.stop()