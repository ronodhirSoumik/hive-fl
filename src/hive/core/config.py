"""
Configuration management for Hive.
Enforces configuration precedence: defaults < config file < environment variables < CLI overrides.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
from pydantic import BaseModel, Field


class HostConfig(BaseModel):
    """Host server configuration with defaults from Section 10."""
    api_port: int = Field(default=9000, description="Public FL API port")
    admin_port: int = Field(default=8000, description="Admin UI and API port (bound to 127.0.0.1)")
    total_rounds: int = Field(default=10, description="Total number of federated rounds")
    min_clients: int = Field(default=2, description="Minimum clients required to aggregate a round")
    max_clients_per_round: int = Field(default=50, description="Maximum clients admitted per round")
    round_timeout_s: int = Field(default=600, description="Round deadline timeout in seconds")
    aggregation: str = Field(default="fedavg", description="Aggregation strategy: fedavg, trimmed-mean, median, krum, multi-krum")
    server_clip_norm: float | None = Field(default=None, description="Server-side L2 clipping norm on delta")
    trim_ratio: float = Field(default=0.2, description="Trim ratio for trimmed-mean aggregator")
    heartbeat_s: int = Field(default=10, description="Heartbeat cadence in seconds")
    offline_after_s: int = Field(default=30, description="Heartbeat timeout marking node offline")
    poll_interval_s: int = Field(default=3, description="Client polling interval recommended by host")
    workdir: Path = Field(default=Path(".hive/host"), description="Storage directory for session, DB and checkpoints")
    require_approval: bool = Field(default=False, description="Require manual admin approval before node joins")
    tunnel: str = Field(default="none", description="NAT traversal tunnel provider: none, cloudflared, manual")
    public_url: str | None = Field(default=None, description="Override public URL for clients")
    package_path: str | None = Field(default=None, description="Path to .hivepkg model package")


class ClientConfig(BaseModel):
    """Client agent configuration with defaults from Section 8 & 10."""
    ui_port: int = Field(default=8765, description="Local web UI port (bound to 127.0.0.1)")
    poll_interval_s: int = Field(default=3, description="Polling interval in seconds")
    heartbeat_s: int = Field(default=10, description="Heartbeat interval in seconds")
    sandbox_mode: str = Field(default="auto", description="Sandbox mode: auto, docker, process")
    max_cpu_percent: int = Field(default=50, description="Max allowed CPU utilization percentage")
    only_when_idle: bool = Field(default=True, description="Pause training when device is in active user use")
    idle_cpu_threshold: float = Field(default=25.0, description="CPU usage threshold below which system is idle")
    idle_minutes: float = Field(default=2.0, description="Minutes of no user input to consider machine idle")
    max_ram_mb: int = Field(default=2048, description="Max RAM memory limit in MB")
    use_gpu: bool = Field(default=False, description="Allow GPU usage in worker if available")
    pause_on_battery: bool = Field(default=True, description="Pause training while running on battery power")
    max_rounds: int = Field(default=0, description="Max rounds to participate in before leaving (0=unlimited)")
    workdir: Path = Field(default=Path(".hive/client"), description="Working directory for client caches and job dirs")
    data_dir: str | None = Field(default=None, description="Local training data folder")
    auto_approve_fingerprint: str | None = Field(default=None, description="Auto-approve host key fingerprint matching URI")
    nickname: str = Field(default="AnonymousNode", description="Node nickname displayed on dashboard")
    uri: str | None = Field(default=None, description="Host session join URI")


def _read_file_config(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    if p.suffix.lower() == ".json" or raw.startswith("{"):
        return json.loads(raw)
    
    # Simple line-based KEY=VALUE or YAML-like fallback for flat config
    result: dict[str, Any] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
        elif "=" in line:
            k, v = line.split("=", 1)
        else:
            continue
        k = k.strip()
        v = v.strip().strip("\"'")
        # Try parse number or bool
        if v.lower() == "true":
            result[k] = True
        elif v.lower() == "false":
            result[k] = False
        elif v.lower() in ("null", "none"):
            result[k] = None
        else:
            try:
                result[k] = int(v)
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
    return result


def _extract_env_overrides(prefix: str, allowed_keys: set[str], env: Mapping[str, str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for raw_k, v in env.items():
        k = raw_k.upper()
        target_field = None
        if k.startswith(f"HIVE_{prefix.upper()}_"):
            cand = k[len(f"HIVE_{prefix.upper()}_"):].lower()
            if cand in allowed_keys:
                target_field = cand
        elif k.startswith("HIVE_"):
            cand = k[len("HIVE_"):].lower()
            if cand in allowed_keys:
                target_field = cand

        if target_field:
            if v.lower() == "true":
                overrides[target_field] = True
            elif v.lower() == "false":
                overrides[target_field] = False
            elif v.lower() in ("null", "none"):
                overrides[target_field] = None
            else:
                try:
                    overrides[target_field] = int(v)
                except ValueError:
                    try:
                        overrides[target_field] = float(v)
                    except ValueError:
                        overrides[target_field] = v
    return overrides


def load_host_config(
    config_file: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> HostConfig:
    """
    Loads HostConfig respecting precedence:
    defaults < config_file < env vars (HIVE_HOST_* / HIVE_*) < cli_overrides
    """
    merged: dict[str, Any] = {}
    if config_file:
        merged.update(_read_file_config(config_file))

    allowed = set(HostConfig.model_fields.keys())
    env_source = os.environ if env is None else env
    merged.update(_extract_env_overrides("HOST", allowed, env_source))

    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None and k in allowed:
                merged[k] = v

    return HostConfig(**merged)


def load_client_config(
    config_file: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> ClientConfig:
    """
    Loads ClientConfig respecting precedence:
    defaults < config_file < env vars (HIVE_CLIENT_* / HIVE_*) < cli_overrides
    """
    merged: dict[str, Any] = {}
    if config_file:
        merged.update(_read_file_config(config_file))

    allowed = set(ClientConfig.model_fields.keys())
    env_source = os.environ if env is None else env
    merged.update(_extract_env_overrides("CLIENT", allowed, env_source))

    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None and k in allowed:
                merged[k] = v

    return ClientConfig(**merged)