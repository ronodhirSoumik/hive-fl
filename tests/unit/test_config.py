"""
Unit tests for configuration loading and precedence rules:
Precedence: defaults < config file < environment variables < CLI overrides.
"""

import json
from pathlib import Path
from hive.core.config import HostConfig, ClientConfig, load_host_config, load_client_config


def test_host_config_defaults() -> None:
    """Verifies default values match the Section 10 specification."""
    cfg = HostConfig()
    assert cfg.api_port == 9000
    assert cfg.admin_port == 8000
    assert cfg.total_rounds == 10
    assert cfg.min_clients == 2
    assert cfg.max_clients_per_round == 50
    assert cfg.round_timeout_s == 600
    assert cfg.aggregation == "fedavg"
    assert cfg.server_clip_norm is None
    assert cfg.trim_ratio == 0.2
    assert cfg.heartbeat_s == 10
    assert cfg.offline_after_s == 30
    assert cfg.poll_interval_s == 3
    assert cfg.require_approval is False
    assert cfg.tunnel == "none"


def test_client_config_defaults() -> None:
    """Verifies client default values match Section 8 & 10."""
    cfg = ClientConfig()
    assert cfg.ui_port == 8765
    assert cfg.poll_interval_s == 3
    assert cfg.heartbeat_s == 10
    assert cfg.sandbox_mode == "auto"
    assert cfg.max_cpu_percent == 50
    assert cfg.only_when_idle is True
    assert cfg.idle_cpu_threshold == 25.0
    assert cfg.idle_minutes == 2.0
    assert cfg.max_ram_mb == 2048
    assert cfg.use_gpu is False
    assert cfg.pause_on_battery is True
    assert cfg.max_rounds == 0


def test_config_precedence(tmp_path: Path) -> None:
    """
    Tests full precedence ladder:
    defaults (9000) < file (9001) < env (9002) < CLI (9003)
    """
    # 1. Base defaults
    cfg_default = load_host_config()
    assert cfg_default.api_port == 9000
    assert cfg_default.total_rounds == 10

    # 2. File overrides defaults
    config_file = tmp_path / "host_config.json"
    config_file.write_text(json.dumps({
        "api_port": 9001,
        "total_rounds": 15,
        "min_clients": 3
    }))
    cfg_file = load_host_config(config_file=config_file)
    assert cfg_file.api_port == 9001
    assert cfg_file.total_rounds == 15
    assert cfg_file.min_clients == 3

    # 3. Environment overrides file and defaults
    fake_env = {
        "HIVE_API_PORT": "9002",
        "HIVE_HOST_TOTAL_ROUNDS": "20",
    }
    cfg_env = load_host_config(config_file=config_file, env=fake_env)
    assert cfg_env.api_port == 9002
    assert cfg_env.total_rounds == 20
    assert cfg_env.min_clients == 3  # Kept from file

    # 4. CLI overrides everything
    cli_overrides = {
        "api_port": 9003,
        "min_clients": 5
    }
    cfg_cli = load_host_config(config_file=config_file, env=fake_env, cli_overrides=cli_overrides)
    assert cfg_cli.api_port == 9003  # From CLI
    assert cfg_cli.total_rounds == 20  # From env
    assert cfg_cli.min_clients == 5  # From CLI


def test_client_config_precedence(tmp_path: Path) -> None:
    """Tests precedence for ClientConfig."""
    config_file = tmp_path / "client.json"
    config_file.write_text(json.dumps({
        "max_cpu_percent": 30,
        "ui_port": 8888
    }))
    fake_env = {"HIVE_CLIENT_MAX_CPU_PERCENT": "40"}
    cli_args = {"ui_port": 9999}

    cfg = load_client_config(config_file=config_file, env=fake_env, cli_overrides=cli_args)
    assert cfg.max_cpu_percent == 40  # From env (overrode file 30)
    assert cfg.ui_port == 9999  # From CLI (overrode file 8888)
    assert cfg.only_when_idle is True  # Default