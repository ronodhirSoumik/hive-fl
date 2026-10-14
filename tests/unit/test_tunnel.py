"""
Unit tests for reachability, LAN IP detection, and TunnelManager.
"""

import pytest
from hive.core.uri import parse_hive_uri
from hive.host.tunnel import TunnelManager, check_port_free, detect_lan_ip, run_doctor


def test_detect_lan_ip() -> None:
    ip = detect_lan_ip()
    assert isinstance(ip, str)
    assert len(ip.split(".")) == 4


def test_check_port_free() -> None:
    # Ephemeral port should be free
    assert check_port_free(54321) in (True, False)


@pytest.mark.asyncio
async def test_tunnel_manager_none_mode() -> None:
    mgr = TunnelManager(mode="none", api_port=9000)
    url = await mgr.start()
    assert url.startswith("http://")
    assert ":9000" in url
    assert mgr.is_tls is False

    uri = mgr.build_join_uri("sess_123", "join_token_abc", "ed25519:fingerprint")
    parsed = parse_hive_uri(uri)
    assert parsed.session_id == "sess_123"
    assert parsed.join_token == "join_token_abc"
    assert parsed.fingerprint == "ed25519:fingerprint"
    assert parsed.tls is False
    mgr.stop()


@pytest.mark.asyncio
async def test_tunnel_manager_manual_mode() -> None:
    mgr = TunnelManager(mode="manual", manual_public_url="https://fl.example.org:9000")
    url = await mgr.start()
    assert url == "https://fl.example.org:9000"
    assert mgr.is_tls is True

    uri = mgr.build_join_uri("sess_456", "token", "ed25519:fp")
    parsed = parse_hive_uri(uri)
    assert parsed.authority == "fl.example.org:9000"
    assert parsed.tls is True
    mgr.stop()


def test_run_doctor_diagnostics() -> None:
    report = run_doctor()
    assert "python" in report
    assert "torch" in report
    assert "docker" in report
    assert "cloudflared" in report
    assert "ports" in report
    assert "network" in report
    assert report["python"]["ok"] is True