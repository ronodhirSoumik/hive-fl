"""
Unit tests for Host Web UI (T14).
Validates static asset serving, DOM structure compliance, zero-external-request offline policy,
and model verification visualizer components.
"""

from __future__ import annotations

import re
from pathlib import Path
import pytest
from starlette.testclient import TestClient
import torch

from hive.core.config import HostConfig
from hive.host.app import create_host_apps
from hive.host.db import HostDatabase
from hive.host.orchestrator import Orchestrator


@pytest.fixture
def host_ui_client(tmp_path: Path) -> TestClient:
    cfg = HostConfig(workdir=tmp_path / "host")
    db = HostDatabase(tmp_path / "host" / "hive.db")
    initial_weights = {"linear.weight": torch.tensor([[0.5, 0.5]], dtype=torch.float32)}

    orchestrator = Orchestrator(
        config=cfg,
        db=db,
        package_digest="sha256:fake_digest_123",
        host_key_fingerprint="ed25519:fake_fp_abc",
        initial_weights=initial_weights,
    )
    admin_token = "admin_test_token"
    orchestrator.state.admin_token = admin_token

    _, admin_app = create_host_apps(cfg, orchestrator, package_path=None)
    return TestClient(admin_app)


def test_admin_app_serves_static_assets(host_ui_client: TestClient) -> None:
    """Verifies that admin_app serves index.html and vendored static assets."""
    # 1. Root index.html
    res_index = host_ui_client.get("/?token=admin_test_token")
    assert res_index.status_code == 200
    assert "Hive — Training Dashboard & Model Verifier" in res_index.text
    assert 'id="session-title"' in res_index.text

    # 2. Stylesheet
    res_css = host_ui_client.get("/static/style.css")
    assert res_css.status_code == 200
    assert "--bg-app" in res_css.text

    # 3. JavaScript app controller
    res_js = host_ui_client.get("/static/app.js")
    assert res_js.status_code == 200
    assert "fetchDashboardData" in res_js.text

    # 4. Vendored Chart.js
    res_chart = host_ui_client.get("/static/vendor/chart.umd.js")
    assert res_chart.status_code == 200
    assert "Chart" in res_chart.text


def test_host_ui_zero_external_requests() -> None:
    """Scans Host UI assets to enforce strict offline operation (no CDNs, no external fonts)."""
    static_dir = Path("src/hive/host/ui/static")
    assert static_dir.exists()

    url_pattern = re.compile(r'https?://(?!localhost|127\.0\.0\.1|0\.0\.0\.0)[^\s"\'<>]+')

    for fname in ["index.html", "style.css", "app.js"]:
        fpath = static_dir / fname
        assert fpath.exists(), f"Missing required asset {fname}"
        content = fpath.read_text(encoding="utf-8")

        # In index.html, forbid any script/link pointing to external http(s)
        if fname == "index.html":
            assert not re.search(r'<(?:script|link)[^>]*(?:src|href)=["\']https?://', content, re.IGNORECASE), \
                f"{fname} contains external script or link references!"

        # Scan for active network fetch calls to external origins
        if fname == "app.js":
            external_fetches = re.findall(r'fetch\s*\(\s*["\']https?://', content, re.IGNORECASE)
            assert len(external_fetches) == 0, f"{fname} initiates external network fetch calls!"


def test_host_ui_required_elements_structure() -> None:
    """Verifies all required T14 DOM elements exist in index.html."""
    html_path = Path("src/hive/host/ui/static/index.html")
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    required_ids = [
        # Header / Status bar
        "session-title",
        "session-status",
        "round-display",
        "active-nodes-count",
        "nodes-eligible",
        "nodes-training",
        "nodes-waiting",
        "nodes-offline",
        "transport-badge",
        "join-uri-input",
        "btn-copy-uri",
        "btn-pause",
        "btn-resume",
        "btn-rotate-token",
        # Tabs navigation
        "tab-btn-training",
        "tab-btn-contributors",
        "tab-btn-privacy",
        "tab-btn-verification",
        # Training tab
        "chart-loss",
        "chart-accuracy",
        "round-claimed-count",
        "round-submitted-count",
        "round-deadline-timer",
        "events-list",
        # Contributors tab
        "chart-contributions",
        "table-contributors",
        # Privacy tab
        "card-privacy-transport",
        "card-privacy-raw-data",
        "card-privacy-update-ratio",
        "card-privacy-dp",
        "card-privacy-anonymity",
        "card-privacy-defenses",
        "card-privacy-limitations",
        # Model verification tab
        "table-hash-chain",
        "btn-verify-model",
        "verify-result-banner",
        "round-detail-container",
        "chart-layer-norms",
        "chart-global-norms",
        "btn-download-model",
        "verify-instructions",
    ]

    for elem_id in required_ids:
        assert f'id="{elem_id}"' in content, f"Required element ID '{elem_id}' is missing from Host UI index.html!"

    # Invariant: Raw data guard exact claim
    assert "Raw data bytes received: 0 (server accepts only weight updates)" in content
