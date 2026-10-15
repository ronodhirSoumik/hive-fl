"""
Unit tests for Client Web UI (T15).
Validates static asset serving, stepper layout, code review viewer, resource control elements,
zero-external-request offline compliance, and payload transparency panel.
"""

from __future__ import annotations

import re
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from hive.client.agent import ClientAgent
from hive.client.config_store import ClientConfigStore
from hive.client.local_api import create_local_app
from hive.client.resources import ResourceController


@pytest.fixture
def client_ui_testclient(tmp_path: Path) -> TestClient:
    workdir = tmp_path / "client_ui_test"
    store = ClientConfigStore(workdir)
    res_ctrl = ResourceController(config_store=store)
    agent = ClientAgent(work_dir=workdir, config_store=store, resource_controller=res_ctrl)

    ui_token = "client_ui_secret_token"
    app = create_local_app(agent, ui_token=ui_token)
    return TestClient(app, base_url="http://127.0.0.1:8765")


def test_client_app_serves_static_assets(client_ui_testclient: TestClient) -> None:
    """Verifies that local_app serves index.html and vendored static assets."""
    # 1. Root index.html
    res_index = client_ui_testclient.get("/?token=client_ui_secret_token")
    assert res_index.status_code == 200
    assert "Hive — Contributor Client" in res_index.text
    assert 'id="screen-connect"' in res_index.text

    # 2. Stylesheet
    res_css = client_ui_testclient.get("/static/style.css")
    assert res_css.status_code == 200
    assert "--color-primary" in res_css.text

    # 3. JavaScript controller
    res_js = client_ui_testclient.get("/static/app.js")
    assert res_js.status_code == 200
    assert "loadPackageReview" in res_js.text


def test_client_ui_zero_external_requests() -> None:
    """Scans Client UI assets to enforce strict offline operation (no CDNs, no external fonts)."""
    static_dir = Path("src/hive/client/ui/static")
    assert static_dir.exists()

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


def test_client_ui_required_elements_structure() -> None:
    """Verifies all required T15 stepper and screen DOM elements exist in index.html."""
    html_path = Path("src/hive/client/ui/static/index.html")
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    required_ids = [
        # Stepper items
        "step-nav-1",
        "step-nav-2",
        "step-nav-3",
        "step-nav-4",
        "step-nav-5",
        # Screen 1: Connect
        "screen-connect",
        "input-uri",
        "input-nickname",
        "btn-connect",
        "connect-error",
        # Screen 2: Review & Approve
        "screen-review",
        "review-fingerprint",
        "review-description",
        "review-requirements",
        "review-digest",
        "review-file-list",
        "code-viewer",
        "code-display",
        "review-lint-list",
        "review-sandbox-mode",
        "check-sandbox-ack",
        "btn-approve-package",
        "btn-decline-package",
        # Screen 3: Data Folder
        "screen-data",
        "input-data-path",
        "btn-browse-folder",
        "btn-validate-data",
        "data-validation-result",
        "sample-count-badge",
        "folder-modal",
        # Screen 4: Controls
        "screen-controls",
        "toggle-only-idle",
        "slider-max-cpu",
        "slider-cpu-threshold",
        "slider-idle-minutes",
        "input-max-rounds",
        "input-max-ram",
        "toggle-use-gpu",
        "toggle-pause-battery",
        "gating-decision-note",
        "btn-pause-client",
        "btn-leave-client",
        # Screen 5: Status & Transparency
        "screen-status",
        "client-state-badge",
        "client-state-reason",
        "client-round-display",
        "live-training-progress",
        "table-client-history",
        "client-bytes-uploaded",
        "client-dp-summary",
        "panel-what-leaves",
        "preview-json-keys",
        "preview-tensor-count",
        "preview-bytes",
        "preview-sha256",
        "preview-tensors-list",
        # Footer
        "footer-node-id",
        "footer-sandbox",
        "footer-gating-note",
        "footer-bytes-uploaded",
    ]

    for elem_id in required_ids:
        assert f'id="{elem_id}"' in content, f"Required element ID '{elem_id}' is missing from Client UI index.html!"

    # Invariant checks: Reassurance statements
    assert "Only the sample count leaves this device — never file names or raw content." in content
    assert "Your raw data files are never uploaded. Only mathematical model weight deltas leave this machine." in content
