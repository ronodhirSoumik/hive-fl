"""
Unit tests for Hive CLI entrypoints (T16).
Validates commands, subcommands, help strings, keygen options, host options,
and security constraints on auto-approval.
"""

from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner

from hive.cli.main import app

runner = CliRunner(env={"COLUMNS": "120"})


def test_cli_help() -> None:
    """Verifies that `hive --help` prints usage and all primary commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Hive: Plug-and-Play Federated Learning Platform." in result.output
    assert "host" in result.output
    assert "join" in result.output
    assert "package" in result.output
    assert "keygen" in result.output
    assert "doctor" in result.output
    assert "demo" in result.output
    assert "verify-demo" in result.output


def test_cli_package_help() -> None:
    """Verifies package subcommands."""
    result = runner.invoke(app, ["package", "--help"])
    assert result.exit_code == 0
    assert "build" in result.output
    assert "inspect" in result.output
    assert "verify" in result.output


def test_cli_keygen(tmp_path: Path) -> None:
    """Verifies that `hive keygen` generates an Ed25519 keypair and prints the fingerprint."""
    key_path = tmp_path / "test_author.pem"
    result = runner.invoke(app, ["keygen", "--out", str(key_path)])
    assert result.exit_code == 0
    assert "Generated Ed25519 keypair" in result.output
    assert "Author key fingerprint: ed25519:" in result.output
    assert key_path.exists()


def test_cli_host_help() -> None:
    """Verifies `hive host --help` describes all aggregation, network, and packaging flags."""
    result = runner.invoke(app, ["host", "--help"])
    assert result.exit_code == 0
    assert "--package" in result.output
    assert "--model-path" in result.output
    assert "--preprocessor-path" in result.output
    assert "--rounds" in result.output
    assert "--min-clients" in result.output
    assert "--aggregation" in result.output
    assert "--tunnel" in result.output
    assert "--public-url" in result.output
    assert "--api-port" in result.output
    assert "--admin-port" in result.output


def test_cli_host_missing_package_error() -> None:
    """Verifies that `hive host` without package or model/preprocessor args exits with error."""
    result = runner.invoke(app, ["host"])
    assert result.exit_code != 0
    assert "Error: Must provide either --package" in result.output


def test_cli_join_help() -> None:
    """Verifies `hive join --help` displays all contributor options and security controls."""
    result = runner.invoke(app, ["join", "--help"])
    assert result.exit_code == 0
    assert "--uri" in result.output
    assert "--data-dir" in result.output
    assert "--nickname" in result.output
    assert "--ui-port" in result.output
    assert "--no-ui" in result.output
    assert "--sandbox" in result.output
    assert "--max-cpu" in result.output
    assert "--only-idle" in result.output
    assert "--auto-approve-fingerprint" in result.output
    assert "--auto-start" in result.output


def test_cli_join_auto_approve_fingerprint_mismatch() -> None:
    """Security invariant: --auto-approve-fingerprint MUST be rejected if it doesn't match the URI."""
    uri = "hive://127.0.0.1:9000/test_session?t=test_token&fp=ed25519:correct_fp"
    result = runner.invoke(
        app,
        ["join", "--uri", uri, "--auto-approve-fingerprint", "ed25519:wrong_attacker_fp", "--no-ui"],
    )
    assert result.exit_code == 1
    assert "Security error" in result.output
    assert "does not match URI fingerprint" in result.output


def test_cli_doctor() -> None:
    """Verifies that `hive doctor` runs environment checks without fatal error."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Running Hive environment diagnostics" in result.output
    assert "[Python]" in result.output
    assert "[PyTorch]" in result.output
    assert "[Docker]" in result.output
    assert "[Cloudflare]" in result.output


def test_cli_demo_and_verify_help() -> None:
    """Verifies `hive demo` and `hive verify-demo` commands."""
    res_demo = runner.invoke(app, ["demo", "--help"])
    assert res_demo.exit_code == 0
    assert "--docker" in res_demo.output
    assert "--local" in res_demo.output
    assert "--dp" in res_demo.output

    res_verify = runner.invoke(app, ["verify-demo", "--help"])
    assert res_verify.exit_code == 0