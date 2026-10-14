"""
Unit tests for Hive CLI entrypoints.
"""

from pathlib import Path
from typer.testing import CliRunner
from hive.cli.main import app

runner = CliRunner()


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