"""
Hive CLI main entrypoint.
Command-line interface for hosting sessions, joining federated training,
building packages, running diagnostics and automated demos.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="hive",
    help="Hive: Plug-and-Play Federated Learning Platform.",
    add_completion=False,
)

package_app = typer.Typer(help="Manage and build signed .hivepkg model packages.")
app.add_typer(package_app, name="package")


@app.command()
def host(
    package: str = typer.Option(..., "--package", "-p", help="Path to .hivepkg model package"),
    rounds: int = typer.Option(10, "--rounds", "-r", help="Total number of federated rounds"),
    min_clients: int = typer.Option(2, "--min-clients", "-m", help="Minimum clients required per round"),
    max_clients: int = typer.Option(50, "--max-clients", help="Maximum clients per round"),
    aggregation: str = typer.Option("fedavg", "--aggregation", help="Aggregation strategy"),
    round_timeout: int = typer.Option(600, "--round-timeout", help="Round timeout in seconds"),
    server_clip: float | None = typer.Option(None, "--server-clip", help="Server L2 clipping norm"),
    tunnel: str = typer.Option("none", "--tunnel", help="Tunnel mode: none, cloudflared, manual"),
    public_url: str | None = typer.Option(None, "--public-url", help="Manual public URL for clients"),
    api_port: int = typer.Option(9000, "--api-port", help="Public FL API port"),
    admin_port: int = typer.Option(8000, "--admin-port", help="Admin dashboard port"),
    require_approval: bool = typer.Option(False, "--require-approval", help="Require manual node approval"),
    workdir: str = typer.Option(".hive/host", "--workdir", help="Directory for database and checkpoints"),
    resume: bool = typer.Option(False, "--resume", help="Resume previous session if present"),
) -> None:
    """Start a federated learning Host server."""
    typer.echo(f"Starting Hive Host for package {package} on port {api_port}...")


@app.command()
def join(
    uri: str = typer.Option(..., "--uri", "-u", help="Hive session join URI (hive://...)"),
    data_dir: str | None = typer.Option(None, "--data-dir", "-d", help="Path to local private data directory"),
    nickname: str = typer.Option("Node", "--nickname", "-n", help="Contributor display nickname"),
    ui_port: int = typer.Option(8765, "--ui-port", help="Local contributor Web UI port"),
    no_ui: bool = typer.Option(False, "--no-ui", help="Run without opening local Web UI"),
    sandbox: str = typer.Option("auto", "--sandbox", help="Sandbox isolation mode: auto, docker, process"),
    max_cpu: int = typer.Option(50, "--max-cpu", help="Maximum CPU limit percent"),
    only_idle: bool = typer.Option(True, "--only-idle/--no-only-idle", help="Train only when user is idle"),
    rounds: int = typer.Option(0, "--rounds", help="Train for N rounds before leaving (0=unlimited)"),
    use_gpu: bool = typer.Option(False, "--use-gpu", help="Enable GPU training if available"),
    auto_approve_fingerprint: str | None = typer.Option(None, "--auto-approve-fingerprint", help="Auto-consent if host key matches"),
    auto_start: bool = typer.Option(False, "--auto-start", help="Automatically start training after validation"),
) -> None:
    """Join an active Hive federated learning session as a compute contributor."""
    typer.echo(f"Connecting to Hive session at {uri}...")


@package_app.command("build")
def package_build(
    src: str | None = typer.Option(None, "--src", help="Source directory containing model.py and preprocess.py"),
    model_path: str | None = typer.Option(None, "--model-path", help="Path to model.py"),
    preprocessor_path: str | None = typer.Option(None, "--preprocessor-path", help="Path to preprocess.py"),
    name: str = typer.Option("model-package", "--name", help="Package name"),
    version: str = typer.Option("0.1.0", "--version", help="Package version"),
    loss: str = typer.Option("cross_entropy", "--loss", help="Loss function"),
    metrics: str = typer.Option("accuracy", "--metrics", help="Evaluation metric"),
    local_epochs: int = typer.Option(2, "--local-epochs", help="Default local epochs"),
    batch_size: int = typer.Option(32, "--batch-size", help="Default batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Default learning rate"),
    key: str | None = typer.Option(None, "--key", help="Private key file for signing"),
    out: str = typer.Option("package.hivepkg", "--out", "-o", help="Output .hivepkg file"),
) -> None:
    """Build and sign a .hivepkg package archive."""
    typer.echo(f"Building package {name} into {out}...")


@package_app.command("inspect")
def package_inspect(
    package: str = typer.Argument(..., help="Path to .hivepkg file"),
) -> None:
    """Inspect metadata, manifest and signatures of a .hivepkg archive."""
    typer.echo(f"Inspecting package: {package}")


@package_app.command("verify")
def package_verify(
    package: str = typer.Argument(..., help="Path to .hivepkg file"),
    fingerprint: str | None = typer.Option(None, "--fingerprint", help="Expected author key fingerprint"),
) -> None:
    """Verify cryptographic signature and digest integrity of a .hivepkg archive."""
    typer.echo(f"Verifying package: {package}")


@app.command()
def doctor() -> None:
    """Run diagnostics on environment, Docker, Python version and network."""
    typer.echo("Running Hive environment diagnostics...")


@app.command()
def demo(
    docker: bool = typer.Option(False, "--docker", help="Run multi-container Docker compose demo"),
    local: bool = typer.Option(False, "--local", help="Run local multi-process demo"),
    dp: bool = typer.Option(False, "--dp", help="Enable differential privacy in demo client"),
    manual: bool = typer.Option(False, "--manual", help="Do not auto-start; wait for user clicks in UI"),
) -> None:
    """Run the BirdSound-Lite federated learning demo."""
    typer.echo("Starting Hive BirdSound-Lite demo...")


@app.command("verify-demo")
def verify_demo() -> None:
    """Execute automated end-to-end verification of the federated learning demo."""
    typer.echo("Running demo verification suite...")


if __name__ == "__main__":
    app()