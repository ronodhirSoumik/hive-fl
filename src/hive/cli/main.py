"""
Hive CLI main entrypoint.
Command-line interface for hosting sessions, joining federated training,
building packages, key management, running diagnostics and automated demos.
"""

from __future__ import annotations

import json
from pathlib import Path
import typer

from hive.core.errors import HiveError
from hive.package.builder import build_package, build_package_from_dir
from hive.package.reader import extract_package, inspect_package, verify_package
from hive.package.signing import (
    fingerprint,
    generate_keypair,
    load_private_key,
    public_key_to_hex,
    save_private_key,
)
from hive.core.schemas import DataRequirements, Hyperparameters, Manifest

app = typer.Typer(
    name="hive",
    help="Hive: Plug-and-Play Federated Learning Platform.",
    add_completion=False,
)

package_app = typer.Typer(help="Manage and build signed .hivepkg model packages.")
app.add_typer(package_app, name="package")


@app.command()
def keygen(
    out: str = typer.Option("~/.hive/keys/author_key.pem", "--out", "-o", help="Output path for Ed25519 private key"),
) -> None:
    """Generate a new Ed25519 signing keypair for model packaging."""
    out_p = Path(out).expanduser()
    priv, pub = generate_keypair()
    save_private_key(priv, out_p)
    fp = fingerprint(pub)
    typer.echo(f"Generated Ed25519 keypair.")
    typer.echo(f"Private key saved to: {out_p}")
    typer.echo(f"Author key fingerprint: {fp}")


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
    key: str | None = typer.Option(None, "--key", "-k", help="Private key file for signing (auto-generated if omitted)"),
    out: str = typer.Option("package.hivepkg", "--out", "-o", help="Output .hivepkg file"),
) -> None:
    """Build and sign a .hivepkg package archive."""
    try:
        # Resolve private key
        if key:
            priv_key = load_private_key(key)
        else:
            default_key = Path("~/.hive/keys/default.pem").expanduser()
            if default_key.exists():
                priv_key = load_private_key(default_key)
            else:
                priv_key, pub = generate_keypair()
                save_private_key(priv_key, default_key)
                typer.echo(f"Generated default signing key at: {default_key}")

        if src:
            pkg_path = build_package_from_dir(src, priv_key, out_path=out)
        elif model_path and preprocessor_path:
            m_code = Path(model_path).read_text(encoding="utf-8")
            p_code = Path(preprocessor_path).read_text(encoding="utf-8")
            manifest = Manifest(
                package_name=name,
                version=version,
                description=f"Hive model package {name}",
                model_entrypoint="model.py:Model",
                preprocessor_entrypoint="preprocess.py:Preprocessor",
                loss=loss,
                metrics=metrics,
                hyperparameters=Hyperparameters(
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    learning_rate=lr,
                ),
                data_requirements=DataRequirements(description="Dataset"),
            )
            pkg_path = build_package(
                manifest=manifest,
                model_code=m_code,
                preprocessor_code=p_code,
                requirements_text="torch\nnumpy",
                private_key=priv_key,
                out_path=out,
            )
        else:
            typer.echo("Error: Must provide either --src directory or both --model-path and --preprocessor-path.")
            raise typer.Exit(code=1)

        typer.echo(f"Successfully built signed package: {pkg_path}")
        fp = fingerprint(priv_key)
        typer.echo(f"Signed with author key fingerprint: {fp}")

    except HiveError as e:
        typer.echo(f"Package build failed: {e}")
        raise typer.Exit(code=1)


@package_app.command("inspect")
def package_inspect(
    package: str = typer.Argument(..., help="Path to .hivepkg file"),
    as_json: bool = typer.Option(False, "--json", help="Output inspection report as JSON"),
) -> None:
    """Inspect metadata, manifest and signatures of a .hivepkg archive."""
    try:
        info = inspect_package(package)
        if as_json:
            typer.echo(json.dumps(info, indent=2))
        else:
            typer.echo(f"Package: {info['package_name']} v{info['version']}")
            typer.echo(f"Description: {info['description']}")
            typer.echo(f"Author Fingerprint: {info['author_fingerprint']}")
            typer.echo(f"Content Digest: {info['content_digest']}")
            typer.echo(f"Files: {', '.join(info['files'])}")
    except Exception as e:
        typer.echo(f"Inspection failed: {e}")
        raise typer.Exit(code=1)


@package_app.command("verify")
def package_verify(
    package: str = typer.Argument(..., help="Path to .hivepkg file"),
    fingerprint: str | None = typer.Option(None, "--fingerprint", "-fp", help="Expected author key fingerprint"),
) -> None:
    """Verify cryptographic signature and digest integrity of a .hivepkg archive."""
    try:
        verify_package(package, expected_fingerprint=fingerprint)
        typer.echo(f"Verification PASS: Package '{package}' has a valid signature and unaltered content digest.")
    except Exception as e:
        typer.echo(f"Verification FAIL: {e}")
        raise typer.Exit(code=1)


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