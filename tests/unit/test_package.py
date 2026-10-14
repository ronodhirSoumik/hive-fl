"""
Unit tests for .hivepkg builder, safe reader, extractor, and verification.
Includes security tests for tar-slip, symlinks, tamper detection, and dependency filtering.
"""

import io
import json
from pathlib import Path
import tarfile
import pytest

from hive.core.errors import PackageError, SignatureError
from hive.core.schemas import DataRequirements, Hyperparameters, Manifest
from hive.package.builder import build_package, compute_package_digest
from hive.package.reader import extract_package, inspect_package, verify_package
from hive.package.signing import fingerprint, generate_keypair


@pytest.fixture
def sample_manifest() -> Manifest:
    return Manifest(
        package_name="test-classifier",
        version="1.0.0",
        description="Test FL Package",
        model_entrypoint="model.py:TestModel",
        preprocessor_entrypoint="preprocess.py:TestPreprocessor",
        loss="cross_entropy",
        metrics="accuracy",
        hyperparameters=Hyperparameters(local_epochs=2, batch_size=16, learning_rate=0.01),
        data_requirements=DataRequirements(description="Synthetic CSV", min_samples=10),
    )


@pytest.fixture
def valid_package(tmp_path: Path, sample_manifest: Manifest) -> tuple[Path, any]:
    priv, pub = generate_keypair()
    model_code = """
import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)
"""
    prep_code = """
from hive.core.contracts import BaseDataPreprocessor, ValidationResult

class TestPreprocessor(BaseDataPreprocessor):
    def validate_local_directory(self, data_dir):
        return ValidationResult(ok=True, num_samples=50)
    def build_dataset(self, data_dir, split, seed=42):
        return []
"""
    reqs = "torch\nnumpy"
    out_pkg = tmp_path / "valid.hivepkg"
    build_package(
        manifest=sample_manifest,
        model_code=model_code,
        preprocessor_code=prep_code,
        requirements_text=reqs,
        private_key=priv,
        readme_text="Test readme",
        out_path=out_pkg,
    )
    return out_pkg, priv


def test_package_build_and_extract_roundtrip(valid_package: tuple[Path, any], tmp_path: Path) -> None:
    pkg_path, priv_key = valid_package
    dest_dir = tmp_path / "extracted"

    # Verify and extract
    author_fp = fingerprint(priv_key)
    manifest = extract_package(pkg_path, dest_dir, expected_fingerprint=author_fp)

    assert manifest.package_name == "test-classifier"
    assert (dest_dir / "model.py").exists()
    assert (dest_dir / "preprocess.py").exists()
    assert (dest_dir / "manifest.json").exists()
    assert (dest_dir / "SIGNATURE").exists()
    assert (dest_dir / "README.md").read_text(encoding="utf-8") == "Test readme"


def test_package_inspect(valid_package: tuple[Path, any]) -> None:
    pkg_path, priv_key = valid_package
    info = inspect_package(pkg_path)

    assert info["package_name"] == "test-classifier"
    assert info["version"] == "1.0.0"
    assert info["author_fingerprint"] == fingerprint(priv_key)
    assert "model.py" in info["files"]
    assert "preprocess.py" in info["files"]
    assert info["content_digest"].startswith("sha256:")


def test_package_tampered_file(valid_package: tuple[Path, any], tmp_path: Path) -> None:
    pkg_path, _ = valid_package

    # Unpack, modify model.py, repack without re-signing
    repack_dir = tmp_path / "tamper_temp"
    repack_dir.mkdir()
    with tarfile.open(pkg_path, "r:gz") as tar:
        tar.extractall(repack_dir)

    # Tamper with model.py
    (repack_dir / "model.py").write_text("# MALICIOUS INJECTION\n", encoding="utf-8")

    tampered_pkg = tmp_path / "tampered.hivepkg"
    with tarfile.open(tampered_pkg, "w:gz") as tar:
        for f in repack_dir.iterdir():
            tar.add(f, arcname=f.name)

    # Verification must fail
    with pytest.raises(PackageError, match="digest mismatch"):
        verify_package(tampered_pkg)


def test_package_wrong_fingerprint(valid_package: tuple[Path, any]) -> None:
    pkg_path, _ = valid_package
    wrong_fp = "ed25519:ffffffffffffffff"
    with pytest.raises(SignatureError, match="fingerprint mismatch"):
        verify_package(pkg_path, expected_fingerprint=wrong_fp)


def test_tar_slip_prevention(tmp_path: Path) -> None:
    """Security test: Archives with path traversal must be rejected immediately."""
    evil_pkg = tmp_path / "evil_tarslip.hivepkg"
    with tarfile.open(evil_pkg, "w:gz") as tar:
        ti = tarfile.TarInfo(name="../evil.py")
        content = b"print('pwned')"
        ti.size = len(content)
        tar.addfile(ti, io.BytesIO(content))

    with pytest.raises(PackageError, match="Path traversal"):
        inspect_package(evil_pkg)


def test_absolute_path_prevention(tmp_path: Path) -> None:
    """Security test: Archives with absolute paths must be rejected."""
    evil_pkg = tmp_path / "evil_abs.hivepkg"
    with tarfile.open(evil_pkg, "w:gz") as tar:
        ti = tarfile.TarInfo(name="/etc/passwd")
        content = b"root:x:0:0"
        ti.size = len(content)
        tar.addfile(ti, io.BytesIO(content))

    with pytest.raises(PackageError, match="Absolute path"):
        inspect_package(evil_pkg)


def test_symlink_prevention(tmp_path: Path) -> None:
    """Security test: Archives with symlinks must be rejected."""
    evil_pkg = tmp_path / "evil_symlink.hivepkg"
    with tarfile.open(evil_pkg, "w:gz") as tar:
        ti = tarfile.TarInfo(name="link.txt")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/shadow"
        tar.addfile(ti)

    with pytest.raises(PackageError, match="Symlink"):
        inspect_package(evil_pkg)


def test_disallowed_dependency_rejection(sample_manifest: Manifest, tmp_path: Path) -> None:
    priv, _ = generate_keypair()
    model_code = "class TestModel:\n    pass\n"
    prep_code = "class TestPreprocessor:\n    pass\n"
    disallowed_reqs = "torch\nrequests>=2.0\n"

    with pytest.raises(PackageError, match="Disallowed dependency 'requests'"):
        build_package(
            manifest=sample_manifest,
            model_code=model_code,
            preprocessor_code=prep_code,
            requirements_text=disallowed_reqs,
            private_key=priv,
            out_path=tmp_path / "bad_req.hivepkg",
        )


def test_missing_entrypoint_rejection(sample_manifest: Manifest, tmp_path: Path) -> None:
    priv, _ = generate_keypair()
    model_code = "class NonExistentModel:\n    pass\n"
    prep_code = "class TestPreprocessor:\n    pass\n"

    with pytest.raises(PackageError, match="Entrypoint .* not found"):
        build_package(
            manifest=sample_manifest,
            model_code=model_code,
            preprocessor_code=prep_code,
            requirements_text="torch",
            private_key=priv,
            out_path=tmp_path / "bad_entry.hivepkg",
        )