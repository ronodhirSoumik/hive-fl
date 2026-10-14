"""
Builder for .hivepkg model packages.
Packages model code, preprocessor, requirements, and manifest into a signed,
deterministic gzip-tar archive.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Sequence

from cryptography.hazmat.primitives.asymmetric import ed25519

from hive.core.errors import PackageError
from hive.core.schemas import Manifest
from hive.package.lint import lint_directory, lint_source
from hive.package.signing import sign_digest

# Whitelist of allowed external dependencies in requirements.txt
ALLOWED_DEPENDENCIES = {
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "scipy",
    "scikit-learn",
    "safetensors",
    "pydantic",
}


def compute_file_sha256(path: Path | str) -> str:
    """Calculates hex SHA-256 hash of a file on disk."""
    p = Path(path)
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Calculates hex SHA-256 hash of in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_package_digest(files_dict: dict[str, bytes]) -> str:
    """
    Computes canonical content digest over a mapping of {rel_path: bytes}.
    Ignores SIGNATURE file.
    Blanks 'content_digest' in manifest.json before hashing.
    Format: SHA-256 over sorted lines of '{rel_path}:{sha256_hex}\n'.
    """
    lines: list[str] = []
    for rel_path in sorted(files_dict.keys()):
        if rel_path == "SIGNATURE":
            continue
        content = files_dict[rel_path]
        if rel_path == "manifest.json":
            # Blank content_digest in manifest before hashing
            try:
                manifest_data = json.loads(content.decode("utf-8"))
                manifest_data["content_digest"] = ""
                content = json.dumps(manifest_data, sort_keys=True, indent=2).encode("utf-8")
            except Exception:
                pass
        sha = compute_bytes_sha256(content)
        lines.append(f"{rel_path}:{sha}\n")

    combined = "".join(lines).encode("utf-8")
    return f"sha256:{hashlib.sha256(combined).hexdigest()}"


def validate_entrypoint(code: str, entrypoint: str, filename: str) -> None:
    """
    Uses AST parsing to confirm the target class or callable exists in the file.
    Does not execute any user code.
    """
    if ":" in entrypoint:
        file_part, name_part = entrypoint.split(":", 1)
    else:
        name_part = entrypoint

    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError as e:
        raise PackageError(f"Syntax error in {filename}: {e}", code="ERR_PACKAGE_SYNTAX")

    found = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name_part:
            found = True
            break
        elif isinstance(node, ast.FunctionDef) and node.name == name_part:
            found = True
            break

    if not found:
        raise PackageError(
            f"Entrypoint '{entrypoint}' not found in {filename}. Expected class or function '{name_part}'.",
            code="ERR_PACKAGE_ENTRYPOINT_MISSING",
        )


def validate_requirements(requirements_text: str) -> None:
    """
    Validates requirements.txt against the allowed library whitelist.
    Raises PackageError if an unauthorized library is requested.
    """
    for line in requirements_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip version specifiers like >=, ==, <, etc.
        pkg_name = line
        for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
            if op in pkg_name:
                pkg_name = pkg_name.split(op)[0]
                break
        pkg_name = pkg_name.strip().lower()
        if pkg_name not in ALLOWED_DEPENDENCIES:
            raise PackageError(
                f"Disallowed dependency '{pkg_name}' in requirements.txt. Allowed dependencies: {sorted(ALLOWED_DEPENDENCIES)}",
                code="ERR_PACKAGE_DISALLOWED_DEPENDENCY",
            )


def build_package(
    manifest: Manifest,
    model_code: str,
    preprocessor_code: str,
    requirements_text: str,
    private_key: ed25519.Ed25519PrivateKey,
    readme_text: str | None = None,
    out_path: Path | str = "package.hivepkg",
) -> Path:
    """
    Builds and signs a .hivepkg archive from in-memory strings.
    Validates AST entrypoints, requirements, lints code, computes digest,
    signs with Ed25519, and saves deterministic gzip-tar.
    """
    # 1. Validate entrypoints via AST
    validate_entrypoint(model_code, manifest.model_entrypoint, "model.py")
    validate_entrypoint(preprocessor_code, manifest.preprocessor_entrypoint, "preprocess.py")

    # 2. Validate requirements against allowlist
    validate_requirements(requirements_text)

    # 3. Lint source files
    findings = lint_source(model_code, "model.py") + lint_source(preprocessor_code, "preprocess.py")
    # Syntax errors in lint raise PackageError
    for f in findings:
        if f.severity == "ERROR":
            raise PackageError(f"Lint error: {f}", code="ERR_PACKAGE_LINT_ERROR")

    # 4. Assemble package files dictionary
    files_dict: dict[str, bytes] = {
        "model.py": model_code.encode("utf-8"),
        "preprocess.py": preprocessor_code.encode("utf-8"),
        "requirements.txt": requirements_text.encode("utf-8"),
    }
    if readme_text:
        files_dict["README.md"] = readme_text.encode("utf-8")

    # Add manifest with blank digest
    manifest_dict = manifest.model_dump()
    manifest_dict["content_digest"] = ""
    files_dict["manifest.json"] = json.dumps(manifest_dict, sort_keys=True, indent=2).encode("utf-8")

    # 5. Compute canonical content digest
    digest = compute_package_digest(files_dict)

    # 6. Update manifest with computed digest
    manifest_dict["content_digest"] = digest
    files_dict["manifest.json"] = json.dumps(manifest_dict, sort_keys=True, indent=2).encode("utf-8")

    # 7. Sign digest
    sig_dict = sign_digest(digest, private_key)
    files_dict["SIGNATURE"] = json.dumps(sig_dict, sort_keys=True, indent=2).encode("utf-8")

    # 8. Create deterministic gzip tar archive
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out_p, "w:gz") as tar:
        for name in sorted(files_dict.keys()):
            data = files_dict[name]
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            ti.mtime = 0
            ti.uid = 0
            ti.gid = 0
            ti.uname = ""
            ti.gname = ""
            ti.mode = 0o644
            tar.addfile(ti, io.BytesIO(data))

    return out_p


def build_package_from_dir(
    source_dir: Path | str,
    private_key: ed25519.Ed25519PrivateKey,
    out_path: Path | str = "package.hivepkg",
) -> Path:
    """
    Builds a .hivepkg archive from a source directory containing manifest.json,
    model.py, preprocess.py, and requirements.txt.
    """
    src = Path(source_dir)
    manifest_file = src / "manifest.json"
    model_file = src / "model.py"
    prep_file = src / "preprocess.py"
    req_file = src / "requirements.txt"

    if not manifest_file.exists():
        raise PackageError(f"Missing manifest.json in {source_dir}", code="ERR_PACKAGE_MISSING_MANIFEST")
    if not model_file.exists():
        raise PackageError(f"Missing model.py in {source_dir}", code="ERR_PACKAGE_MISSING_MODEL")
    if not prep_file.exists():
        raise PackageError(f"Missing preprocess.py in {source_dir}", code="ERR_PACKAGE_MISSING_PREPROCESS")

    manifest = Manifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
    model_code = model_file.read_text(encoding="utf-8")
    preprocessor_code = prep_file.read_text(encoding="utf-8")
    requirements_text = req_file.read_text(encoding="utf-8") if req_file.exists() else ""

    readme_file = src / "README.md"
    readme_text = readme_file.read_text(encoding="utf-8") if readme_file.exists() else None

    return build_package(
        manifest=manifest,
        model_code=model_code,
        preprocessor_code=preprocessor_code,
        requirements_text=requirements_text,
        private_key=private_key,
        readme_text=readme_text,
        out_path=out_path,
    )