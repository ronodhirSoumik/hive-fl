"""
Safe reader, verifier, and extractor for .hivepkg package archives.
Defends against path traversal (tar-slip), symlinks, zip bombs, and tampering.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

from hive.core.errors import PackageError, SignatureError
from hive.core.schemas import Manifest
from hive.package.builder import compute_package_digest
from hive.package.signing import fingerprint, verify_signature

MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_FILES = 200


def _validate_tar_member(member: tarfile.TarInfo, seen_names: set[str], current_total_bytes: int) -> int:
    """
    Validates a tar member against path traversal, symlink, and resource exhaustion attacks.
    Returns the new cumulative total bytes.
    """
    name = member.name

    # Reject duplicate names
    if name in seen_names:
        raise PackageError(f"Duplicate file entry '{name}' in package archive", code="ERR_PACKAGE_DUPLICATE_ENTRY")
    seen_names.add(name)

    # Reject absolute paths (POSIX / or Windows C:\)
    if name.startswith("/") or name.startswith("\\") or (len(name) > 1 and name[1] == ":"):
        raise PackageError(f"Absolute path '{name}' detected in archive", code="ERR_PACKAGE_TAR_SLIP")

    # Reject path traversal components
    parts = Path(name).parts
    if ".." in parts:
        raise PackageError(f"Path traversal '..' detected in archive entry '{name}'", code="ERR_PACKAGE_TAR_SLIP")

    # Reject non-regular files (symlinks, hardlinks, character/block devices, fifos)
    if member.issym() or member.islnk():
        raise PackageError(f"Symlink or hardlink entry '{name}' is strictly forbidden", code="ERR_PACKAGE_SYMLINK_FORBIDDEN")
    if member.ischr() or member.isblk() or member.isfifo():
        raise PackageError(f"Device or FIFO file entry '{name}' is strictly forbidden", code="ERR_PACKAGE_DEVICE_FORBIDDEN")

    # Enforce size limits
    new_total = current_total_bytes + member.size
    if new_total > MAX_UNCOMPRESSED_BYTES:
        raise PackageError(
            f"Archive exceeds uncompressed size limit of {MAX_UNCOMPRESSED_BYTES} bytes",
            code="ERR_PACKAGE_OVERSIZED",
        )

    return new_total


def inspect_package(pkg_path: Path | str) -> dict[str, Any]:
    """
    Inspects package metadata, manifest, author fingerprint, and file list in-memory
    without extracting files to disk.
    """
    p = Path(pkg_path)
    if not p.exists():
        raise FileNotFoundError(f"Package file not found: {p}")

    seen_names: set[str] = set()
    total_bytes = 0
    files_map: dict[str, bytes] = {}

    with tarfile.open(p, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) > MAX_FILES:
            raise PackageError(f"Package contains {len(members)} entries, exceeding limit of {MAX_FILES}", code="ERR_PACKAGE_TOO_MANY_FILES")

        for m in members:
            total_bytes = _validate_tar_member(m, seen_names, total_bytes)
            if m.isfile():
                f = tar.extractfile(m)
                if f is not None:
                    files_map[m.name] = f.read()

    if "manifest.json" not in files_map:
        raise PackageError("Package archive missing required manifest.json", code="ERR_PACKAGE_MISSING_MANIFEST")
    if "SIGNATURE" not in files_map:
        raise PackageError("Package archive missing required SIGNATURE file", code="ERR_PACKAGE_MISSING_SIGNATURE")

    manifest = Manifest.model_validate_json(files_map["manifest.json"].decode("utf-8"))
    sig_data = json.loads(files_map["SIGNATURE"].decode("utf-8"))
    pub_key_hex = sig_data.get("public_key", "")
    author_fp = fingerprint(bytes.fromhex(pub_key_hex)) if pub_key_hex else "unknown"

    return {
        "package_name": manifest.package_name,
        "version": manifest.version,
        "description": manifest.description,
        "content_digest": manifest.content_digest,
        "author_fingerprint": author_fp,
        "files": sorted(files_map.keys()),
        "manifest": manifest.model_dump(),
        "signature": sig_data,
    }


def verify_package(pkg_path: Path | str, expected_fingerprint: str | None = None) -> bool:
    """
    Performs full cryptographic verification of a .hivepkg archive:
    1. Validates safe tar structure and limits.
    2. Recomputes canonical content digest over members.
    3. Verifies Ed25519 digital signature.
    4. Confirms author fingerprint matches expected_fingerprint if specified.
    """
    p = Path(pkg_path)
    if not p.exists():
        raise FileNotFoundError(f"Package file not found: {p}")

    seen_names: set[str] = set()
    total_bytes = 0
    files_map: dict[str, bytes] = {}

    with tarfile.open(p, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) > MAX_FILES:
            raise PackageError(f"Package exceeds file count limit: {len(members)} > {MAX_FILES}", code="ERR_PACKAGE_TOO_MANY_FILES")

        for m in members:
            total_bytes = _validate_tar_member(m, seen_names, total_bytes)
            if m.isfile():
                f = tar.extractfile(m)
                if f is not None:
                    files_map[m.name] = f.read()

    if "manifest.json" not in files_map:
        raise PackageError("Package archive missing manifest.json", code="ERR_PACKAGE_MISSING_MANIFEST")
    if "SIGNATURE" not in files_map:
        raise PackageError("Package archive missing SIGNATURE", code="ERR_PACKAGE_MISSING_SIGNATURE")

    # 1. Parse manifest and stated digest
    manifest = Manifest.model_validate_json(files_map["manifest.json"].decode("utf-8"))
    stated_digest = manifest.content_digest

    # 2. Recompute content digest
    computed_digest = compute_package_digest(files_map)
    if stated_digest != computed_digest:
        raise PackageError(
            f"Content digest mismatch. Package files have been altered or corrupted! Stated: {stated_digest}, Computed: {computed_digest}",
            code="ERR_PACKAGE_DIGEST_MISMATCH",
        )

    # 3. Verify signature
    sig_data = json.loads(files_map["SIGNATURE"].decode("utf-8"))
    verify_signature(computed_digest, sig_data, expected_fingerprint=expected_fingerprint)
    return True


def extract_package(
    pkg_path: Path | str,
    dest_dir: Path | str,
    expected_fingerprint: str | None = None,
) -> Manifest:
    """
    Safely verifies and extracts a .hivepkg archive into dest_dir.
    Raises PackageError or SignatureError if validation fails.
    Returns the parsed Manifest.
    """
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # Verify digest and signature first
    verify_package(pkg_path, expected_fingerprint=expected_fingerprint)

    seen_names: set[str] = set()
    total_bytes = 0

    with tarfile.open(pkg_path, "r:gz") as tar:
        for m in tar.getmembers():
            total_bytes = _validate_tar_member(m, seen_names, total_bytes)
            target_path = (dest / m.name).resolve()
            # Ensure target path does not escape dest_dir
            if not str(target_path).startswith(str(dest)):
                raise PackageError(f"Target path '{target_path}' escapes extraction directory", code="ERR_PACKAGE_TAR_SLIP")

            if m.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                f_in = tar.extractfile(m)
                if f_in is not None:
                    target_path.write_bytes(f_in.read())

    manifest_file = dest / "manifest.json"
    if not manifest_file.exists():
        raise PackageError("Extracted package missing manifest.json", code="ERR_PACKAGE_MISSING_MANIFEST")

    return Manifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))