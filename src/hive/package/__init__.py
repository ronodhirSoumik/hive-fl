"""
Hive model packaging and cryptographic signing module.
"""

from hive.package.builder import (
    ALLOWED_DEPENDENCIES,
    build_package,
    build_package_from_dir,
    compute_package_digest,
)
from hive.package.lint import LintFinding, lint_directory, lint_source
from hive.package.reader import extract_package, inspect_package, verify_package
from hive.package.signing import (
    fingerprint,
    generate_keypair,
    load_private_key,
    public_key_to_hex,
    save_private_key,
    sign_digest,
    verify_signature,
)

__all__ = [
    "ALLOWED_DEPENDENCIES",
    "build_package",
    "build_package_from_dir",
    "compute_package_digest",
    "extract_package",
    "inspect_package",
    "verify_package",
    "generate_keypair",
    "load_private_key",
    "save_private_key",
    "public_key_to_hex",
    "fingerprint",
    "sign_digest",
    "verify_signature",
    "lint_source",
    "lint_directory",
    "LintFinding",
]