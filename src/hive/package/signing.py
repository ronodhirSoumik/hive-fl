"""
Cryptographic signing and verification for Hive model packages.
Uses Ed25519 digital signatures and canonical fingerprint calculation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from hive.core.errors import SignatureError


def generate_keypair() -> tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
    """Generates a new Ed25519 private/public keypair."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def save_private_key(key: ed25519.Ed25519PrivateKey, path: Path | str) -> None:
    """
    Saves an Ed25519 private key to disk in PEM format.
    Sets strict file permissions (0600 on POSIX).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    p.write_bytes(pem)
    # Set read/write only for owner on POSIX
    if os.name == "posix":
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass


def load_private_key(path: Path | str) -> ed25519.Ed25519PrivateKey:
    """Loads an Ed25519 private key from a PEM file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Private key file not found: {p}")
    pem_data = p.read_bytes()
    key = serialization.load_pem_private_key(pem_data, password=None)
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise SignatureError("Key is not an Ed25519 private key", code="ERR_INVALID_KEY_TYPE")
    return key


def get_public_key_bytes(key: ed25519.Ed25519PrivateKey | ed25519.Ed25519PublicKey) -> bytes:
    """Extracts raw 32-byte public key representation."""
    pub = key.public_key() if isinstance(key, ed25519.Ed25519PrivateKey) else key
    return pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def public_key_to_hex(key: ed25519.Ed25519PrivateKey | ed25519.Ed25519PublicKey) -> str:
    """Returns hexadecimal string of raw 32-byte public key."""
    return get_public_key_bytes(key).hex()


def fingerprint(key: ed25519.Ed25519PrivateKey | ed25519.Ed25519PublicKey | bytes | str) -> str:
    """
    Computes a canonical fingerprint for an Ed25519 public key.
    Format: ed25519:<16-char-hex-prefix-of-sha256>
    """
    if isinstance(key, (ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey)):
        raw_bytes = get_public_key_bytes(key)
    elif isinstance(key, str):
        cleaned = key.strip()
        if cleaned.startswith("ed25519:"):
            # Already a fingerprint
            return cleaned
        raw_bytes = bytes.fromhex(cleaned)
    elif isinstance(key, bytes):
        raw_bytes = key
    else:
        raise TypeError(f"Unsupported key representation for fingerprint: {type(key)}")

    h = hashlib.sha256(raw_bytes).hexdigest()[:16]
    return f"ed25519:{h}"


def sign_digest(digest: str, private_key: ed25519.Ed25519PrivateKey) -> dict[str, str]:
    """
    Signs a content digest string using Ed25519.
    Returns signature dictionary suitable for SIGNATURE file.
    """
    digest_bytes = digest.encode("utf-8")
    signature = private_key.sign(digest_bytes)
    pub_hex = public_key_to_hex(private_key)
    return {
        "alg": "ed25519",
        "public_key": pub_hex,
        "signature": signature.hex(),
    }


def verify_signature(
    digest: str,
    signature_data: dict[str, Any] | str,
    expected_fingerprint: str | None = None,
) -> bool:
    """
    Verifies an Ed25519 digital signature against a digest string.
    Optionally verifies that the signer matches an expected fingerprint.
    Raises SignatureError on verification failure.
    """
    if isinstance(signature_data, str):
        try:
            signature_dict = json.loads(signature_data)
        except Exception as e:
            raise SignatureError(f"Malformed SIGNATURE JSON: {e}", code="ERR_SIGNATURE_INVALID")
    else:
        signature_dict = signature_data

    alg = signature_dict.get("alg")
    if alg != "ed25519":
        raise SignatureError(f"Unsupported signature algorithm '{alg}'. Only 'ed25519' is accepted.", code="ERR_SIGNATURE_UNSUPPORTED_ALG")

    pub_hex = signature_dict.get("public_key")
    sig_hex = signature_dict.get("signature")

    if not pub_hex or not sig_hex:
        raise SignatureError("SIGNATURE file missing public_key or signature field", code="ERR_SIGNATURE_INVALID")

    try:
        pub_bytes = bytes.fromhex(pub_hex)
        sig_bytes = bytes.fromhex(sig_hex)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
    except Exception as e:
        raise SignatureError(f"Failed to parse public key or signature hex: {e}", code="ERR_SIGNATURE_INVALID")

    # Verify fingerprint pinning if provided
    if expected_fingerprint is not None:
        computed_fp = fingerprint(pub_bytes)
        expected_fp = expected_fingerprint.strip()
        if not expected_fp.startswith("ed25519:"):
            expected_fp = f"ed25519:{expected_fp}"
        if computed_fp != expected_fp:
            raise SignatureError(
                f"Signer fingerprint mismatch: expected '{expected_fp}', got '{computed_fp}'",
                code="ERR_FINGERPRINT_MISMATCH",
            )

    # Verify signature
    digest_bytes = digest.encode("utf-8")
    try:
        public_key.verify(sig_bytes, digest_bytes)
    except InvalidSignature:
        raise SignatureError("Ed25519 digital signature verification failed. Digest has been tampered with.", code="ERR_SIGNATURE_MISMATCH")

    return True