"""
Unit tests for Ed25519 package signing and fingerprint verification.
"""

from pathlib import Path
import pytest
from hive.core.errors import SignatureError
from hive.package.signing import (
    fingerprint,
    generate_keypair,
    get_public_key_bytes,
    load_private_key,
    public_key_to_hex,
    save_private_key,
    sign_digest,
    verify_signature,
)


def test_keypair_generation_and_saving(tmp_path: Path) -> None:
    priv, pub = generate_keypair()
    assert priv is not None
    assert pub is not None

    key_file = tmp_path / "test_key.pem"
    save_private_key(priv, key_file)
    assert key_file.exists()

    loaded_priv = load_private_key(key_file)
    assert get_public_key_bytes(priv) == get_public_key_bytes(loaded_priv)


def test_fingerprint_generation() -> None:
    priv, pub = generate_keypair()
    fp1 = fingerprint(pub)
    fp2 = fingerprint(priv)
    assert fp1 == fp2
    assert fp1.startswith("ed25519:")
    assert len(fp1) == 8 + 16  # "ed25519:" + 16 hex chars

    # Hex string input works
    raw_hex = public_key_to_hex(pub)
    assert fingerprint(raw_hex) == fp1


def test_sign_and_verify_signature() -> None:
    priv, pub = generate_keypair()
    digest = "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"

    sig_data = sign_digest(digest, priv)
    assert sig_data["alg"] == "ed25519"
    assert "public_key" in sig_data
    assert "signature" in sig_data

    # Valid verification
    assert verify_signature(digest, sig_data) is True

    # Valid verification with correct expected fingerprint
    fp = fingerprint(pub)
    assert verify_signature(digest, sig_data, expected_fingerprint=fp) is True


def test_signature_tampered_digest() -> None:
    priv, pub = generate_keypair()
    digest = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    sig_data = sign_digest(digest, priv)

    tampered_digest = "sha256:2222222222222222222222222222222222222222222222222222222222222222"
    with pytest.raises(SignatureError, match="tampered"):
        verify_signature(tampered_digest, sig_data)


def test_signature_wrong_fingerprint() -> None:
    priv, pub = generate_keypair()
    digest = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    sig_data = sign_digest(digest, priv)

    wrong_fp = "ed25519:deadbeefdeadbeef"
    with pytest.raises(SignatureError, match="fingerprint mismatch"):
        verify_signature(digest, sig_data, expected_fingerprint=wrong_fp)


def test_signature_tampered_signature_bytes() -> None:
    priv, pub = generate_keypair()
    digest = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    sig_data = sign_digest(digest, priv)
    # Alter last hex char
    bad_sig = sig_data["signature"][:-1] + ("0" if sig_data["signature"][-1] != "0" else "1")
    sig_data["signature"] = bad_sig

    with pytest.raises(SignatureError):
        verify_signature(digest, sig_data)