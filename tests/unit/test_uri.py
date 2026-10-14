"""
Unit tests for Hive URI building and parsing.
Format: hive://<authority>/<session_id>?t=<join_token>&fp=<fingerprint>&tls=<0|1>
"""

import pytest
from hive.core.errors import ProtocolError
from hive.core.uri import HiveURI, build_hive_uri, parse_hive_uri


def test_uri_build_and_parse_roundtrip() -> None:
    authority = "127.0.0.1:9000"
    session_id = "sess_demo_123"
    token = "token_xyz"
    fingerprint = "ed25519:abcdef123456"

    # With TLS = True
    uri_str = build_hive_uri(authority, session_id, token, fingerprint, tls=True)
    assert uri_str.startswith("hive://127.0.0.1:9000/sess_demo_123?")
    assert "t=token_xyz" in uri_str
    assert "fp=ed25519%3Aabcdef123456" in uri_str or "fp=ed25519:abcdef123456" in uri_str
    assert "tls=1" in uri_str

    parsed = parse_hive_uri(uri_str)
    assert parsed.authority == authority
    assert parsed.session_id == session_id
    assert parsed.join_token == token
    assert parsed.fingerprint == fingerprint
    assert parsed.tls is True
    assert parsed.base_http_url == "https://127.0.0.1:9000"

    # Re-serialization matches
    assert parse_hive_uri(str(parsed)).authority == authority


def test_uri_plain_http() -> None:
    uri_str = build_hive_uri("myhost.local:9000", "sess_plain", "tok", "fp", tls=False)
    assert "tls=0" in uri_str
    parsed = parse_hive_uri(uri_str)
    assert parsed.tls is False
    assert parsed.base_http_url == "http://myhost.local:9000"


def test_uri_parse_errors() -> None:
    # Invalid scheme
    with pytest.raises(ProtocolError, match="Invalid URI scheme"):
        parse_hive_uri("http://localhost:9000/sess?t=tok&fp=fp")

    # Missing authority
    with pytest.raises(ProtocolError, match="Missing authority"):
        parse_hive_uri("hive:///sess?t=tok&fp=fp")

    # Missing session ID
    with pytest.raises(ProtocolError, match="Missing session ID"):
        parse_hive_uri("hive://localhost:9000/?t=tok&fp=fp")

    # Missing join token
    with pytest.raises(ProtocolError, match="Missing required join token"):
        parse_hive_uri("hive://localhost:9000/sess?fp=fp")

    # Missing fingerprint
    with pytest.raises(ProtocolError, match="Missing required host key fingerprint"):
        parse_hive_uri("hive://localhost:9000/sess?t=tok")


def test_build_uri_validation() -> None:
    with pytest.raises(ProtocolError):
        build_hive_uri("", "sess", "tok", "fp")

    with pytest.raises(ProtocolError):
        build_hive_uri("auth", "", "tok", "fp")

    with pytest.raises(ProtocolError):
        build_hive_uri("auth", "sess", "", "fp")

    with pytest.raises(ProtocolError):
        build_hive_uri("auth", "sess", "tok", "")