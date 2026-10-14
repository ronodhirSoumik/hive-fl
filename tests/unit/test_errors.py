"""
Unit tests for Hive error hierarchy and error codes.
"""

from hive.core.errors import (
    AggregationError,
    ConfigurationError,
    ConsentRequired,
    HiveError,
    PackageError,
    ProtocolError,
    ResourceError,
    SandboxError,
    SignatureError,
)


def test_error_hierarchy() -> None:
    err = PackageError("Package corrupted", code="ERR_PACKAGE_CORRUPT")
    assert isinstance(err, HiveError)
    assert str(err) == "[ERR_PACKAGE_CORRUPT] Package corrupted"
    assert err.to_dict() == {"code": "ERR_PACKAGE_CORRUPT", "message": "Package corrupted"}


def test_default_codes() -> None:
    assert HiveError("msg").code == "ERR_HIVE_INTERNAL"
    assert PackageError("msg").code == "ERR_PACKAGE_INVALID"
    assert SignatureError("msg").code == "ERR_SIGNATURE_INVALID"
    assert ProtocolError("msg").code == "ERR_PROTOCOL_VIOLATION"
    assert SandboxError("msg").code == "ERR_SANDBOX_FAILED"
    assert ConsentRequired().code == "ERR_CONSENT_REQUIRED"
    assert ResourceError("msg").code == "ERR_RESOURCE_LIMIT"
    assert AggregationError("msg").code == "ERR_AGGREGATION_FAILED"
    assert ConfigurationError("msg").code == "ERR_CONFIG_INVALID"