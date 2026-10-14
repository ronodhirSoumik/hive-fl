"""
Standard error hierarchy for Hive with stable machine-readable error codes.
"""

from __future__ import annotations


class HiveError(Exception):
    """Base exception for all Hive domain errors."""

    def __init__(self, message: str, code: str = "ERR_HIVE_INTERNAL") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class PackageError(HiveError):
    """Raised when a package archive is malformed, untrusted, or fails linting."""

    def __init__(self, message: str, code: str = "ERR_PACKAGE_INVALID") -> None:
        super().__init__(message, code=code)


class SignatureError(HiveError):
    """Raised when cryptographic verification of a package signature fails."""

    def __init__(self, message: str, code: str = "ERR_SIGNATURE_INVALID") -> None:
        super().__init__(message, code=code)


class ProtocolError(HiveError):
    """Raised on invalid protocol messages, unauthorized requests, or round state violations."""

    def __init__(self, message: str, code: str = "ERR_PROTOCOL_VIOLATION") -> None:
        super().__init__(message, code=code)


class SandboxError(HiveError):
    """Raised when the sandbox execution environment fails, timeouts, or violates policies."""

    def __init__(self, message: str, code: str = "ERR_SANDBOX_FAILED") -> None:
        super().__init__(message, code=code)


class ConsentRequired(HiveError):
    """Raised when package code has not been explicitly approved/consented by contributor."""

    def __init__(self, message: str = "Contributor consent required for this package", code: str = "ERR_CONSENT_REQUIRED") -> None:
        super().__init__(message, code=code)


class ResourceError(HiveError):
    """Raised when hardware or policy constraints prevent execution."""

    def __init__(self, message: str, code: str = "ERR_RESOURCE_LIMIT") -> None:
        super().__init__(message, code=code)


class AggregationError(HiveError):
    """Raised when server-side weight aggregation cannot proceed or fails Byzantine checks."""

    def __init__(self, message: str, code: str = "ERR_AGGREGATION_FAILED") -> None:
        super().__init__(message, code=code)


class ConfigurationError(HiveError):
    """Raised when supplied configuration values are invalid or contradictory."""

    def __init__(self, message: str, code: str = "ERR_CONFIG_INVALID") -> None:
        super().__init__(message, code=code)