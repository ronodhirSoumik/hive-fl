"""
URI parsing and building for Hive join URLs.
Specification: hive://<authority>/<session_id>?t=<join_token>&fp=<fingerprint>&tls=<0|1>
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlparse
from hive.core.errors import ProtocolError


@dataclass(frozen=True)
class HiveURI:
    """Parsed representation of a Hive join URI."""
    authority: str
    session_id: str
    join_token: str
    fingerprint: str
    tls: bool = True

    @property
    def base_http_url(self) -> str:
        """Returns the HTTP(S) base URL corresponding to this Hive URI."""
        scheme = "https" if self.tls else "http"
        return f"{scheme}://{self.authority}"

    def to_uri_string(self) -> str:
        """Serializes the HiveURI back to canonical string form."""
        params = {
            "t": self.join_token,
            "fp": self.fingerprint,
            "tls": "1" if self.tls else "0",
        }
        query = urlencode(params)
        return f"hive://{self.authority}/{self.session_id}?{query}"

    def __str__(self) -> str:
        return self.to_uri_string()


def build_hive_uri(
    authority: str,
    session_id: str,
    join_token: str,
    fingerprint: str,
    tls: bool = True,
) -> str:
    """
    Constructs a canonical Hive join URI string.
    
    :param authority: Host and port or domain (e.g. 192.168.1.10:9000 or tunnel.trycloudflare.com)
    :param session_id: Unique session identifier
    :param join_token: Secret join token for admission
    :param fingerprint: Author host key Ed25519 fingerprint
    :param tls: True for HTTPS, False for plain HTTP
    """
    cleaned_auth = authority.strip().rstrip("/")
    if cleaned_auth.startswith("http://"):
        cleaned_auth = cleaned_auth[7:]
    elif cleaned_auth.startswith("https://"):
        cleaned_auth = cleaned_auth[8:]
    elif cleaned_auth.startswith("hive://"):
        cleaned_auth = cleaned_auth[7:]

    cleaned_session = session_id.strip().strip("/")
    if not cleaned_auth:
        raise ProtocolError("Cannot build Hive URI with empty authority", code="ERR_URI_MALFORMED")
    if not cleaned_session:
        raise ProtocolError("Cannot build Hive URI with empty session_id", code="ERR_URI_MALFORMED")
    if not join_token:
        raise ProtocolError("Cannot build Hive URI with empty join_token", code="ERR_URI_MALFORMED")
    if not fingerprint:
        raise ProtocolError("Cannot build Hive URI with empty fingerprint", code="ERR_URI_MALFORMED")

    obj = HiveURI(
        authority=cleaned_auth,
        session_id=cleaned_session,
        join_token=join_token.strip(),
        fingerprint=fingerprint.strip(),
        tls=tls,
    )
    return obj.to_uri_string()


def parse_hive_uri(uri_str: str) -> HiveURI:
    """
    Parses and validates a Hive join URI.
    Raises ProtocolError if scheme is invalid or any required components are missing.
    """
    raw = uri_str.strip()
    parsed = urlparse(raw)

    if parsed.scheme.lower() != "hive":
        raise ProtocolError(
            f"Invalid URI scheme '{parsed.scheme}'. Hive URIs must start with 'hive://'",
            code="ERR_URI_INVALID_SCHEME",
        )

    authority = parsed.netloc
    if not authority:
        raise ProtocolError("Missing authority (host:port) in Hive URI", code="ERR_URI_NO_AUTHORITY")

    path = parsed.path.strip("/")
    if not path:
        raise ProtocolError("Missing session ID in Hive URI path", code="ERR_URI_NO_SESSION_ID")

    session_id = path.split("/")[0]
    qs = parse_qs(parsed.query)

    tokens = qs.get("t")
    if not tokens or not tokens[0]:
        raise ProtocolError("Missing required join token query parameter 't'", code="ERR_URI_NO_TOKEN")
    join_token = tokens[0]

    fingerprints = qs.get("fp")
    if not fingerprints or not fingerprints[0]:
        raise ProtocolError("Missing required host key fingerprint query parameter 'fp'", code="ERR_URI_NO_FINGERPRINT")
    fingerprint = fingerprints[0]

    tls_vals = qs.get("tls", ["1"])
    tls = tls_vals[0].lower() not in ("0", "false", "no")

    return HiveURI(
        authority=authority,
        session_id=session_id,
        join_token=join_token,
        fingerprint=fingerprint,
        tls=tls,
    )