"""
Client configuration and credentials store.
Persists ResourcePolicy, node credentials, and consent records under ~/.hive/client/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hive.core.schemas import ResourcePolicy


def get_client_dir() -> Path:
    """Returns the base storage directory for client state (~/.hive/client)."""
    p = Path.home() / ".hive" / "client"
    p.mkdir(parents=True, exist_ok=True)
    return p


class ClientConfigStore:
    """Persistent storage manager for client configurations, credentials, and consents."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir else get_client_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.policy_file = self.base_dir / "resource_policy.json"
        self.creds_file = self.base_dir / "credentials.json"
        self.consent_file = self.base_dir / "consents.json"

    # ---------------- Resource Policy ----------------

    def load_resource_policy(self) -> ResourcePolicy:
        """Loads saved resource policy or returns default policy."""
        if not self.policy_file.exists():
            default_pol = ResourcePolicy()
            self.save_resource_policy(default_pol)
            return default_pol
        try:
            data = json.loads(self.policy_file.read_text(encoding="utf-8"))
            return ResourcePolicy.model_validate(data)
        except Exception:
            return ResourcePolicy()

    def save_resource_policy(self, policy: ResourcePolicy) -> None:
        """Persists resource policy to disk."""
        self.policy_file.write_text(json.dumps(policy.model_dump(), indent=2), encoding="utf-8")

    # ---------------- Node Credentials ----------------

    def load_node_credentials(self) -> dict[str, str] | None:
        """Loads stored node credentials if present."""
        if not self.creds_file.exists():
            return None
        try:
            return json.loads(self.creds_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_node_credentials(self, creds: dict[str, str]) -> None:
        """Saves active node session credentials (node_id, node_secret, uri)."""
        self.creds_file.write_text(json.dumps(creds, indent=2), encoding="utf-8")

    def clear_node_credentials(self) -> None:
        """Clears saved node credentials on leave/disconnect."""
        if self.creds_file.exists():
            self.creds_file.unlink(missing_ok=True)

    # ---------------- Consent Approvals ----------------

    def is_digest_consented(self, package_digest: str) -> bool:
        """Checks if a given package digest has been explicitly approved by the user."""
        if not self.consent_file.exists():
            return False
        try:
            consents = json.loads(self.consent_file.read_text(encoding="utf-8"))
            return bool(consents.get(package_digest))
        except Exception:
            return False

    def save_consent(self, package_digest: str, details: dict[str, Any] | None = None) -> None:
        """Records explicit consent for a package digest."""
        consents = {}
        if self.consent_file.exists():
            try:
                consents = json.loads(self.consent_file.read_text(encoding="utf-8"))
            except Exception:
                consents = {}
        consents[package_digest] = details or {"approved": True}
        self.consent_file.write_text(json.dumps(consents, indent=2), encoding="utf-8")

    def clear_all(self) -> None:
        """Wipes all local credentials and consents (e.g. on clean leave)."""
        self.clear_node_credentials()
        if self.consent_file.exists():
            self.consent_file.unlink(missing_ok=True)
