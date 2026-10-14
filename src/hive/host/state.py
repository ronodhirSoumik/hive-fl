"""
In-memory session state management for Hive Host.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

import torch

from hive.core.schemas import SessionState


class HostState:
    """Manages volatile runtime state, locks, and active in-memory models."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.status: SessionState = SessionState.IDLE
        self.global_weights: dict[str, torch.Tensor] = {}
        self.model_hash: str = ""
        self.current_round_id: int = 0
        self.active_claims: set[str] = set()
        self.round_updates: list[Any] = []
        self.admin_token: str = ""
        self.join_token: str = ""
        self.public_url: str = ""
        self.is_tls: bool = False