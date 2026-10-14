"""
SQLite persistence layer for Hive Host sessions, rounds, nodes, metrics, and audit events.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping

from hive.core.schemas import Capabilities, NodeState, SessionState


class HostDatabase:
    """Thread-safe SQLite database manager for Hive Host state persistence."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS session (
                session_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                data_requirements TEXT NOT NULL,
                package_digest TEXT NOT NULL,
                host_key_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                total_rounds INTEGER NOT NULL,
                current_round INTEGER NOT NULL,
                join_token TEXT NOT NULL,
                created_at REAL NOT NULL,
                closed_at REAL
            );

            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                nickname TEXT NOT NULL,
                secret_hash TEXT NOT NULL,
                state TEXT NOT NULL,
                capabilities_json TEXT NOT NULL,
                num_samples INTEGER DEFAULT 0,
                last_seen REAL NOT NULL,
                joined_at REAL NOT NULL,
                approved INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS rounds (
                round_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                opened_at REAL NOT NULL,
                deadline REAL NOT NULL,
                closed_at REAL,
                model_hash_in TEXT NOT NULL,
                model_hash_out TEXT,
                aggregator_used TEXT,
                fallback_note TEXT
            );

            CREATE TABLE IF NOT EXISTS claims (
                round_id INTEGER NOT NULL,
                node_id TEXT NOT NULL,
                claimed_at REAL NOT NULL,
                PRIMARY KEY (round_id, node_id),
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            );

            CREATE TABLE IF NOT EXISTS updates (
                round_id INTEGER NOT NULL,
                node_id TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                norm REAL NOT NULL,
                cosine REAL NOT NULL,
                accepted INTEGER NOT NULL,
                reject_reason TEXT,
                bytes INTEGER NOT NULL,
                uploaded_at REAL NOT NULL,
                PRIMARY KEY (round_id, node_id),
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            );

            CREATE TABLE IF NOT EXISTS metrics (
                round_id INTEGER PRIMARY KEY,
                train_loss REAL NOT NULL,
                eval_loss REAL NOT NULL,
                eval_accuracy REAL NOT NULL,
                participants INTEGER NOT NULL,
                duration_s REAL NOT NULL,
                FOREIGN KEY (round_id) REFERENCES rounds(round_id)
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS byte_accounting (
                node_id TEXT NOT NULL,
                category TEXT NOT NULL,
                bytes_in INTEGER DEFAULT 0,
                bytes_out INTEGER DEFAULT 0,
                PRIMARY KEY (node_id, category)
            );
            """)

    # ---------------- Session Operations ----------------

    def create_or_load_session(
        self,
        session_id: str,
        name: str,
        description: str,
        data_requirements: str,
        package_digest: str,
        host_key_fingerprint: str,
        total_rounds: int,
        join_token: str,
    ) -> dict[str, Any]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM session WHERE session_id = ?", (session_id,)).fetchone()
            if row:
                return dict(row)
            now = time.time()
            conn.execute(
                """
                INSERT INTO session (
                    session_id, name, description, data_requirements,
                    package_digest, host_key_fingerprint, status,
                    total_rounds, current_round, join_token, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, name, description, data_requirements,
                    package_digest, host_key_fingerprint, SessionState.RUNNING.value,
                    total_rounds, 0, join_token, now
                ),
            )
            return self.get_session() or {}

    def get_session(self) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM session LIMIT 1").fetchone()
            return dict(row) if row else None

    def update_session(self, **kwargs: Any) -> None:
        if not kwargs:
            return
        fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values())
        with self._get_connection() as conn:
            conn.execute(f"UPDATE session SET {fields}", values)

    # ---------------- Node Operations ----------------

    def add_node(
        self,
        node_id: str,
        nickname: str,
        secret_hash: str,
        capabilities: Capabilities | dict[str, Any],
        approved: bool = True,
    ) -> None:
        cap_json = json.dumps(capabilities.model_dump() if isinstance(capabilities, Capabilities) else capabilities)
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO nodes (node_id, nickname, secret_hash, state, capabilities_json, last_seen, joined_at, approved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (node_id, nickname, secret_hash, NodeState.REGISTERED.value, cap_json, now, now, 1 if approved else 0),
            )

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
            return dict(row) if row else None

    def list_nodes(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM nodes ORDER BY joined_at ASC").fetchall()
            return [dict(r) for r in rows]

    def update_node(self, node_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [node_id]
        with self._get_connection() as conn:
            conn.execute(f"UPDATE nodes SET {fields} WHERE node_id = ?", values)

    # ---------------- Round Operations ----------------

    def create_round(
        self,
        round_id: int,
        deadline: float,
        model_hash_in: str,
    ) -> None:
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO rounds (round_id, status, opened_at, deadline, model_hash_in)
                VALUES (?, 'open', ?, ?, ?)
                """,
                (round_id, now, deadline, model_hash_in),
            )

    def get_round(self, round_id: int) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM rounds WHERE round_id = ?", (round_id,)).fetchone()
            return dict(row) if row else None

    def get_latest_round(self) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM rounds ORDER BY round_id DESC LIMIT 1").fetchone()
            return dict(row) if row else None

    def list_rounds(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM rounds ORDER BY round_id ASC").fetchall()
            return [dict(r) for r in rows]

    def close_round(
        self,
        round_id: int,
        status: str,
        model_hash_out: str | None = None,
        aggregator_used: str | None = None,
        fallback_note: str | None = None,
    ) -> None:
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE rounds
                SET status = ?, closed_at = ?, model_hash_out = ?, aggregator_used = ?, fallback_note = ?
                WHERE round_id = ?
                """,
                (status, now, model_hash_out, aggregator_used, fallback_note, round_id),
            )

    # ---------------- Claims & Updates ----------------

    def add_claim(self, round_id: int, node_id: str) -> bool:
        now = time.time()
        with self._get_connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO claims (round_id, node_id, claimed_at) VALUES (?, ?, ?)",
                    (round_id, node_id, now),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def list_claims(self, round_id: int) -> list[str]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT node_id FROM claims WHERE round_id = ?", (round_id,)).fetchall()
            return [r["node_id"] for r in rows]

    def add_update(
        self,
        round_id: int,
        node_id: str,
        meta_json: str,
        norm: float,
        cosine: float,
        accepted: bool,
        reject_reason: str | None,
        byte_count: int,
    ) -> None:
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO updates (
                    round_id, node_id, meta_json, norm, cosine,
                    accepted, reject_reason, bytes, uploaded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (round_id, node_id, meta_json, norm, cosine, 1 if accepted else 0, reject_reason, byte_count, now),
            )

    def list_updates(self, round_id: int) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM updates WHERE round_id = ?", (round_id,)).fetchall()
            return [dict(r) for r in rows]

    def list_all_updates(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM updates ORDER BY round_id ASC").fetchall()
            return [dict(r) for r in rows]

    # ---------------- Metrics & Events ----------------

    def save_metrics(
        self,
        round_id: int,
        train_loss: float,
        eval_loss: float,
        eval_accuracy: float,
        participants: int,
        duration_s: float,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO metrics (
                    round_id, train_loss, eval_loss, eval_accuracy, participants, duration_s
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (round_id, train_loss, eval_loss, eval_accuracy, participants, duration_s),
            )

    def list_metrics(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM metrics ORDER BY round_id ASC").fetchall()
            return [dict(r) for r in rows]

    def add_event(self, level: str, message: str) -> None:
        now = time.time()
        with self._get_connection() as conn:
            conn.execute("INSERT INTO events (ts, level, message) VALUES (?, ?, ?)", (now, level, message))

    def list_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in reversed(rows)]

    # ---------------- Byte Accounting ----------------

    def record_bytes(self, node_id: str, category: str, bytes_in: int = 0, bytes_out: int = 0) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO byte_accounting (node_id, category, bytes_in, bytes_out)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(node_id, category) DO UPDATE SET
                    bytes_in = bytes_in + excluded.bytes_in,
                    bytes_out = bytes_out + excluded.bytes_out
                """,
                (node_id, category, bytes_in, bytes_out),
            )

    def get_byte_accounting_summary(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM byte_accounting").fetchall()
            return [dict(r) for r in rows]