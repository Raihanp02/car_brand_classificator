"""SQLite API-key auth (car-classifier v1).

Tables (spec section 6b)::

    api_keys   id PK, key_hash TEXT UNIQUE, name, created_at, revoked, rate_limit_per_day
    usage_logs id PK, key_id FK, ts, endpoint, top1_label, confidence,
               latency_ms, status_code, ip

Only the sha256 ``key_hash`` is stored; raw keys (``ck_...``) are shown once
at creation. Postgres migration keeps the same three public signatures.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "backend" / "data" / "keys.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys(
    id INTEGER PRIMARY KEY,
    key_hash TEXT UNIQUE,
    name TEXT,
    created_at TEXT,
    revoked INTEGER DEFAULT 0,
    rate_limit_per_day INTEGER DEFAULT 1000
);
CREATE TABLE IF NOT EXISTS usage_logs(
    id INTEGER PRIMARY KEY,
    key_id INTEGER REFERENCES api_keys(id),
    ts TEXT,
    endpoint TEXT,
    top1_label TEXT,
    confidence REAL,
    latency_ms INTEGER,
    status_code INTEGER,
    ip TEXT
);
"""


def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.executescript(SCHEMA)
    return con


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def create_key(name: str, rate_limit_per_day: int = 1000) -> str:
    """Create a key named ``name``; return the raw ``ck_...`` secret (once)."""
    raw = "ck_" + secrets.token_urlsafe(24)
    con = _db()
    try:
        con.execute(
            "INSERT INTO api_keys(key_hash, name, created_at, revoked,"
            " rate_limit_per_day) VALUES (?, ?, ?, 0, ?)",
            (
                _hash(raw),
                name,
                datetime.now(timezone.utc).isoformat(),
                rate_limit_per_day,
            ),
        )
        con.commit()
    finally:
        con.close()
    return raw


def validate_key(raw: str) -> int | None:
    """Return ``api_keys.id`` if ``raw`` matches a non-revoked key, else None.

    Falsy when invalid, so ``if validate_key(k):`` works as a boolean check.
    """
    if not raw:
        return None
    con = _db()
    try:
        row = con.execute(
            "SELECT id, revoked FROM api_keys WHERE key_hash = ?",
            (_hash(raw),),
        ).fetchone()
    finally:
        con.close()
    if row is None or row[1]:
        return None
    return int(row[0])


def revoke_key(raw: str) -> bool:
    """Revoke the key matching ``raw``; True if a row was revoked."""
    if not raw:
        return False
    con = _db()
    try:
        cur = con.execute(
            "UPDATE api_keys SET revoked = 1 WHERE key_hash = ?", (_hash(raw),)
        )
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()


def key_daily_usage(key_id: int, day_iso: str) -> int:
    """Count SERVED predictions (2xx) for ``key_id`` with ``ts >= day_iso``.

    Failed attempts (4xx/5xx) are logged but never consume quota.
    """
    con = _db()
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM usage_logs"
            " WHERE key_id = ? AND ts >= ?"
            " AND status_code BETWEEN 200 AND 299",
            (key_id, day_iso),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def key_rate_limit(key_id: int) -> int:
    con = _db()
    try:
        row = con.execute(
            "SELECT rate_limit_per_day FROM api_keys WHERE id = ?", (key_id,)
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 1000
    finally:
        con.close()


def log_usage(
    key_id: int | None,
    endpoint: str,
    top1_label: str,
    confidence: float,
    latency_ms: int,
    status_code: int,
    ip: str,
) -> None:
    """Insert one usage_logs row (``key_id`` None = keyless demo hit)."""
    con = _db()
    try:
        con.execute(
            "INSERT INTO usage_logs(key_id, ts, endpoint, top1_label,"
            " confidence, latency_ms, status_code, ip)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key_id,
                datetime.now(timezone.utc).isoformat(),
                endpoint,
                top1_label,
                float(confidence),
                int(latency_ms),
                int(status_code),
                ip,
            ),
        )
        con.commit()
    finally:
        con.close()
