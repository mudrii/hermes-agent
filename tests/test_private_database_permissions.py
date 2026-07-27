"""Creation-time permission contracts for every Hermes-owned SQLite store."""

from __future__ import annotations

import importlib
import os
import stat
from pathlib import Path

import pytest


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.parametrize(
    "module_name",
    [
        "agent.verification_evidence",
        "tools.async_delegation",
        "gateway.delivery_ledger",
    ],
)
def test_simple_sqlite_stores_create_owner_only_without_changing_parent(
    module_name, monkeypatch, tmp_path
):
    module = importlib.import_module(module_name)
    parent = tmp_path / module_name.replace(".", "-")
    parent.mkdir(mode=0o755)
    path = parent / "store.db"
    monkeypatch.setattr(module, "_db_path", lambda: path)

    previous = os.umask(0)
    try:
        conn = module._connect()
    finally:
        os.umask(previous)
    conn.close()

    assert _mode(path) == 0o600
    assert _mode(parent) == 0o755


def test_kanban_database_is_owner_only_without_changing_parent(tmp_path):
    from hermes_cli import kanban_db

    parent = tmp_path / "kanban-parent"
    parent.mkdir(mode=0o755)
    path = parent / "kanban.db"
    previous = os.umask(0)
    try:
        conn = kanban_db.connect(db_path=path)
    finally:
        os.umask(previous)
    conn.close()

    assert _mode(path) == 0o600
    assert _mode(parent) == 0o755


def test_response_store_database_is_owner_only_without_changing_parent(tmp_path):
    from gateway.platforms.api_server import ResponseStore

    parent = tmp_path / "response-parent"
    parent.mkdir(mode=0o755)
    path = parent / "response_store.db"
    previous = os.umask(0)
    try:
        store = ResponseStore(db_path=str(path))
    finally:
        os.umask(previous)
    store.close()

    assert _mode(path) == 0o600
    assert _mode(parent) == 0o755


def test_holographic_memory_database_is_owner_only_without_changing_parent(tmp_path):
    from plugins.memory.holographic.store import MemoryStore

    parent = tmp_path / "holographic-parent"
    parent.mkdir(mode=0o755)
    path = parent / "memory_store.db"
    previous = os.umask(0)
    try:
        store = MemoryStore(db_path=path)
    finally:
        os.umask(previous)
    store.close()

    assert _mode(path) == 0o600
    assert _mode(parent) == 0o755


def test_retaindb_queue_database_is_owner_only_without_changing_parent(tmp_path):
    from plugins.memory.retaindb import _WriteQueue

    class _Client:
        def ingest_session(self, *args, **kwargs):
            return None

    parent = tmp_path / "retaindb-parent"
    parent.mkdir(mode=0o755)
    path = parent / "pending.db"
    previous = os.umask(0)
    try:
        queue = _WriteQueue(_Client(), path)
    finally:
        os.umask(previous)
    try:
        assert _mode(path) == 0o600
        assert _mode(parent) == 0o755
    finally:
        queue.shutdown()
        conn = getattr(queue._local, "conn", None)
        if conn is not None:
            conn.close()


def test_discord_recovery_database_is_owner_only_without_changing_parent(tmp_path):
    from plugins.platforms.discord.recovery import DiscordRecoveryStore

    parent = tmp_path / "discord-home"
    parent.mkdir(mode=0o755)
    store = DiscordRecoveryStore(parent)
    previous = os.umask(0)
    try:
        assert store.call(lambda conn: conn.execute("SELECT 1").fetchone()) == (1,)
    finally:
        os.umask(previous)

    assert _mode(store.path()) == 0o600
    assert _mode(parent) == 0o755
