"""Focused adversarial contracts for private file helpers."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

from hermes_cli import private_files


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _symlink_dir_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")


def _symlink_file_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")


def test_connect_private_sqlite_allows_symlinked_directory_ancestors(tmp_path):
    real_home = tmp_path / "real-home"
    real_home.mkdir(mode=0o755)
    linked_home = tmp_path / "linked-home"
    _symlink_dir_or_skip(linked_home, real_home)
    db_path = linked_home / "nested" / "state ?# ü.db"

    previous = os.umask(0)
    try:
        conn = private_files.connect_private_sqlite(db_path)
    finally:
        os.umask(previous)
    try:
        conn.execute("CREATE TABLE sample (value TEXT)")
        conn.commit()
    finally:
        conn.close()

    assert linked_home.is_symlink()
    assert (real_home / "nested" / "state ?# ü.db").is_file()
    assert _mode(real_home) == 0o755
    assert _mode(real_home / "nested") == 0o700
    assert _mode(real_home / "nested" / "state ?# ü.db") == 0o600


def test_connect_private_sqlite_rejects_final_component_symlink(tmp_path):
    target = tmp_path / "target.db"
    target.write_bytes(b"do not touch")
    link = tmp_path / "state.db"
    _symlink_file_or_skip(link, target)

    with pytest.raises(OSError) as exc_info:
        private_files.connect_private_sqlite(link)

    assert exc_info.value.errno == errno.ELOOP
    assert target.read_bytes() == b"do not touch"


def test_atomic_private_writer_supports_symlinked_parent(tmp_path):
    real_output = tmp_path / "real-output"
    real_output.mkdir(mode=0o755)
    linked_output = tmp_path / "linked-output"
    _symlink_dir_or_skip(linked_output, real_output)
    destination = linked_output / "backup.zip"

    previous = os.umask(0)
    try:
        with private_files.open_private_atomic_binary(destination) as handle:
            handle.write(b"complete archive")
    finally:
        os.umask(previous)

    assert destination.read_bytes() == b"complete archive"
    assert _mode(real_output / "backup.zip") == 0o600
    assert _mode(real_output) == 0o755
    assert list(real_output.glob(".backup.zip.*.tmp")) == []


def test_prepare_sqlite_path_bounds_repeated_creation_races(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    real_open = private_files.os.open
    attempts = 0

    def racing_open(candidate, flags, mode=0o777):
        nonlocal attempts
        if Path(candidate) == path and flags & os.O_EXCL:
            attempts += 1
            raise FileExistsError(errno.EEXIST, "forced creation race", str(path))
        return real_open(candidate, flags, mode)

    monkeypatch.setattr(private_files.os, "open", racing_open)

    with pytest.raises(OSError) as exc_info:
        private_files.prepare_sqlite_path(path)

    assert exc_info.value.errno == errno.EAGAIN
    assert attempts <= 3
