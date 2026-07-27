"""Fail-closed helpers for private Hermes files and SQLite databases.

The helpers in this module establish restrictive permissions at creation time,
reject symbolic-link destinations, and never chmod an existing parent
directory.  They are intentionally small so backup, session, projects, and
cron state use one security contract.
"""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
import sqlite3
import stat
import warnings
from pathlib import Path
from typing import IO, Any, Callable, Iterator, Union

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
Pathish = Union[str, os.PathLike]


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _reject_symlink(path: Path, st: os.stat_result | None = None) -> None:
    st = _lstat(path) if st is None else st
    if st is not None and stat.S_ISLNK(st.st_mode):
        raise OSError(errno.ELOOP, f"refusing symlink path: {path}", str(path))


def _verify_mode(path: Path, expected: int) -> None:
    """Verify a POSIX mode, warning explicitly on Windows.

    Windows' chmod does not provide an owner-only ACL guarantee.  Callers still
    get creation without following the final symlink, but the limitation must
    never be silent.
    """
    if os.name == "nt":
        warnings.warn(
            f"owner-only permissions cannot be guaranteed for {path} on Windows",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    actual = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    if actual != expected:
        raise PermissionError(
            f"could not enforce owner-only mode {oct(expected)} on {path}; "
            f"actual mode is {oct(actual)}"
        )


def ensure_private_directory(path: Pathish) -> bool:
    """Ensure *path* is a directory; create missing components as 0700.

    Existing directories are validated but never chmodded.  Returns ``True``
    only when the final directory was created by this call.
    """
    path = Path(path).expanduser()
    st = _lstat(path)
    _reject_symlink(path, st)
    if st is not None:
        if not stat.S_ISDIR(st.st_mode):
            raise NotADirectoryError(f"private directory path is not a directory: {path}")
        return False

    parent = path.parent
    if parent != path:
        ensure_private_directory(parent)
    try:
        os.mkdir(path, PRIVATE_DIR_MODE)
        created = True
    except FileExistsError:
        # Another creator won the race. Validate it; do not chmod an object we
        # did not create.
        st = _lstat(path)
        _reject_symlink(path, st)
        if st is None or not stat.S_ISDIR(st.st_mode):
            raise NotADirectoryError(f"private directory path is not a directory: {path}")
        return False

    try:
        os.chmod(path, PRIVATE_DIR_MODE, follow_symlinks=False)
    except (NotImplementedError, TypeError):
        if os.name != "nt":
            raise
    _verify_mode(path, PRIVATE_DIR_MODE)
    return created


def _open_private_fd(path: Path, *, truncate: bool) -> int:
    path = Path(path).expanduser()
    ensure_private_directory(path.parent)
    st_before = _lstat(path)
    _reject_symlink(path, st_before)
    if st_before is not None and not stat.S_ISREG(st_before.st_mode):
        raise OSError(errno.EINVAL, f"private file path is not a regular file: {path}")

    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, PRIVATE_FILE_MODE)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OSError(errno.ELOOP, f"refusing symlink path: {path}", str(path)) from exc
        raise

    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError(errno.EINVAL, f"private file path is not a regular file: {path}")
        try:
            os.fchmod(fd, PRIVATE_FILE_MODE)
        except (AttributeError, NotImplementedError):
            try:
                os.chmod(path, PRIVATE_FILE_MODE, follow_symlinks=False)
            except (NotImplementedError, TypeError):
                if os.name != "nt":
                    raise
        current = os.stat(path, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError(errno.EAGAIN, f"private file path changed while opening: {path}")
        _verify_mode(path, PRIVATE_FILE_MODE)
        if truncate:
            os.ftruncate(fd, 0)
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextlib.contextmanager
def open_private_binary(path: Pathish) -> Iterator[IO[bytes]]:
    """Open *path* for binary replacement through a verified 0600 descriptor."""
    fd = _open_private_fd(Path(path), truncate=True)
    handle = os.fdopen(fd, "w+b")
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()


@contextlib.contextmanager
def open_private_text(path: Pathish, *, encoding: str = "utf-8") -> Iterator[IO[str]]:
    """Open *path* for text replacement through a verified 0600 descriptor."""
    fd = _open_private_fd(Path(path), truncate=True)
    handle = os.fdopen(fd, "w", encoding=encoding)
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()


def copy_private_file(src: Pathish, dst: Pathish) -> None:
    """Copy a regular non-symlink file to a 0600 destination."""
    src = Path(src)
    st = _lstat(src)
    _reject_symlink(src, st)
    if st is None or not stat.S_ISREG(st.st_mode):
        raise OSError(errno.EINVAL, f"snapshot source is not a regular file: {src}")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        source_fd = os.open(src, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OSError(errno.ELOOP, f"refusing symlink path: {src}", str(src)) from exc
        raise

    try:
        opened = os.fstat(source_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError(errno.EINVAL, f"snapshot source is not a regular file: {src}")
        current = os.stat(src, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError(errno.EAGAIN, f"snapshot source changed while opening: {src}")
        with os.fdopen(source_fd, "rb", closefd=False) as source:
            with open_private_binary(dst) as destination:
                shutil.copyfileobj(source, destination)
    finally:
        os.close(source_fd)


def sqlite_file_uri(path: Pathish, *, mode: str) -> str:
    """Return an encoded absolute SQLite URI for a literal filesystem path."""
    if mode not in {"ro", "rw"}:
        raise ValueError(f"unsupported SQLite mode: {mode}")
    absolute = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    # SQLite accepts the single-slash absolute URI form. Keep it for backward
    # compatibility with callers/tests that inspect the literal connection
    # target while retaining Path.as_uri()'s percent-encoding.
    encoded = absolute.as_uri().replace("file:///", "file:/", 1)
    return f"{encoded}?mode={mode}"


def prepare_sqlite_path(path: Pathish, *, read_only: bool = False) -> str:
    """Validate or privately create a SQLite file and return its encoded URI.

    Writable databases are pre-created as 0600 and then opened with ``mode=rw``
    so SQLite never performs an umask-controlled creation. Existing files are
    left at their operator-selected mode; migration of historical artifacts is
    a separate, explicit maintenance operation.
    """
    path = Path(path).expanduser()
    st = _lstat(path)
    _reject_symlink(path, st)
    if st is not None:
        if not stat.S_ISREG(st.st_mode):
            raise OSError(errno.EINVAL, f"SQLite path is not a regular file: {path}")
    elif read_only:
        raise FileNotFoundError(path)
    else:
        ensure_private_directory(path.parent)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags, PRIVATE_FILE_MODE)
        except FileExistsError:
            # A concurrent creator won. Re-run validation instead of following
            # whatever appeared at the path.
            return prepare_sqlite_path(path, read_only=read_only)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise OSError(
                    errno.ELOOP, f"refusing symlink path: {path}", str(path)
                ) from exc
            raise
        try:
            try:
                os.fchmod(fd, PRIVATE_FILE_MODE)
            except (AttributeError, NotImplementedError):
                pass
        finally:
            os.close(fd)
        _verify_mode(path, PRIVATE_FILE_MODE)

    return sqlite_file_uri(path, mode="ro" if read_only else "rw")


def connect_private_sqlite(
    path: Pathish,
    *,
    read_only: bool = False,
    connect: Callable[..., sqlite3.Connection] = sqlite3.connect,
    **kwargs: Any,
) -> sqlite3.Connection:
    """Open a validated SQLite path without allowing SQLite to create it."""
    uri = prepare_sqlite_path(path, read_only=read_only)
    return connect(uri, uri=True, **kwargs)
