r"""Contain the base class to implement a data loader or saver
object."""

from __future__ import annotations

__all__ = [
    "BaseFileSaver",
    "BaseLoader",
    "BaseSaver",
    "is_loader_config",
    "is_saver_config",
    "resolve_loader",
    "resolve_saver",
]

import contextlib
import logging
import os
import socket
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from coola.factory import is_object_config, resolve_object
from coola.io.utils import add_uuid_suffix

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path

T = TypeVar("T")

logger: logging.Logger = logging.getLogger(__name__)

# Per-path locks used by ``BaseFileSaver.save`` to serialize concurrent
# ``save`` calls targeting the same path within this process. A
# ``WeakValueDictionary`` is used so a lock is discarded automatically once
# no ``save`` call references it anymore, instead of accumulating one entry
# per distinct path saved over the lifetime of the process.
_SAVE_LOCKS_GUARD = threading.Lock()
_SAVE_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()

# Default timeout/poll interval for the cross-process file lock acquired by
# ``_save_lock`` below.
_FILE_LOCK_TIMEOUT = 60.0
_FILE_LOCK_POLL_INTERVAL = 0.01
# A lock file older than this many seconds is assumed to have been left
# behind by a process that died while holding it, and is broken.
_FILE_LOCK_STALE_AFTER = 300.0


def _get_save_lock(path: Path) -> threading.Lock:
    r"""Get (creating if needed) the lock used to serialize ``save``
    calls targeting the given path."""
    key = str(path)
    with _SAVE_LOCKS_GUARD:
        lock = _SAVE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SAVE_LOCKS[key] = lock
        return lock


def _make_lock_token() -> str:
    r"""Return a unique token that identifies the owner of a lock.

    The token starts with the host name and the process id (``host:pid:``)
    so a waiter can tell that the owner is dead.
    """
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


def _lock_owner_is_dead(lock_path: Path) -> bool:
    r"""Indicate if the process that owns the lock file is dead.

    Only a lock taken on this host by a process that does not exist
    anymore is reported as dead. If the owner cannot be found out (other
    host, unknown token format, Windows), ``False`` is returned.
    """
    if os.name != "posix":
        return False
    try:
        host, pid, _ = lock_path.read_text().split(":", maxsplit=2)
        if host != socket.gethostname():
            return False
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except (OSError, ValueError):
        return False
    return False


def _break_stale_lock(lock_path: Path, stale_after: float) -> bool:
    r"""Remove ``lock_path`` if its age exceeds ``stale_after`` seconds,
    or if the process that owns it is dead.

    The file is first renamed to a unique name so that, if several
    waiters notice the same stale lock, only one of them removes it.

    Returns:
        ``True`` if a stale lock was removed.
    """
    try:
        age = time.time() - lock_path.stat().st_mtime
        if age <= stale_after and not _lock_owner_is_dead(lock_path):
            return False
        broken = lock_path.with_name(f"{lock_path.name}.{os.getpid()}.{uuid.uuid4().hex}.stale")
        lock_path.rename(broken)
    except FileNotFoundError:
        return False
    broken.unlink(missing_ok=True)
    logger.warning("Removed stale save lock %s (age %.1fs, limit %ss)", lock_path, age, stale_after)
    return True


def _acquire_file_lock(
    path: Path,
    timeout: float = _FILE_LOCK_TIMEOUT,
    poll_interval: float = _FILE_LOCK_POLL_INTERVAL,
    stale_after: float = _FILE_LOCK_STALE_AFTER,
    token: str = "",
) -> Path:
    r"""Create a lock file next to ``path``, blocking until it can be
    created exclusively.

    ``os.open`` with ``O_CREAT | O_EXCL`` atomically fails with
    ``FileExistsError`` if the lock file already exists, which makes
    this safe as a mutual-exclusion primitive across processes, unlike
    the in-process-only ``threading.Lock`` returned by
    ``_get_save_lock``.

    A lock file older than ``stale_after`` seconds is treated as left
    over by a crashed process and is broken, so a killed writer does not
    block every later save.
    """
    lock_path = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, token.encode())
            finally:
                os.close(fd)
        except FileExistsError:
            if _break_stale_lock(lock_path, stale_after):
                continue
            if time.monotonic() >= deadline:
                msg = (
                    f"timed out after {timeout}s waiting for the save lock on {path} "
                    f"({lock_path} still exists)"
                )
                raise TimeoutError(msg) from None
            time.sleep(poll_interval)
        else:
            return lock_path


def _tmp_path(path: Path) -> Path:
    r"""Return the path of the temporary file used to write ``path``.

    The name starts with a dot, so the file is hidden and a reader that
    lists the directory does not take it for the final file while it is
    being written. The extension is kept because some writers choose the
    format or add an extension from it.
    """
    return add_uuid_suffix(path.with_name(f".{path.name}"))


def _fsync_file(path: Path) -> None:
    r"""Flush the content of a file to the disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    r"""Flush the entries of a directory to the disk, so a rename or a
    link made in it survives a power failure.

    It is a best effort: some platforms and filesystems do not support
    opening or syncing a directory.
    """
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _release_file_lock(lock_path: Path, token: str) -> None:
    r"""Remove the lock file, only if it is still owned by ``token``.

    If the lock was broken as stale by another process, the file now
    belongs to this other process and must not be removed.
    """
    try:
        owner = lock_path.read_text()
    except FileNotFoundError:
        return
    if owner != token:
        logger.warning("The save lock %s is owned by another process: not removed", lock_path)
        return
    lock_path.unlink(missing_ok=True)


def _keep_lock_fresh(lock_path: Path, token: str, interval: float, stop: threading.Event) -> None:
    r"""Update the modification time of the lock file until ``stop`` is
    set, so a slow save is not mistaken for a dead process.

    The thread stops if the lock is not owned by ``token`` anymore.
    """
    while not stop.wait(interval):
        try:
            if lock_path.read_text() != token:
                return
            os.utime(lock_path)
        except OSError:
            return


@contextlib.contextmanager
def _save_lock(path: Path, stale_after: float = _FILE_LOCK_STALE_AFTER) -> Generator[None]:
    with _get_save_lock(path):
        token = _make_lock_token()
        lock_path = _acquire_file_lock(path, stale_after=stale_after, token=token)
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=_keep_lock_fresh,
            args=(lock_path, token, stale_after / 3, stop),
            name="coola-save-lock-heartbeat",
            daemon=True,
        )
        heartbeat.start()
        try:
            yield
        finally:
            stop.set()
            heartbeat.join()
            _release_file_lock(lock_path, token)


class BaseLoader(ABC, Generic[T]):
    r"""Define the base class to implement a data loader.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_json, JsonLoader
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.json")
        ...     save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = JsonLoader().load(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.io import JsonLoader, TextLoader
            >>> JsonLoader().equal(JsonLoader())
            True
            >>> JsonLoader().equal(TextLoader())
            False

            ```
        """

    @abstractmethod
    def load(self, path: Path) -> T:
        r"""Load the data from the given path.

        Args:
            path: The path with the data to load.

        Returns:
            The data

        Example:
            ```pycon
            >>> import tempfile
            >>> from pathlib import Path
            >>> from coola.io import save_json, JsonLoader
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = Path(tmpdir).joinpath("data.json")
            ...     save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
            ...     data = JsonLoader().load(path)
            ...     data
            ...
            {'key1': [1, 2, 3], 'key2': 'abc'}

            ```
        """


class BaseSaver(ABC, Generic[T]):
    r"""Define the base class to implement a data saver.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import JsonSaver, JsonLoader
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.json")
        ...     JsonSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = JsonLoader().load(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.io import JsonSaver, TextSaver
            >>> JsonSaver().equal(JsonSaver())
            True
            >>> JsonSaver().equal(TextSaver())
            False

            ```
        """

    @abstractmethod
    def save(self, to_save: T, path: Path, *, exist_ok: bool = False) -> None:
        r"""Save the data into the given path.

        Args:
            to_save: The data to save. The data should be compatible
                with the saving engine.
            path: The path where to save the data.
            exist_ok: If ``exist_ok`` is ``False`` (the default),
                an exception is raised if the target path already
                exists.

        Example:
            ```pycon
            >>> import tempfile
            >>> from pathlib import Path
            >>> from coola.io import JsonSaver, JsonLoader
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = Path(tmpdir).joinpath("data.json")
            ...     JsonSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
            ...     data = JsonLoader().load(path)
            ...     data
            ...
            {'key1': [1, 2, 3], 'key2': 'abc'}

            ```
        """


class BaseFileSaver(BaseSaver[T]):
    r"""Define the base class to implement a file saver.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import JsonSaver, JsonLoader
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.json")
        ...     JsonSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = JsonLoader().load(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """

    def save(self, to_save: T, path: Path, *, exist_ok: bool = False, fsync: bool = False) -> None:
        r"""Save the data into the given path.

        Args:
            to_save: The data to save. The data should be compatible
                with the saving engine.
            path: The path where to save the data.
            exist_ok: If ``exist_ok`` is ``False`` (the default),
                ``FileExistsError`` is raised if the target file
                already exists. If ``exist_ok`` is ``True``,
                ``FileExistsError`` will not be raised unless the
                given path already exists in the file system and is
                not a file.

                Note: the ``exist_ok=False`` guarantee is protected
                against a concurrent creation of ``path`` between the
                initial existence check and the commit step (the
                commit uses ``os.link`` which atomically fails with
                ``FileExistsError`` in that case).

                Note: concurrent ``save`` calls targeting the same
                ``path``, whether from *within this process* (an
                in-memory lock) or from *other processes* (a
                ``path.lock`` lock file created next to ``path``), are
                serialized so they cannot interleave their
                write/commit steps and corrupt or partially overwrite
                each other's data. This holds for ``exist_ok=True``
                too: two concurrent calls with ``exist_ok=True`` can no
                longer race each other's ``tmp_path.replace(path)``
                commit; each call's replace happens fully before the
                next one starts. Which call's data ends up on disk is
                still whichever one wins the lock last (no ordering is
                guaranteed beyond "no interleaving"), and the lock file
                is removed once the holder releases it, so it does not
                accumulate on disk.

                The lock file is kept fresh while the data is written,
                so a long save is not mistaken for a dead process. A
                lock file that was not refreshed for five minutes
                (the process died) is broken. The data is first
                written to a hidden temporary file next to ``path``
                (``.<name>-<uuid>.<ext>``), which is then committed.

            fsync: If ``True``, the data is flushed to the disk before
                the file is committed, and the directory is flushed
                after, so the file is not empty or missing after a
                power failure or a system crash. It is slower.
                If ``False`` (the default), the data is only written
                to the operating system cache: the file is still
                replaced atomically (a reader never sees a partially
                written file), but its content can be lost if the
                machine crashes right after the call. This is fine for
                caches and most logs.

        Note:
            The caller needs write access to the *directory* of
            ``path``, not only to the file: the data is first written
            to a temporary file next to ``path``, and a ``path.lock``
            file is created there while the file is saved (see the
            ``exist_ok`` argument). The saved file is committed with a
            hard link, or with a rename if the filesystem does not
            support hard links (e.g. FAT/exFAT, and some network and
            container volumes).

        Raises:
            FileExistsError: if the file already exists and ``exist_ok`` is ``False``.
            IsADirectoryError: If ``path`` is a directory.

        Example:
            ```pycon
            >>> import tempfile
            >>> from pathlib import Path
            >>> from coola.io import JsonSaver, JsonLoader
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     path = Path(tmpdir).joinpath("data.json")
            ...     JsonSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
            ...     data = JsonLoader().load(path)
            ...     data
            ...
            {'key1': [1, 2, 3], 'key2': 'abc'}

            ```
        """
        if path.is_dir():
            msg = f"path ({path}) is a directory"
            raise IsADirectoryError(msg)
        if path.is_file() and not exist_ok:
            msg = f"path ({path}) already exists. Use `exist_ok=True` to overwrite the file"
            raise FileExistsError(msg)
        path.parent.mkdir(exist_ok=True, parents=True)

        # Serialize concurrent ``save`` calls targeting the same path, both
        # in-process and across processes (see the note in the docstring):
        # without this, two calls could each write their own tmp file and
        # then interleave their commit steps below, e.g. call A's
        # ``os.link``/``replace`` landing after call B's, silently
        # discarding whichever of A's or B's data lost the race in a
        # non-deterministic, hard-to-reproduce way.
        with _save_lock(path):
            # Save to tmp, then commit by moving the file in case the job
            # gets interrupted while writing the file.
            tmp_path = _tmp_path(path)
            try:
                self._save_file(to_save, tmp_path)
                if fsync:
                    _fsync_file(tmp_path)
                if exist_ok:
                    tmp_path.replace(path)
                else:
                    # ``os.link`` + unlink is used instead of a plain rename so the
                    # ``exist_ok=False`` guarantee also holds if ``path`` was created
                    # concurrently between the check above and this commit step:
                    # ``link`` atomically fails with ``FileExistsError`` in that case.
                    try:
                        self._link(tmp_path, path)
                    finally:
                        # Unlinks ``tmp_path`` whether ``os.link`` succeeded (the
                        # data now lives at ``path`` via the hard link) or failed
                        # (nothing to keep); the redundant ``unlink(missing_ok=True)``
                        # below in the outer ``except`` is then a no-op. Any error
                        # from this cleanup ``unlink`` itself (e.g. a permission
                        # issue) is suppressed rather than propagated, so it cannot
                        # mask an in-flight ``os.link`` exception.
                        with contextlib.suppress(OSError):
                            tmp_path.unlink(missing_ok=True)
            except BaseException:
                # Covers a failure in ``_save_file`` itself (``tmp_path`` may or
                # may not have been created) as well as a failure of
                # ``tmp_path.replace(path)`` in the ``exist_ok=True`` branch;
                # ``missing_ok=True`` makes this safe to call even when the
                # ``finally`` above (or ``_save_file``) already removed it. The
                # cleanup ``unlink`` is wrapped so that if it itself raises
                # (e.g. a permission error), that secondary error is suppressed
                # and the original exception being handled here still propagates
                # unmasked.
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)
                raise
            if fsync:
                _fsync_directory(path.parent)

    @staticmethod
    def _link(tmp_path: Path, path: Path) -> None:
        r"""Commit ``tmp_path`` to ``path`` without overwriting ``path``.

        ``os.link`` is atomic: it fails with ``FileExistsError`` if
        ``path`` exists. When the filesystem does not support hard links,
        it falls back on a check followed by a rename. This is safe
        between the processes that use ``save``, because they hold the
        lock of ``path`` while committing.

        Args:
            tmp_path: The temporary file with the data.
            path: The target path, which must not exist.

        Raises:
            FileExistsError: if ``path`` exists.
        """
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            raise
        except OSError:
            if path.exists():
                msg = f"path ({path}) already exists. Use `exist_ok=True` to overwrite the file"
                raise FileExistsError(msg) from None
            tmp_path.replace(path)

    @abstractmethod
    def _save_file(self, to_save: T, path: Path) -> None:
        r"""Save the data into the given file.

        Args:
            to_save: The data to save. The data should be compatible
                with the saving engine.
            path: The path where to save the data.
        """


def is_loader_config(
    config: dict[Any, Any], *, allowed_prefixes: Sequence[str] | None = None
) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseLoader``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.
        allowed_prefixes: An optional allow-list of module prefixes that
            the ``"_target_"`` must match, e.g. ``["coola.io"]``. If
            ``None`` (default), any ``"_target_"`` is allowed.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseLoader`` object.

    Security:
        The ``"_target_"`` is imported to check its class: see
        :func:`coola.factory.is_object_config`.

    Example:
        ```pycon
        >>> from coola.io import is_loader_config
        >>> is_loader_config({"_target_": "coola.io.JsonLoader"})
        True

        ```
    """
    return is_object_config(config, BaseLoader, allowed_prefixes=allowed_prefixes)


def is_saver_config(
    config: dict[Any, Any], *, allowed_prefixes: Sequence[str] | None = None
) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseSaver``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.
        allowed_prefixes: An optional allow-list of module prefixes that
            the ``"_target_"`` must match, e.g. ``["coola.io"]``. If
            ``None`` (default), any ``"_target_"`` is allowed.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseSaver`` object.

    Security:
        The ``"_target_"`` is imported to check its class: see
        :func:`coola.factory.is_object_config`.

    Example:
        ```pycon
        >>> from coola.io import is_saver_config
        >>> is_saver_config({"_target_": "coola.io.JsonSaver"})
        True

        ```
    """
    return is_object_config(config, BaseSaver, allowed_prefixes=allowed_prefixes)


def resolve_loader(
    loader: BaseLoader[T] | dict[Any, Any], *, allowed_prefixes: Sequence[str] | None = None
) -> BaseLoader[T]:
    r"""Set up a data loader.

    The data loader is instantiated from its configuration by using the
    ``BaseLoader`` factory function.

    Args:
        loader: The data loader or its configuration.
        allowed_prefixes: An optional allow-list of module prefixes that
            the ``"_target_"`` must match, e.g. ``["coola.io"]``. If
            ``None`` (default), any ``"_target_"`` is allowed.

    Returns:
        The instantiated data loader.

    Security:
        When ``loader`` is a configuration :class:`dict`, its
        ``"_target_"`` value is imported and instantiated with no
        allowlist by default (see :func:`coola.factory.factory`). Only
        call this function with a configuration that comes from a
        trusted source, or restrict it with ``allowed_prefixes``; never
        resolve a loader configuration derived from untrusted input
        without it, as this is a remote-code-execution vector.

    Example:
        ```pycon
        >>> from coola.io import resolve_loader
        >>> loader = resolve_loader({"_target_": "coola.io.JsonLoader"})
        >>> loader
        JsonLoader(encoding=None)

        ```
    """
    return cast(
        "BaseLoader[T]", resolve_object(loader, BaseLoader, allowed_prefixes=allowed_prefixes)
    )


def resolve_saver(
    saver: BaseSaver[T] | dict[Any, Any], *, allowed_prefixes: Sequence[str] | None = None
) -> BaseSaver[T]:
    r"""Set up a data saver.

    The data saver is instantiated from its configuration by using the
    ``BaseSaver`` factory function.

    Args:
        saver: The data saver or its configuration.
        allowed_prefixes: An optional allow-list of module prefixes that
            the ``"_target_"`` must match, e.g. ``["coola.io"]``. If
            ``None`` (default), any ``"_target_"`` is allowed.

    Returns:
        The instantiated data saver.

    Security:
        When ``saver`` is a configuration :class:`dict`, its
        ``"_target_"`` value is imported and instantiated with no
        allowlist by default (see :func:`coola.factory.factory`). Only
        call this function with a configuration that comes from a
        trusted source, or restrict it with ``allowed_prefixes``; never
        resolve a saver configuration derived from untrusted input
        without it, as this is a remote-code-execution vector.

    Example:
        ```pycon
        >>> from coola.io import resolve_saver
        >>> saver = resolve_saver({"_target_": "coola.io.JsonSaver"})
        >>> saver
        JsonSaver(encoding='utf-8')

        ```
    """
    return cast("BaseSaver[T]", resolve_object(saver, BaseSaver, allowed_prefixes=allowed_prefixes))
