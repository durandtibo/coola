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
import hashlib
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from coola.display import InlineDisplayMixin
from coola.equality import objects_are_equal
from coola.factory import is_object_config, resolve_object
from coola.utils.introspection import is_same_type
from coola.utils.path import add_uuid_suffix, sanitize_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

T = TypeVar("T")

logger: logging.Logger = logging.getLogger(__name__)

# Longest target name (in bytes) for which the temporary name are
# derived from the name. Most filesystems limit a name to 255 bytes, and these
# names add a prefix, a uuid or a suffix to it.
_MAX_DERIVED_NAME_BYTES = 100


def _short_name(name: str) -> str:
    r"""Return ``name``, or a short stable name if ``name`` is too long.

    The short name is a hash of ``name`` followed by its extension, so
    the names derived from it (temporary file) stay below the filesystem
    limit for any legal target name.

    Args:
        name: The name of the file.

    Returns:
        ``name``, or a hash of ``name`` followed by its extension if
            ``name`` is too long.
    """
    if len(name.encode()) <= _MAX_DERIVED_NAME_BYTES:
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:16]
    suffix = PurePath(name).suffix
    if len(suffix.encode()) > 16:
        suffix = ""
    return f"{digest}{suffix}"


def _make_parent_dir(path: Path) -> None:
    r"""Create the parent directory of ``path``.

    Args:
        path: The path of the file.

    Raises:
        NotADirectoryError: if the parent exists and is not a directory.
    """
    try:
        path.parent.mkdir(exist_ok=True, parents=True)
    except FileExistsError as error:
        if path.parent.is_dir():
            raise
        msg = f"the parent of path ({path}) exists and is not a directory"
        raise NotADirectoryError(msg) from error


def _tmp_path(path: Path) -> Path:
    r"""Return the path of the temporary file used to write ``path``.

    The name starts with a dot, so the file is hidden and a reader that
    lists the directory does not take it for the final file while it is
    being written. The extension is kept because some writers choose the
    format or add an extension from it.

    Args:
        path: The final path of the file.

    Returns:
        The path of the hidden temporary file.
    """
    return add_uuid_suffix(path.with_name(f".{_short_name(path.name)}"))


def _fsync_file(path: Path) -> None:
    r"""Flush the content of a file to the disk.

    Args:
        path: The path of the file.
    """
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    r"""Flush the entries of a directory to the disk, so a rename or a
    link made in it survives a power failure.

    It is a best effort: some platforms and filesystems do not support
    opening or syncing a directory.

    Args:
        path: The path of the directory.
    """
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class BaseLoader(InlineDisplayMixin, ABC, Generic[T]):
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

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two objects are equal or not.

        Two objects are equal if they have the same type and the same
        configuration, i.e. the fields returned by ``_display_fields``
        (the fields shown by ``repr``) are equal. A format does not
        override this method.

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
        if not is_same_type(other, self):
            return False
        return objects_are_equal(
            self._display_fields(), other._display_fields(), equal_nan=equal_nan
        )

    def load(self, path: str | os.PathLike[str]) -> T:
        r"""Load the data from the given path.

        The path is converted to a ``Path`` (``~`` and ``file://``
        URIs are expanded) before ``_load`` is called, so a format
        always receives a ``Path``.

        Args:
            path: The path with the data to load. A ``str`` or any path-like object is converted to a ``Path``.

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
        return self._load(sanitize_path(path, resolve=False))

    @abstractmethod
    def _load(self, path: Path) -> T:
        r"""Load the data from the given path.

        Args:
            path: The path with the data to load.

        Returns:
            The data.
        """


class BaseSaver(InlineDisplayMixin, ABC, Generic[T]):
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

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two objects are equal or not.

        Two objects are equal if they have the same type and the same
        configuration, i.e. the fields returned by ``_display_fields``
        (the fields shown by ``repr``) are equal. A format does not
        override this method.

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
        if not is_same_type(other, self):
            return False
        return objects_are_equal(
            self._display_fields(), other._display_fields(), equal_nan=equal_nan
        )

    @abstractmethod
    def save(self, to_save: T, path: str | os.PathLike[str], *, exist_ok: bool = False) -> None:
        r"""Save the data into the given path.

        Args:
            to_save: The data to save. The data should be compatible
                with the saving engine.
            path: The path where to save the data. A ``str`` or any path-like object is converted to a ``Path``.
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

    def save(
        self,
        to_save: T,
        path: str | os.PathLike[str],
        *,
        exist_ok: bool = False,
        fsync: bool = False,
    ) -> None:
        r"""Save the data into the given path.

        Args:
            to_save: The data to save. The data should be compatible
                with the saving engine.
            path: The path where to save the data. A ``str`` or any path-like object is converted to a ``Path``.
            exist_ok: If ``exist_ok`` is ``False`` (the default),
                ``FileExistsError`` is raised if the target file
                already exists. If ``exist_ok`` is ``True``,
                ``FileExistsError`` will not be raised unless the
                given path already exists in the file system and is
                not a file.

                Note: the ``exist_ok=False`` guarantee holds against a
                concurrent creation of ``path`` between the initial
                existence check and the commit step (the commit uses
                ``os.link`` which atomically fails with
                ``FileExistsError`` in that case).

                Note: concurrent ``save`` calls targeting the same
                ``path``, from this process or from other processes,
                never produce a mixed file: each writes its own
                hidden temporary file (``.<name>-<uuid><ext>``) and
                commits it atomically, so the file is the complete
                content of one of them. With ``exist_ok=True`` which
                one is unspecified (the last commit wins); with
                ``exist_ok=False`` exactly one wins and the others raise
                ``FileExistsError``. ``save`` takes no lock: a caller
                that needs mutual exclusion around a read-modify-write
                of one file uses its own lock.

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
            to a hidden temporary file next to ``path``. A process killed
            between the write and the commit leaves this file; it is
            safe to delete. The saved file is committed with a hard
            link, or with a rename if the filesystem does not support
            hard links (e.g. FAT/exFAT, and some network and container
            volumes); in that case two processes that create the same
            file at the same instant can both succeed (the last one
            wins, no file is torn).

        Raises:
            FileExistsError: if the file already exists and ``exist_ok`` is ``False``.
            IsADirectoryError: If ``path`` is a directory.
            NotADirectoryError: If the parent of ``path`` is a file.

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
        path = sanitize_path(path, resolve=False)
        if path.is_dir():
            msg = f"path ({path}) is a directory"
            raise IsADirectoryError(msg)
        if path.is_file() and not exist_ok:
            msg = f"path ({path}) already exists. Use `exist_ok=True` to overwrite the file"
            raise FileExistsError(msg)
        if exist_ok and path.is_symlink():
            # Overwrite the file that the link points to, not the link itself.
            path = path.resolve()
        _make_parent_dir(path)

        # Save to tmp, then commit by moving the file in case the job
        # gets interrupted while writing the file.
        tmp_path = _tmp_path(path)
        try:
            self._save_file(to_save, tmp_path)
            if fsync:
                _fsync_file(tmp_path)
            if exist_ok:
                if path.is_file():
                    # Keep the permissions of the file that is overwritten.
                    with contextlib.suppress(OSError):
                        shutil.copymode(path, tmp_path)
                tmp_path.replace(path)
            else:
                # ``os.link`` + unlink is used instead of a plain rename so the
                # ``exist_ok=False`` guarantee also holds if ``path`` was created
                # concurrently between the check above and this commit step:
                # ``link`` atomically fails with ``FileExistsError`` in that case.
                self._link(tmp_path, path)
        finally:
            # Removes ``tmp_path`` whether the commit succeeded (with ``os.link``
            # the data now lives at ``path``), failed, or ``_save_file`` failed.
            # An error of this cleanup is suppressed, so it cannot mask the
            # exception being handled.
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
        if fsync:
            _fsync_directory(path.parent)

    @staticmethod
    def _link(tmp_path: Path, path: Path) -> None:
        r"""Commit ``tmp_path`` to ``path`` without overwriting ``path``.

        ``os.link`` is atomic: it fails with ``FileExistsError`` if
        ``path`` exists. When the filesystem does not support hard links,
        it falls back on a check followed by a rename. Two processes that
        create ``path`` at the same instant can then both succeed (the
        last one wins).

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
            path: The path where to save the data. A ``str`` or any path-like object is converted to a ``Path``.
        """


def is_loader_config(
    config: dict[Any, Any],
    *,
    allowed_prefixes: Sequence[str] | None = None,
    import_target: bool = False,
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
        import_target: If ``True``, the module of ``"_target_"`` is
            imported if it is not imported yet, even without
            ``allowed_prefixes``. By default, a ``"_target_"`` whose
            module is not imported yet and that is not restricted by
            ``allowed_prefixes`` is not a configuration for a
            ``BaseLoader`` (see ``coola.factory.is_object_config``).

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseLoader`` object.

    Security:
        The module of the ``"_target_"`` is imported to check its class
        only with ``allowed_prefixes`` or ``import_target=True``: see
        ``coola.factory.is_object_config``.

    Example:
        ```pycon
        >>> from coola.io import is_loader_config
        >>> is_loader_config({"_target_": "coola.io.JsonLoader"})
        True

        ```
    """
    return is_object_config(
        config, BaseLoader, allowed_prefixes=allowed_prefixes, import_target=import_target
    )


def is_saver_config(
    config: dict[Any, Any],
    *,
    allowed_prefixes: Sequence[str] | None = None,
    import_target: bool = False,
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
        import_target: If ``True``, the module of ``"_target_"`` is
            imported if it is not imported yet, even without
            ``allowed_prefixes``. By default, a ``"_target_"`` whose
            module is not imported yet and that is not restricted by
            ``allowed_prefixes`` is not a configuration for a
            ``BaseSaver`` (see ``coola.factory.is_object_config``).

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseSaver`` object.

    Security:
        The module of the ``"_target_"`` is imported to check its class
        only with ``allowed_prefixes`` or ``import_target=True``: see
        ``coola.factory.is_object_config``.

    Example:
        ```pycon
        >>> from coola.io import is_saver_config
        >>> is_saver_config({"_target_": "coola.io.JsonSaver"})
        True

        ```
    """
    return is_object_config(
        config, BaseSaver, allowed_prefixes=allowed_prefixes, import_target=import_target
    )


def resolve_loader(
    loader: BaseLoader[T] | dict[Any, Any],
    *,
    allowed_prefixes: Sequence[str] | None = None,
    strict: bool = False,
) -> BaseLoader[T]:
    r"""Set up a data loader.

    The data loader is instantiated from its configuration by using the
    ``BaseLoader`` factory function.

    Args:
        loader: The data loader or its configuration.
        allowed_prefixes: An optional allow-list of module prefixes that
            the ``"_target_"`` must match, e.g. ``["coola.io"]``. If
            ``None`` (default), any ``"_target_"`` is allowed.
        strict: If ``True``, a ``"_target_"`` that is not a class and
            has no usable return type hint is refused before it is
            called (see ``resolve_object``).

    Returns:
        The instantiated data loader.

    Security:
        When ``loader`` is a configuration ``dict``, its
        ``"_target_"`` value is imported and instantiated with no
        allowlist by default (see ``coola.factory.factory``). Only
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
        "BaseLoader[T]",
        resolve_object(loader, BaseLoader, allowed_prefixes=allowed_prefixes, strict=strict),
    )


def resolve_saver(
    saver: BaseSaver[T] | dict[Any, Any],
    *,
    allowed_prefixes: Sequence[str] | None = None,
    strict: bool = False,
) -> BaseSaver[T]:
    r"""Set up a data saver.

    The data saver is instantiated from its configuration by using the
    ``BaseSaver`` factory function.

    Args:
        saver: The data saver or its configuration.
        allowed_prefixes: An optional allow-list of module prefixes that
            the ``"_target_"`` must match, e.g. ``["coola.io"]``. If
            ``None`` (default), any ``"_target_"`` is allowed.
        strict: If ``True``, a ``"_target_"`` that is not a class and
            has no usable return type hint is refused before it is
            called (see ``resolve_object``).

    Returns:
        The instantiated data saver.

    Security:
        When ``saver`` is a configuration ``dict``, its
        ``"_target_"`` value is imported and instantiated with no
        allowlist by default (see ``coola.factory.factory``). Only
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
    return cast(
        "BaseSaver[T]",
        resolve_object(saver, BaseSaver, allowed_prefixes=allowed_prefixes, strict=strict),
    )
