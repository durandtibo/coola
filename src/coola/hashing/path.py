r"""Provide deterministic hashing functions for paths."""

from __future__ import annotations

__all__ = ["PathHasher", "ResolvedPathHasher", "hash_path"]

from contextlib import suppress
from pathlib import Path, PurePath

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, tagged
from coola.hashing.string import hash_tagged


class PathHasher(InlineDisplayMixin, BaseHasher[PurePath]):
    r"""Hasher for ``pathlib.PurePath`` and ``pathlib.Path`` objects.

    This hasher describes the POSIX string representation of the path
    as written. It never touches the
    filesystem and does not depend on the current working directory, so
    hashes are portable across machines. A relative path and the
    absolute path of the same location therefore hash differently; use
    ``ResolvedPathHasher`` to hash the real location instead.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from coola.hashing import PathHasher
        >>> hasher = PathHasher()
        >>> hasher
        PathHasher()
        >>> hasher.encode(Path("/tmp/file.txt"))
        b'Path\x00/tmp/file.txt'

        ```
    """

    def encode(self, data: PurePath) -> Encoding:
        r"""Describe a path.

        Args:
            data: The path to describe.

        Returns:
            The leaf ``tagged("Path", posix_form).
        """
        return tagged("Path", _posix(data, resolve=False))


class ResolvedPathHasher(InlineDisplayMixin, BaseHasher[Path]):
    r"""Hasher for ``pathlib.Path`` objects that hashes the resolved
    location.

    The path is resolved (following symlinks, collapsing ``.``/``..``
    segments, making it absolute) before hashing, so that two
    differently written but logically equivalent paths hash the same.
    The result depends on the filesystem and on the current working
    directory, so it is not portable across machines. If resolution
    fails, the unresolved path is hashed instead.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from coola.hashing import ResolvedPathHasher
        >>> hasher = ResolvedPathHasher()
        >>> hasher
        ResolvedPathHasher()
        >>> hasher.encode(Path("file.txt")).startswith(b"Path")
        True

        ```
    """

    def encode(self, data: Path) -> Encoding:
        r"""Describe a path.

        Args:
            data: The path to describe.

        Returns:
            The leaf ``tagged("Path", posix_form) of the resolved path.
        """
        return tagged("Path", _posix(data, resolve=True))


def hash_path(path: PurePath, length: int = 64, resolve: bool = False) -> str:
    """Compute a stable hash string for a path, based on its POSIX
    string representation.

    By default the path is hashed as written, without any filesystem
    access. With ``resolve=True`` the path is resolved first (following
    symlinks, collapsing ``.``/``..`` segments, making it absolute), so
    that two differently written but logically equivalent paths hash
    the same. If resolution fails (e.g. permission error, OS-level
    issue), falls back to hashing the unresolved path as given.

    Uses ``as_posix()`` rather than ``str(path)`` so the hash is
    consistent across platforms (POSIX vs Windows separators).

    Args:
        path: The path to hash.
        length: The desired length of the returned hex string. Must be
            an even number between 2 and 128 inclusive. Defaults to 64.
        resolve: If ``True``, resolve the path before hashing. This
            makes the hash depend on the filesystem and on the current
            working directory. Only supported for ``pathlib.Path``.
            Defaults to ``False``.

    Returns:
        A lowercase hexadecimal string of exactly ``length``
        characters.

    Raises:
        ValueError: If ``length`` is not an even number between 2 and
            128.
    """
    return hash_tagged("Path", _posix(path, resolve=resolve), length=length)


def _posix(path: PurePath, *, resolve: bool = False) -> str:
    r"""Return the POSIX form of a path, resolved if requested.

    If the resolution fails (permission error, OS-level issue), the
    path is used as given.
    """
    if resolve and isinstance(path, Path):
        with suppress(OSError):
            path = path.resolve()
    return path.as_posix()
