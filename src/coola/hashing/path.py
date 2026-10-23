r"""Provide deterministic hashing functions for paths."""

from __future__ import annotations

__all__ = ["PathHasher", "ResolvedPathHasher", "hash_path"]

from pathlib import Path, PurePath
from typing import TYPE_CHECKING

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class PathHasher(InlineDisplayMixin, BaseHasher[PurePath]):
    r"""Hasher for ``pathlib.PurePath`` and ``pathlib.Path`` objects.

    This hasher delegates to ``hash_path``, which hashes the POSIX
    string representation of the path as written. It never touches the
    filesystem and does not depend on the current working directory, so
    hashes are portable across machines. A relative path and the
    absolute path of the same location therefore hash differently; use
    ``ResolvedPathHasher`` to hash the real location instead.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from coola.hashing import PathHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = PathHasher()
        >>> hasher
        PathHasher()
        >>> len(hasher.hash(Path("/tmp/file.txt"), registry=registry))
        64

        ```
    """

    def hash(
        self,
        data: PurePath,
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        r"""Compute a deterministic hash of a path.

        Args:
            data: The path to hash.
            registry: The hasher registry. Unused by this hasher since
                the path is hashed directly with no need to dispatch to
                another hasher for nested data; accepted only to
                satisfy the common ``BaseHasher`` interface.
            length: The desired length of the returned hex string. See
                ``hash_path`` for constraints. Defaults to 64.
            ignore_unhashable: Unused by this hasher; accepted only to
                satisfy the common ``BaseHasher`` interface.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            ValueError: If ``length`` is not an even number between 2
                and 128.
        """
        return hash_path(data, length=length)


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
        >>> from coola.hashing import ResolvedPathHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = ResolvedPathHasher()
        >>> hasher
        ResolvedPathHasher()
        >>> len(hasher.hash(Path("file.txt"), registry=registry))
        64

        ```
    """

    def hash(
        self,
        data: Path,
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        r"""Compute a deterministic hash of the resolved path.

        Args:
            data: The path to hash.
            registry: The hasher registry. Unused; accepted only to
                satisfy the common ``BaseHasher`` interface.
            length: The desired length of the returned hex string. See
                ``hash_path`` for constraints. Defaults to 64.
            ignore_unhashable: Unused; accepted only to satisfy the
                common ``BaseHasher`` interface.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            ValueError: If ``length`` is not an even number between 2
                and 128.
        """
        return hash_path(data, length=length, resolve=True)


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
    resolved = path
    if resolve and isinstance(path, Path):
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
    return hash_tagged("Path", resolved.as_posix(), length=length)
