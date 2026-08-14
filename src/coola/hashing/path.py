r"""Provide deterministic hashing functions for paths."""

from __future__ import annotations

__all__ = ["PathHasher", "hash_path"]

from pathlib import Path
from typing import TYPE_CHECKING

from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_string

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class PathHasher(BaseHasher[Path]):
    r"""Hasher for ``pathlib.Path`` objects.

    This hasher delegates to ``hash_path``, which resolves the path
    (following symlinks and collapsing ``.``/``..`` segments) before
    hashing its POSIX string representation, so that two differently
    written but logically equivalent paths hash the same. If resolution
    fails, the unresolved path is hashed instead.

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

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def hash(
        self,
        data: Path,
        registry: HasherRegistry,  # noqa: ARG002
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


def hash_path(path: Path, length: int = 64) -> str:
    """Compute a stable hash string for a Path, based on its POSIX
    string representation.

    Attempts to resolve the path first (following symlinks, collapsing
    ``.``/``..`` segments, making it absolute), so that two differently
    written but logically equivalent paths hash the same. If resolution
    fails (e.g. permission error, OS-level issue), falls back to
    hashing the unresolved path as given.

    Uses ``as_posix()`` rather than ``str(path)`` so the hash is
    consistent across platforms (POSIX vs Windows separators).

    Args:
        path: The path to hash.
        length: The desired length of the returned hex string. Must be
            an even number between 2 and 128 inclusive. Defaults to 64.

    Returns:
        A lowercase hexadecimal string of exactly ``length``
        characters.

    Raises:
        ValueError: If ``length`` is not an even number between 2 and
            128.
    """
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    return hash_string(resolved.as_posix(), length=length)
