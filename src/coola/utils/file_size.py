r"""Contain file size utility functions."""

from __future__ import annotations

__all__ = ["get_file_size"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def get_file_size(path: Path) -> int:
    r"""Return the size of a file or directory, in bytes.

    If ``path`` is a directory, returns the total size of all files
    it contains (recursively). Symlinks are not followed.

    Args:
        path: The path to the file or directory.

    Returns:
        The size in bytes.

    Raises:
        FileNotFoundError: if ``path`` does not exist.

    Example usage:

    ```pycon
    >>> from pathlib import Path
    >>> from coola.utils.file_size import get_file_size
    >>> get_file_size(Path(__file__))  # doctest: +SKIP
    512

    ```
    """
    if not path.exists():
        msg = f"path does not exist: {path}"
        raise FileNotFoundError(msg)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
