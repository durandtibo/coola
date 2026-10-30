r"""Contain file size utility functions."""

from __future__ import annotations

__all__ = ["get_file_size"]

import os
import stat
from pathlib import Path


def get_file_size(path: Path) -> int:
    r"""Return the size of a file or directory, in bytes.

    If ``path`` is a directory, returns the total size of all files
    it contains (recursively). Symlinks inside the directory are
    neither followed nor counted, and files deleted while the directory
    is being scanned are skipped.

    Args:
        path: The path to the file or directory.

    Returns:
        The size in bytes.

    Raises:
        FileNotFoundError: if ``path`` does not exist.

    Example:
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
    total = 0
    for dirpath, _, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            try:
                info = (Path(dirpath) / name).lstat()
            except OSError:  # the file vanished during the scan
                continue
            if stat.S_ISREG(info.st_mode):  # a regular file, not a symlink
                total += info.st_size
    return total
