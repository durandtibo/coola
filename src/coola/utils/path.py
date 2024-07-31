r"""Contain path utility functions."""

from __future__ import annotations

__all__ = ["sanitize_path", "working_directory"]

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from collections.abc import Generator


def sanitize_path(path: Path | str) -> Path:
    r"""Sanitize the given path.

    Args:
        path: The path to sanitize. The path can be a string or a
            ``pathlib.Path`` object.

    Returns:
        The sanitized path as a ``pathlib.Path`` object.

    Example usage:

    ```pycon

    >>> from pathlib import Path
    >>> from coola.utils.path import sanitize_path
    >>> sanitize_path("something")
    PosixPath('.../something')
    >>> sanitize_path("")
    PosixPath('...')
    >>> sanitize_path(Path("something"))
    PosixPath('.../something')
    >>> sanitize_path(Path("something/./../"))
    PosixPath('...')

    ```
    """
    if isinstance(path, str):
        # use urlparse to parse file URI
        # source: https://stackoverflow.com/a/15048213
        path = Path(unquote(urlparse(path).path))
    return path.expanduser().resolve()


@contextlib.contextmanager
def working_directory(path: Path) -> Generator[None]:
    r"""Context manager to change the working directory to the given
    path, and then changes it back to its previous value on exit.

    source: https://gist.github.com/nottrobin/3d675653244f8814838a

    Args:
        path: The path to the temporary working directory.

    Example usage:

    ```pycon

    >>> from coola.utils.path import working_directory
    >>> with working_directory(Path("src")):
    ...     x = 1
    ...

    ```
    """
    path = sanitize_path(path)
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
