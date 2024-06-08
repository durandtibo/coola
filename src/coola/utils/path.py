r"""Contain path utility functions."""

from __future__ import annotations

__all__ = ["sanitize_path"]

from pathlib import Path
from urllib.parse import unquote, urlparse


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
