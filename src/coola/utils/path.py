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


def sanitize_path(path: str | os.PathLike[str]) -> Path:
    r"""Sanitize the given path.

    Args:
        path: The path to sanitize. This can be any path-like object,
            i.e. a string or an object implementing the
            ``os.PathLike`` protocol such as ``pathlib.Path``.
            ``file://`` URIs are recognized and decoded, but only if
            given as a ``str`` (or a ``PathLike`` whose
            ``__fspath__`` returns the raw URI string). Wrapping a
            URI in ``pathlib.Path`` first is not supported: ``Path``
            collapses ``"file:///a"`` to ``"file:/a"`` before this
            function ever sees it, so the URI can no longer be
            detected and it is treated as a plain relative path.

    Returns:
        The sanitized path as a ``pathlib.Path`` object.

    Example:
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
    if not isinstance(path, str):
        path = os.fspath(path)
    # use urlparse to parse file URI
    # source: https://stackoverflow.com/a/15048213
    path = Path(unquote(urlparse(path).path)) if path.startswith("file://") else Path(path)
    return path.expanduser().resolve()


@contextlib.contextmanager
def working_directory(path: Path) -> Generator[None]:
    r"""Context manager to change the working directory to the given
    path, and then changes it back to its previous value on exit.

    source: https://gist.github.com/nottrobin/3d675653244f8814838a

    Args:
        path: The path to the temporary working directory.

    Example:
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
