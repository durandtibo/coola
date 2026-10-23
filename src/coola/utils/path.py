r"""Contain path utility functions."""

from __future__ import annotations

__all__ = ["find_root_package_parent", "sanitize_path", "to_path", "working_directory"]

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from collections.abc import Generator


def find_root_package_parent(start_path: str | os.PathLike[str]) -> Path:
    r"""Given a file or directory path, walk upward through directories
    that contain ``__init__.py`` (i.e., are part of a package) and
    return the parent directory of the outermost (root) package.

    Args:
        start_path: The path to a file or directory inside the
            package. Can be any path-like object, i.e. a string or
            an object implementing the ``os.PathLike`` protocol such
            as ``pathlib.Path``.

    Returns:
        The absolute path to the parent directory of the root
            package. If ``start_path`` itself is not part of a
            package (no ``__init__.py`` in its directory), its own
            directory is returned.

    Example:
        ```pycon
        >>> from coola.utils.path import find_root_package_parent
        >>> find_root_package_parent("something")  # doctest: +SKIP
        PosixPath('.../my_project')

        ```

        Given the following directory layout::

            my_project/
                pkg/
                    __init__.py
                    sub/
                        __init__.py
                        module.py

        ``find_root_package_parent(".../pkg/sub/module.py")`` returns
        ``Path(".../my_project")``.
    """
    path = sanitize_path(start_path)

    # If it's a file (or a path that does not exist), start from its containing directory
    if not path.is_dir():
        path = path.parent

    current = path
    while True:
        parent = current.parent
        init_file = current / "__init__.py"

        if not init_file.is_file():
            # 'current' is not a package -> the last package's parent was the root's parent
            return current

        if parent == current:
            # Reached filesystem root without leaving a package
            return current

        current = parent


def to_path(path: str | os.PathLike[str]) -> Path:
    r"""Convert a ``str`` or a path-like object to a ``pathlib.Path``.

    Unlike ``sanitize_path``, the path is not expanded or resolved: it
    is only converted, so a relative path stays relative.

    Args:
        path: The path to convert. A ``str`` or an object implementing
            the ``os.PathLike`` protocol.

    Returns:
        The path as a ``pathlib.Path`` object. A ``Path`` is returned
            as is (the same object).

    Raises:
        TypeError: if ``path`` is neither a ``str`` nor a path-like
            object.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from coola.utils.path import to_path
        >>> to_path("data/a.txt")
        PosixPath('data/a.txt')
        >>> to_path(Path("data/a.txt"))
        PosixPath('data/a.txt')

        ```
    """
    if isinstance(path, Path):
        return path
    return Path(os.fspath(path))


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

    Raises:
        ValueError: if a ``file://`` URI has a host other than
            ``localhost`` (for example ``file://server/share/x``), or
            if the home directory of ``~user`` cannot be found.

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
    if path.startswith("file://"):
        uri = urlparse(path)
        if uri.netloc not in {"", "localhost"}:
            msg = f"file URI with a host is not supported, got {path!r}"
            raise ValueError(msg)
        path = Path(unquote(uri.path))
    else:
        path = Path(path)
    try:
        return path.expanduser().resolve()
    except RuntimeError as error:
        msg = f"Could not expand the home directory of {str(path)!r}"
        raise ValueError(msg) from error


@contextlib.contextmanager
def working_directory(path: str | os.PathLike[str]) -> Generator[None]:
    r"""Context manager to change the working directory to the given
    path, and then changes it back to its previous value on exit.

    source: https://gist.github.com/nottrobin/3d675653244f8814838a

    Note:
        The working directory is process-wide, so this context manager
        is **not thread-safe**: other threads see the new directory.

    Args:
        path: The path to the temporary working directory. This can
            be any path-like object, i.e. a string or an object
            implementing the ``os.PathLike`` protocol such as
            ``pathlib.Path``. See ``sanitize_path`` for details on
            how the path is resolved.

    Example:
        ```pycon
        >>> from coola.utils.path import working_directory
        >>> with working_directory("src"):
        ...     x = 1
        ...

        ```
    """
    path = sanitize_path(path)
    with contextlib.chdir(path):
        yield
