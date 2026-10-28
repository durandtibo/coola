r"""Contain path utility functions."""

from __future__ import annotations

__all__ = ["add_uuid_suffix", "find_root_package_parent", "sanitize_path"]

import os
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

# The compound extensions kept together; any other dotted part of the name
# (e.g. a version number in ``v1.2.model``) belongs to the stem.
_COMPOUND_EXTENSIONS = frozenset({".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tar.lz4"})


def add_uuid_suffix(path: Path) -> Path:
    r"""Return a new path with a unique UUID suffix added to the file
    name.

    This is typically used to derive a unique staging path from a
    target path, e.g. to write to a temporary file before atomically
    renaming it to the target path.

    Args:
        path: The input path.

    Returns:
        A new path with the same parent and extension, and a unique
            UUID appended to the stem. The extension is the last
            suffix, or a compound extension such as ``.tar.gz``.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.utils.path import add_uuid_suffix
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = add_uuid_suffix(Path(tmpdir).joinpath("data.pt"))
        ...     path
        ...
        PosixPath('/.../data-....pt')

        ```
    """
    h = uuid.uuid4().hex
    compound = "".join(path.suffixes[-2:])
    extension = compound if compound.lower() in _COMPOUND_EXTENSIONS else path.suffix
    stem = path.name.removesuffix(extension)
    return path.with_name(f"{stem}-{h}{extension}")


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


def sanitize_path(path: str | os.PathLike[str], *, resolve: bool = True) -> Path:
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
        resolve: If ``True`` (default), the path is made absolute and
            the symbolic links are resolved. If ``False``, the path is
            only converted to a ``Path`` (``~`` and ``file://`` URIs
            are still expanded): a relative path stays relative and a
            symbolic link is kept.

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
        >>> sanitize_path("something", resolve=False)
        PosixPath('something')

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
        path = path.expanduser()
        return path.resolve() if resolve else path
    except RuntimeError as error:
        msg = f"Could not expand the home directory of {str(path)!r}"
        raise ValueError(msg) from error
