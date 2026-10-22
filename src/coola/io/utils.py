r"""Contain I/O utility functions."""

from __future__ import annotations

__all__ = ["add_uuid_suffix"]

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


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
        >>> from coola.io import add_uuid_suffix
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
