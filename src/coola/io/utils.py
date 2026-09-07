r"""Contain I/O utility functions."""

from __future__ import annotations

__all__ = ["add_uuid_suffix"]

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


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
            UUID appended to the stem.

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
    extension = "".join(path.suffixes)[1:]
    if extension:
        extension = "." + extension
        stem = path.name[: -len(extension)]
    else:
        stem = path.name
    return path.with_name(f"{stem}-{h}{extension}")
