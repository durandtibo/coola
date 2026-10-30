r"""Contain text-based data loaders and savers."""

from __future__ import annotations

__all__ = ["TextLoader", "TextSaver", "load_text", "save_text"]

from typing import TYPE_CHECKING, Any

from coola.io.base import BaseFileSaver, BaseLoader

if TYPE_CHECKING:
    import os
    from pathlib import Path

DEFAULT_ENCODING = "utf-8"


class TextLoader(BaseLoader[str]):
    r"""Implement a data loader to load data from a text file.

    Args:
        encoding: The file encoding to use when reading the file.
            Defaults to ``"utf-8"``.

    Note:
        The line endings are returned as they are in the file
        (no universal newline translation), so ``\\r\\n`` and ``\\r`` are
        preserved.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_text, TextLoader
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.txt")
        ...     save_text("hello", path)
        ...     data = TextLoader().load(path)
        ...     data
        ...
        'hello'

        ```
    """

    def __init__(self, encoding: str = DEFAULT_ENCODING) -> None:
        self._encoding = encoding

    def _display_fields(self) -> dict[str, Any]:
        return {"encoding": self._encoding}

    def _load(self, path: Path) -> str:
        with path.open(encoding=self._encoding, newline="") as file:
            return file.read()


class TextSaver(BaseFileSaver[str]):
    r"""Implement a file saver to save data to a text file.

    Args:
        encoding: The file encoding to use when writing the file.
            Defaults to ``"utf-8"``.

    Note:
        The line endings are written as they are in the string (no
        translation of ``\\n``, even on Windows).

    Raises:
        TypeError: if the data to save is not a ``str``. Call ``str``
            on the data to save its text representation.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import TextSaver, TextLoader
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.txt")
        ...     TextSaver().save("hello", path)
        ...     data = TextLoader().load(path)
        ...     data
        ...
        'hello'

        ```
    """

    def __init__(self, encoding: str = DEFAULT_ENCODING) -> None:
        self._encoding = encoding

    def _display_fields(self) -> dict[str, Any]:
        return {"encoding": self._encoding}

    def _save_file(self, to_save: str, path: Path) -> None:
        r"""Save a text into a file.

        Args:
            to_save: The text to save.
            path: The path of the file.

        Raises:
            TypeError: If ``to_save`` is not a ``str``.
        """
        if not isinstance(to_save, str):
            msg = (
                f"{type(self).__name__} expects a str, got {type(to_save).__name__}; "
                "call str() on the data to save its text representation"
            )
            raise TypeError(msg)
        with path.open(mode="w", encoding=self._encoding, newline="") as file:
            file.write(to_save)


def load_text(path: str | os.PathLike[str], encoding: str = DEFAULT_ENCODING) -> str:
    r"""Load the data from a given text file.

    Args:
        path: The path to the text file. A ``str`` or any path-like object is converted to a ``Path``.
        encoding: The file encoding to use when reading the file.
            Defaults to ``"utf-8"``.

    Returns:
        The text content of the file as a string.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_text, load_text
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.txt")
        ...     save_text("hello", path)
        ...     data = load_text(path)
        ...     data
        ...
        'hello'

        ```
    """
    return TextLoader(encoding=encoding).load(path)


def save_text(
    to_save: str,
    path: str | os.PathLike[str],
    *,
    encoding: str = DEFAULT_ENCODING,
    exist_ok: bool = False,
    fsync: bool = False,
) -> None:
    r"""Save the given data to a text file.

    Args:
        to_save: The string to write to the text file.
        path: The path where to write the text file. A ``str`` or any path-like object is converted to a ``Path``.
        encoding: The file encoding to use when writing the file.
            Defaults to ``"utf-8"``.
        exist_ok: If ``exist_ok`` is ``False`` (the default),
            ``FileExistsError`` is raised if the target file
            already exists. If ``exist_ok`` is ``True``,
            ``FileExistsError`` will not be raised unless the
            given path already exists in the file system and is
            not a file.
        fsync: If ``True``, flush the data to the disk before committing
            the file, so it survives a power failure or a system crash.
            It is slower. Defaults to ``False``: the file is replaced
            atomically, but its content may be lost if the machine
            crashes right after the call.

    Raises:
        FileExistsError: if the file already exists.
        TypeError: if the data to save is not a ``str``. Call ``str``
            on the data to save its text representation.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_text, load_text
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.txt")
        ...     save_text("hello", path)
        ...     data = load_text(path)
        ...     data
        ...
        'hello'

        ```
    """
    TextSaver(encoding=encoding).save(to_save, path, exist_ok=exist_ok, fsync=fsync)
