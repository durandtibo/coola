r"""Contain text-based data loaders and savers."""

from __future__ import annotations

__all__ = ["TextLoader", "TextSaver", "load_text", "save_text"]

from pathlib import Path
from typing import Any, TypeVar

from coola.display import InlineDisplayMixin
from coola.io.base import BaseFileSaver, BaseLoader

T = TypeVar("T")

DEFAULT_ENCODING = "utf-8"


class TextLoader(InlineDisplayMixin, BaseLoader[str]):
    r"""Implement a data loader to load data from a text file.

    Args:
        encoding: The file encoding to use when reading the file.
            Defaults to ``"utf-8"``.

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

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"encoding": self._encoding}

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        if type(other) is not type(self):
            return False
        return self._encoding == other._encoding

    def load(self, path: Path) -> str:
        with Path.open(path, encoding=self._encoding) as file:
            return file.read()


class TextSaver(InlineDisplayMixin, BaseFileSaver[str]):
    r"""Implement a file saver to save data to a text file.

    Args:
        encoding: The file encoding to use when writing the file.
            Defaults to ``"utf-8"``.

    Note:
        If the data to save is not a string, it is converted to
            a string before being saved by using ``str``.

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

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"encoding": self._encoding}

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        if type(other) is not type(self):
            return False
        return self._encoding == other._encoding

    def _save_file(self, to_save: str, path: Path) -> None:
        with Path.open(path, mode="w", encoding=self._encoding) as file:
            file.write(str(to_save))


def load_text(path: Path, encoding: str = DEFAULT_ENCODING) -> str:
    r"""Load the data from a given text file.

    Args:
        path: The path to the text file.
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
    to_save: Any,
    path: Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    exist_ok: bool = False,
) -> None:
    r"""Save the given data to a text file.

    Args:
        to_save: The data to write to the text file.
        path: The path where to write the text file.
        encoding: The file encoding to use when writing the file.
            Defaults to ``"utf-8"``.
        exist_ok: If ``exist_ok`` is ``False`` (the default),
            ``FileExistsError`` is raised if the target file
            already exists. If ``exist_ok`` is ``True``,
            ``FileExistsError`` will not be raised unless the
            given path already exists in the file system and is
            not a file.

    Raises:
        FileExistsError: if the file already exists.

    Note:
        If the data to save is not a string, it is converted to
            a string before being saved by using ``str``.

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
    TextSaver(encoding=encoding).save(to_save, path, exist_ok=exist_ok)
