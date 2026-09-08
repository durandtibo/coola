r"""Contain JSON-based data loaders and savers."""

from __future__ import annotations

__all__ = ["JsonLoader", "JsonSaver", "load_json", "save_json"]

import json
from pathlib import Path
from typing import Any, TypeVar

from coola.equality.interface import objects_are_equal
from coola.io.base import BaseFileSaver, BaseLoader
from coola.utils.format import repr_mapping_line

T = TypeVar("T")

DEFAULT_ENCODING = "utf-8"


class JsonLoader(BaseLoader[T]):
    r"""Implement a data loader to load data in a JSON file.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_json, JsonLoader
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.json")
        ...     save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = JsonLoader().load(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return type(other) is type(self)

    def load(self, path: Path) -> T:
        with Path.open(path, mode="rb") as file:
            return json.load(file)


class JsonSaver(BaseFileSaver[T]):
    r"""Implement a file saver to save data with a JSON file.

    Args:
        encoding: The file encoding to use when writing the file.
            Defaults to ``"utf-8"``.
        **kwargs: Additional arguments passed to ``json.dump``, e.g.
            ``indent`` or ``sort_keys``.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import JsonSaver, JsonLoader
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.json")
        ...     JsonSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = JsonLoader().load(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """

    def __init__(self, encoding: str = DEFAULT_ENCODING, **kwargs: Any) -> None:
        self._encoding = encoding
        self._kwargs = kwargs

    def __repr__(self) -> str:
        kwargs = f", {repr_mapping_line(self._kwargs)}" if self._kwargs else ""
        return f"{self.__class__.__qualname__}(encoding={self._encoding}{kwargs})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return self._encoding == other._encoding and objects_are_equal(
            self._kwargs, other._kwargs, equal_nan=equal_nan
        )

    def _save_file(self, to_save: T, path: Path) -> None:
        with Path.open(path, mode="w", encoding=self._encoding) as file:
            json.dump(to_save, file, **self._kwargs)


def load_json(path: Path) -> Any:
    r"""Load the data from a given JSON file.

    Args:
        path: The path to the JSON file.

    Returns:
        The data from the JSON file.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_json, load_json
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.json")
        ...     save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = load_json(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """
    return JsonLoader().load(path)


def save_json(
    to_save: Any,
    path: Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    exist_ok: bool = False,
    **kwargs: Any,
) -> None:
    r"""Save the given data in a JSON file.

    Args:
        to_save: The data to write in a JSON file.
        path: The path where to write the JSON file.
        encoding: The file encoding to use when writing the file.
            Defaults to ``"utf-8"``.
        exist_ok: If ``exist_ok`` is ``False`` (the default),
            ``FileExistsError`` is raised if the target file
            already exists. If ``exist_ok`` is ``True``,
            ``FileExistsError`` will not be raised unless the
            given path already exists in the file system and is
            not a file.
        **kwargs: Additional arguments passed to ``json.dump``, e.g.
            ``indent`` or ``sort_keys``.

    Raises:
        FileExistsError: if the file already exists.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_json, load_json
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.json")
        ...     save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = load_json(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """
    JsonSaver(encoding=encoding, **kwargs).save(to_save, path, exist_ok=exist_ok)
