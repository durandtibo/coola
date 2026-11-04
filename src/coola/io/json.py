r"""Contain JSON-based data loaders and savers."""

from __future__ import annotations

__all__ = ["JsonLoader", "JsonSaver", "load_json", "save_json"]

import json
from typing import TYPE_CHECKING, Any

from coola.io.base import BaseFileSaver, BaseLoader
from coola.jsonable import to_jsonable

if TYPE_CHECKING:
    import os
    from pathlib import Path


DEFAULT_ENCODING = "utf-8"


class JsonLoader(BaseLoader[Any]):
    r"""Implement a data loader to load data in a JSON file.

    Args:
        encoding: The encoding of the file, which should be the one
            used to save it. If ``None`` (default), the file is read
            in binary mode and ``json`` detects the encoding, which
            only works for UTF-8, UTF-16 and UTF-32.

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

    def __init__(self, encoding: str | None = None) -> None:
        self._encoding = encoding

    def _display_fields(self) -> dict[str, Any]:
        return {"encoding": self._encoding}

    def _load(self, path: Path) -> Any:
        if self._encoding is None:
            with path.open(mode="rb") as file:
                return json.load(file)
        with path.open(encoding=self._encoding) as file:
            return json.load(file)


class JsonSaver(BaseFileSaver[Any]):
    r"""Implement a file saver to save data with a JSON file.

    Args:
        encoding: The file encoding to use when writing the file.
            Defaults to ``"utf-8"``.
        allow_nan: If ``False`` (the default), ``NaN`` and infinite
            floats raise a ``ValueError``, because the file would not be
            valid JSON. If ``True``, they are written as the
            non-standard ``NaN``/``Infinity`` tokens.
        jsonable: If ``True``, the data is converted with
            ``coola.jsonable.to_jsonable`` before it is written, so the
            objects that ``json`` does not support (tuples are lists,
            NumPy arrays, ...) are converted. Defaults to ``False``.
        **kwargs: Additional arguments passed to ``json.dump``, e.g.
            ``indent`` or ``sort_keys``.

    Note:
        JSON has only string keys: ``int`` keys are loaded back as
        ``str``.

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

    def __init__(
        self,
        encoding: str = DEFAULT_ENCODING,
        *,
        allow_nan: bool = False,
        jsonable: bool = False,
        **kwargs: Any,
    ) -> None:
        self._encoding = encoding
        self._allow_nan = allow_nan
        self._jsonable = jsonable
        self._kwargs = kwargs

    def _display_fields(self) -> dict[str, Any]:
        return {
            "encoding": self._encoding,
            "allow_nan": self._allow_nan,
            "jsonable": self._jsonable,
            **self._kwargs,
        }

    def _save_file(self, to_save: Any, path: Path) -> None:
        if self._jsonable:
            to_save = to_jsonable(to_save)
        with path.open(mode="w", encoding=self._encoding) as file:
            json.dump(to_save, file, allow_nan=self._allow_nan, **self._kwargs)


def load_json(path: str | os.PathLike[str], encoding: str | None = None) -> Any:
    r"""Load the data from a given JSON file.

    Args:
        path: The path to the JSON file. A ``str`` or any path-like object is converted to a ``Path``.
        encoding: The encoding of the file. If ``None``, the encoding
            is detected (UTF-8, UTF-16 and UTF-32 only).

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
    return JsonLoader(encoding=encoding).load(path)


def save_json(
    to_save: Any,
    path: str | os.PathLike[str],
    *,
    encoding: str = DEFAULT_ENCODING,
    exist_ok: bool = False,
    fsync: bool = False,
    allow_nan: bool = False,
    jsonable: bool = False,
    **kwargs: Any,
) -> None:
    r"""Save the given data in a JSON file.

    The file is what ``json.dump`` writes: the keys that are not
    strings (for example ``int``) are written as strings, so they are
    strings after ``load_json``. ``NaN`` and infinite floats are not
    valid JSON, so they raise a ``ValueError`` unless ``allow_nan=True``
    (they are then written as the non-standard tokens ``NaN`` and
    ``Infinity``). The data is not converted unless ``jsonable=True``:
    the objects that ``json`` does not support (NumPy arrays, ...) raise
    a ``TypeError``.

    Args:
        to_save: Anyhe data to write in a JSON file.
        path: The path where to write the JSON file. A ``str`` or any path-like object is converted to a ``Path``.
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
        allow_nan: If ``False`` (the default), ``NaN`` and infinite
            floats raise a ``ValueError``. If ``True``, they are
            written as the non-standard ``NaN``/``Infinity`` tokens.
        jsonable: If ``True``, convert the data with
            ``coola.jsonable.to_jsonable`` before it is written.
            Defaults to ``False``.
        **kwargs: Additional arguments passed to ``json.dump``, e.g.
            ``indent`` or ``sort_keys``.

    Raises:
        FileExistsError: if the file already exists.
        ValueError: if the data has a ``NaN`` or an infinite float and
            ``allow_nan`` is ``False``.

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
    JsonSaver(encoding=encoding, allow_nan=allow_nan, jsonable=jsonable, **kwargs).save(
        to_save, path, exist_ok=exist_ok, fsync=fsync
    )
