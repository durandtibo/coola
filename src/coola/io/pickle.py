r"""Contain pickle-based data loaders and savers."""

from __future__ import annotations

__all__ = ["PickleLoader", "PickleSaver", "load_pickle", "save_pickle"]

import pickle
from typing import TYPE_CHECKING, Any

from coola.io.base import BaseFileSaver, BaseLoader

if TYPE_CHECKING:
    import os
    from pathlib import Path


class PickleLoader(BaseLoader[Any]):
    r"""Implement a data loader to load data in a pickle file.

    Warning:
        ``load`` calls ``pickle.load`` on the given file. Unpickling
        can execute arbitrary code as a side effect of deserialization.
        Only load pickle files that come from a trusted source; never
        load a pickle file that comes from an untrusted or
        unauthenticated source.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_pickle, PickleLoader
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.pkl")
        ...     save_pickle({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = PickleLoader().load(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """

    def _load(self, path: Path) -> Any:
        r"""Load a pickle file.

        Args:
            path: The path of the file.

        Returns:
            The data of the file.
        """
        with path.open(mode="rb") as file:
            return pickle.load(file)  # noqa: S301


class PickleSaver(BaseFileSaver[Any]):
    r"""Implement a file saver to save data with a pickle file.

    Args:
        **kwargs: Additional arguments passed to ``pickle.dump``.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import PickleSaver, PickleLoader
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.pkl")
        ...     PickleSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = PickleLoader().load(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def _display_fields(self) -> dict[str, Any]:
        r"""Return the fields shown in the representation.

        Returns:
            The arguments of ``pickle.dump``.
        """
        return self._kwargs

    def _save_file(self, to_save: Any, path: Path) -> None:
        r"""Save the data into a pickle file.

        Args:
            to_save: The data to save.
            path: The path of the file.
        """
        with path.open(mode="wb") as file:
            pickle.dump(to_save, file, **self._kwargs)


def load_pickle(path: str | os.PathLike[str]) -> Any:
    r"""Load the data from a given pickle file.

    Args:
        path: The path to the pickle file. A ``str`` or any path-like object is converted to a ``Path``.

    Returns:
        The data from the pickle file.

    Warning:
        This function unpickles the file's content, which can execute
        arbitrary code as a side effect of deserialization. Only load
        pickle files that come from a trusted source; never load a
        pickle file that comes from an untrusted or unauthenticated
        source.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_pickle, load_pickle
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.pkl")
        ...     save_pickle({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = load_pickle(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """
    return PickleLoader().load(path)


def save_pickle(
    to_save: Any,
    path: str | os.PathLike[str],
    *,
    exist_ok: bool = False,
    fsync: bool = False,
    **kwargs: Any,
) -> None:
    r"""Save the given data in a pickle file.

    Args:
        to_save: Anyhe data to write in a pickle file.
        path: The path where to write the pickle file. A ``str`` or any path-like object is converted to a ``Path``.
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
        **kwargs: Additional arguments passed to ``pickle.dump``.

    Raises:
        FileExistsError: if the file already exists.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_pickle, load_pickle
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.pkl")
        ...     save_pickle({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = load_pickle(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """
    PickleSaver(**kwargs).save(to_save, path, exist_ok=exist_ok, fsync=fsync)
