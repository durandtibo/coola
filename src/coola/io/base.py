r"""Contain the base class to implement a data loader or saver
object."""

from __future__ import annotations

__all__ = [
    "BaseFileSaver",
    "BaseLoader",
    "BaseSaver",
    "is_loader_config",
    "is_saver_config",
    "resolve_loader",
    "resolve_saver",
]

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from coola.equality.tester import EqualNanEqualityTester, get_default_registry
from coola.factory import is_object_config, resolve_object
from coola.io.utils import add_uuid_suffix

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")

logger: logging.Logger = logging.getLogger(__name__)


class BaseLoader(ABC, Generic[T]):
    r"""Define the base class to implement a data loader.

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

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.io import JsonLoader, TextLoader
            >>> JsonLoader().equal(JsonLoader())
            True
            >>> JsonLoader().equal(TextLoader())
            False

            ```
        """

    @abstractmethod
    def load(self, path: Path) -> T:
        r"""Load the data from the given path.

        Args:
            path: The path with the data to load.

        Returns:
            The data

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


class BaseSaver(ABC, Generic[T]):
    r"""Define the base class to implement a data saver.

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

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.io import JsonSaver, TextSaver
            >>> JsonSaver().equal(JsonSaver())
            True
            >>> JsonSaver().equal(TextSaver())
            False

            ```
        """

    @abstractmethod
    def save(self, to_save: T, path: Path, *, exist_ok: bool = False) -> None:
        r"""Save the data into the given path.

        Args:
            to_save: The data to save. The data should be compatible
                with the saving engine.
            path: The path where to save the data.
            exist_ok: If ``exist_ok`` is ``False`` (the default),
                an exception is raised if the target path already
                exists.

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


class BaseFileSaver(BaseSaver[T]):
    r"""Define the base class to implement a file saver.

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

    def save(self, to_save: T, path: Path, *, exist_ok: bool = False) -> None:
        r"""Save the data into the given path.

        Args:
            to_save: The data to save. The data should be compatible
                with the saving engine.
            path: The path where to save the data.
            exist_ok: If ``exist_ok`` is ``False`` (the default),
                ``FileExistsError`` is raised if the target file
                already exists. If ``exist_ok`` is ``True``,
                ``FileExistsError`` will not be raised unless the
                given path already exists in the file system and is
                not a file.

        Raises:
            FileExistsError: if the file already exists.

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
        if path.is_dir():
            msg = f"path ({path}) is a directory"
            raise IsADirectoryError(msg)
        if path.is_file() and not exist_ok:
            msg = f"path ({path}) already exists. Use `exist_ok=True` to overwrite the file"
            raise FileExistsError(msg)
        path.parent.mkdir(exist_ok=True, parents=True)

        # Save to tmp, then commit by moving the file in case the job gets
        # interrupted while writing the file
        tmp_path = add_uuid_suffix(path)
        try:
            self._save_file(to_save, tmp_path)
            if exist_ok:
                tmp_path.replace(path)
            else:
                # ``os.link`` + unlink is used instead of a plain rename so the
                # ``exist_ok=False`` guarantee also holds if ``path`` was created
                # concurrently between the check above and this commit step:
                # ``link`` atomically fails with ``FileExistsError`` in that case.
                try:
                    os.link(tmp_path, path)
                finally:
                    tmp_path.unlink(missing_ok=True)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @abstractmethod
    def _save_file(self, to_save: T, path: Path) -> None:
        r"""Save the data into the given file.

        Args:
            to_save: The data to save. The data should be compatible
                with the saving engine.
            path: The path where to save the data.
        """


def is_loader_config(config: dict[Any, Any]) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseLoader``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseLoader`` object.

    Example:
        ```pycon
        >>> from coola.io import is_loader_config
        >>> is_loader_config({"_target_": "coola.io.JsonLoader"})
        True

        ```
    """
    return is_object_config(config, BaseLoader)


def is_saver_config(config: dict[Any, Any]) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseSaver``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseSaver`` object.

    Example:
        ```pycon
        >>> from coola.io import is_saver_config
        >>> is_saver_config({"_target_": "coola.io.JsonSaver"})
        True

        ```
    """
    return is_object_config(config, BaseSaver)


def resolve_loader(loader: BaseLoader[T] | dict[Any, Any]) -> BaseLoader[T]:
    r"""Set up a data loader.

    The data loader is instantiated from its configuration by using the
    ``BaseLoader`` factory function.

    Args:
        loader: The data loader or its configuration.

    Returns:
        The instantiated data loader.

    Example:
        ```pycon
        >>> from coola.io import resolve_loader
        >>> loader = resolve_loader({"_target_": "coola.io.JsonLoader"})
        >>> loader
        JsonLoader()

        ```
    """
    return resolve_object(loader, BaseLoader)


def resolve_saver(saver: BaseSaver[T] | dict[Any, Any]) -> BaseSaver[T]:
    r"""Set up a data saver.

    The data saver is instantiated from its configuration by using the
    ``BaseSaver`` factory function.

    Args:
        saver: The data saver or its configuration.

    Returns:
        The instantiated data saver.

    Example:
        ```pycon
        >>> from coola.io import resolve_saver
        >>> saver = resolve_saver({"_target_": "coola.io.JsonSaver"})
        >>> saver
        JsonSaver()

        ```
    """
    return resolve_object(saver, BaseSaver)


get_default_registry().register_many(
    {BaseLoader: EqualNanEqualityTester(), BaseSaver: EqualNanEqualityTester()}, exist_ok=True
)
