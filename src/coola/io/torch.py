r"""Contain torch-based data loaders and savers."""

from __future__ import annotations

__all__ = ["TorchLoader", "TorchSaver", "load_torch", "save_torch"]

from typing import TYPE_CHECKING, Any, TypeVar

from coola.equality.interface import objects_are_equal
from coola.io.base import BaseFileSaver, BaseLoader
from coola.utils.format import repr_mapping_line
from coola.utils.imports import check_torch, is_torch_available

if TYPE_CHECKING:
    from pathlib import Path

if TYPE_CHECKING or is_torch_available():  # pragma: no cover
    import torch

T = TypeVar("T")


class TorchLoader(BaseLoader[T]):
    r"""Implement a data loader to load data in a PyTorch file.

    Args:
        **kwargs: Additional arguments passed to ``torch.load``.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_torch, TorchLoader
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.pt")
        ...     save_torch({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = TorchLoader().load(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        check_torch()
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({repr_mapping_line(self._kwargs)})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self._kwargs, other._kwargs, equal_nan=equal_nan)

    def load(self, path: Path) -> T:
        return torch.load(path, **self._kwargs)


class TorchSaver(BaseFileSaver[T]):
    r"""Implement a file saver to save data with a PyTorch file.

    Args:
        **kwargs: Additional arguments passed to ``torch.save``.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import TorchSaver, TorchLoader
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.pt")
        ...     TorchSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = TorchLoader().load(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        check_torch()
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({repr_mapping_line(self._kwargs)})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        return objects_are_equal(self._kwargs, other._kwargs, equal_nan=equal_nan)

    def _save_file(self, to_save: T, path: Path) -> None:
        torch.save(to_save, path, **self._kwargs)


def load_torch(path: Path, **kwargs: Any) -> Any:
    r"""Load the data from a given PyTorch file.

    Args:
        path: The path to the PyTorch file.
        **kwargs: Additional arguments passed to ``torch.load``.

    Returns:
        The data from the PyTorch file.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_torch, load_torch
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.pt")
        ...     save_torch({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = load_torch(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """
    return TorchLoader(**kwargs).load(path)


def save_torch(to_save: Any, path: Path, *, exist_ok: bool = False, **kwargs: Any) -> None:
    r"""Save the given data in a PyTorch file.

    Args:
        to_save: The data to write in a PyTorch file.
        path: The path where to write the PyTorch file.
        exist_ok: If ``exist_ok`` is ``False`` (the default),
            ``FileExistsError`` is raised if the target file
            already exists. If ``exist_ok`` is ``True``,
            ``FileExistsError`` will not be raised unless the
            given path already exists in the file system and is
            not a file.
        **kwargs: Additional arguments passed to ``torch.save``.

    Raises:
        FileExistsError: if the file already exists.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import save_torch, load_torch
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.pt")
        ...     save_torch({"key1": [1, 2, 3], "key2": "abc"}, path)
        ...     data = load_torch(path)
        ...     data
        ...
        {'key1': [1, 2, 3], 'key2': 'abc'}

        ```
    """
    TorchSaver(**kwargs).save(to_save, path, exist_ok=exist_ok)
