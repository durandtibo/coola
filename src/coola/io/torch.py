r"""Contain torch-based data loaders and savers."""

from __future__ import annotations

__all__ = ["TorchLoader", "TorchSaver", "load_torch", "save_torch"]

from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import InlineDisplayMixin
from coola.equality import objects_are_equal
from coola.io.base import BaseFileSaver, BaseLoader
from coola.utils.imports import check_torch, is_torch_available
from coola.utils.introspection import is_same_type
from coola.utils.path import sanitize_path

if TYPE_CHECKING:
    import os
    from pathlib import Path

if TYPE_CHECKING or is_torch_available():  # pragma: no cover
    import torch

T = TypeVar("T")


class TorchLoader(InlineDisplayMixin, BaseLoader[T]):
    r"""Implement a data loader to load data in a PyTorch file.

    Args:
        **kwargs: Additional arguments passed to ``torch.load``.
            ``weights_only`` defaults to ``True``; pass
            ``weights_only=False`` only for trusted files.

    Security:
        Loading with ``weights_only=False`` can execute arbitrary code
        (it uses pickle). Only do so for files from a trusted source.

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

    def _display_fields(self) -> dict[str, Any]:
        return self._kwargs

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not is_same_type(other, self):
            return False
        # Compare the arguments that ``load`` uses, so the default is not
        # different from an explicit ``weights_only=True``.
        return objects_are_equal(self._load_kwargs(), other._load_kwargs(), equal_nan=equal_nan)

    def _load_kwargs(self) -> dict[str, Any]:
        return {"weights_only": True, **self._kwargs}

    def load(self, path: str | os.PathLike[str]) -> T:
        path = sanitize_path(path, resolve=False)
        return torch.load(path, **self._load_kwargs())


class TorchSaver(InlineDisplayMixin, BaseFileSaver[T]):
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

    def _display_fields(self) -> dict[str, Any]:
        return self._kwargs

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not is_same_type(other, self):
            return False
        return objects_are_equal(self._kwargs, other._kwargs, equal_nan=equal_nan)

    def _save_file(self, to_save: T, path: Path) -> None:
        torch.save(to_save, path, **self._kwargs)


def load_torch(path: str | os.PathLike[str], **kwargs: Any) -> Any:
    r"""Load the data from a given PyTorch file.

    Args:
        path: The path to the PyTorch file. A ``str`` or any path-like object is converted to a ``Path``.
        **kwargs: Additional arguments passed to ``torch.load``.

    Returns:
        The data from the PyTorch file.

    Security:
        ``weights_only`` defaults to ``True``, which restricts the
        unpickler to tensors and simple containers. Pass
        ``weights_only=False`` only for files from a trusted source:
        it uses pickle, which can execute arbitrary code.

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
    return TorchLoader[Any](**kwargs).load(path)


def save_torch(
    to_save: Any,
    path: str | os.PathLike[str],
    *,
    exist_ok: bool = False,
    fsync: bool = False,
    **kwargs: Any,
) -> None:
    r"""Save the given data in a PyTorch file.

    Args:
        to_save: The data to write in a PyTorch file.
        path: The path where to write the PyTorch file. A ``str`` or any path-like object is converted to a ``Path``.
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
    TorchSaver[Any](**kwargs).save(to_save, path, exist_ok=exist_ok, fsync=fsync)
