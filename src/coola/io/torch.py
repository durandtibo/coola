r"""Contain torch-based data loaders and savers."""

from __future__ import annotations

__all__ = ["TorchLoader", "TorchSaver", "load_torch", "save_torch"]

from typing import TYPE_CHECKING, Any

from coola.io.base import BaseFileSaver, BaseLoader
from coola.utils.imports import check_torch, is_torch_available

if TYPE_CHECKING:
    import os
    from pathlib import Path

if TYPE_CHECKING or is_torch_available():  # pragma: no cover
    import torch


class TorchLoader(BaseLoader[Any]):
    r"""Implement a data loader to load data in a PyTorch file.

    Args:
        weights_only: If ``True`` (the default), only tensors and
            simple containers are loaded. Pass ``False`` only for
            trusted files.
        **kwargs: Additional arguments passed to ``torch.load``.

    Security:
        Loading with ``weights_only=False`` can execute arbitrary code
        (it uses pickle). Only do so for files from a trusted source.

    Raises:
        MissingPackageError: If ``torch`` is not installed.

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

    def __init__(self, *, weights_only: bool = True, **kwargs: Any) -> None:
        check_torch()
        self._weights_only = weights_only
        self._kwargs = kwargs

    def _display_fields(self) -> dict[str, Any]:
        r"""Return the fields shown in the representation.

        Returns:
            The ``weights_only`` option and the arguments of
                ``torch.load``.
        """
        return {"weights_only": self._weights_only, **self._kwargs}

    def _load(self, path: Path) -> Any:
        r"""Load a file with ``torch.load``.

        Args:
            path: The path of the file.

        Returns:
            The data of the file.
        """
        return torch.load(path, weights_only=self._weights_only, **self._kwargs)


class TorchSaver(BaseFileSaver[Any]):
    r"""Implement a file saver to save data with a PyTorch file.

    Args:
        **kwargs: Additional arguments passed to ``torch.save``.

    Raises:
        MissingPackageError: If ``torch`` is not installed.

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
        r"""Return the fields shown in the representation.

        Returns:
            The arguments of ``torch.save``.
        """
        return self._kwargs

    def _save_file(self, to_save: Any, path: Path) -> None:
        r"""Save the data with ``torch.save``.

        Args:
            to_save: The data to save.
            path: The path of the file.
        """
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
    return TorchLoader(**kwargs).load(path)


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
        to_save: Anyhe data to write in a PyTorch file.
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
    TorchSaver(**kwargs).save(to_save, path, exist_ok=exist_ok, fsync=fsync)
