r"""Define the functions that pick the loader or the saver from the
suffix of a path."""

from __future__ import annotations

__all__ = ["load", "save"]

from pathlib import Path
from typing import TYPE_CHECKING, Any

from coola.factory import import_object
from coola.validation import validate_in

if TYPE_CHECKING:
    import os

# The loaders and the savers are given by their path, so PyTorch is imported only for a
# ``.pt`` file.
_LOADERS = {
    ".json": "coola.io.json.JsonLoader",
    ".pkl": "coola.io.pickle.PickleLoader",
    ".pickle": "coola.io.pickle.PickleLoader",
    ".pt": "coola.io.torch.TorchLoader",
    ".pth": "coola.io.torch.TorchLoader",
    ".txt": "coola.io.text.TextLoader",
}
_SAVERS = {
    ".json": "coola.io.json.JsonSaver",
    ".pkl": "coola.io.pickle.PickleSaver",
    ".pickle": "coola.io.pickle.PickleSaver",
    ".pt": "coola.io.torch.TorchSaver",
    ".pth": "coola.io.torch.TorchSaver",
    ".txt": "coola.io.text.TextSaver",
}


def load(path: str | os.PathLike[str], **kwargs: Any) -> Any:
    r"""Load the data from a file, with the loader of its suffix.

    The suffixes are ``.json`` (``JsonLoader``), ``.pkl`` and
    ``.pickle`` (``PickleLoader``), ``.pt`` and ``.pth``
    (``TorchLoader``) and ``.txt`` (``TextLoader``). The suffix is
    case-insensitive.

    Warning:
        A ``.pkl`` file is loaded with ``pickle.load``, which can
        execute arbitrary code. Only load a pickle file that comes from
        a trusted source.

    Args:
        path: The path to the file.
        **kwargs: The arguments of the loader, for example ``encoding``
            for ``.json`` and ``.txt``, or ``weights_only`` for ``.pt``.

    Returns:
        The data from the file.

    Raises:
        ValueError: If the suffix of ``path`` is not supported.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import load, save
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.json")
        ...     save({"key": [1, 2, 3]}, path)
        ...     load(path)
        ...
        {'key': [1, 2, 3]}

        ```
    """
    suffix = Path(path).suffix.lower()
    validate_in(suffix, tuple(_LOADERS), name="suffix of the path")
    return import_object(_LOADERS[suffix])(**kwargs).load(path)


def save(
    to_save: Any,
    path: str | os.PathLike[str],
    *,
    exist_ok: bool = False,
    fsync: bool = False,
    **kwargs: Any,
) -> None:
    r"""Save the data in a file, with the saver of its suffix.

    The suffixes are ``.json`` (``JsonSaver``), ``.pkl`` and
    ``.pickle`` (``PickleSaver``), ``.pt`` and ``.pth``
    (``TorchSaver``) and ``.txt`` (``TextSaver``). The suffix is
    case-insensitive.

    Args:
        to_save: The data to write.
        path: The path where to write the file.
        exist_ok: If ``False`` (the default), ``FileExistsError`` is
            raised if the file already exists.
        fsync: If ``True``, flush the data to the disk before committing
            the file. Defaults to ``False``.
        **kwargs: The arguments of the saver, for example ``indent``
            for ``.json`` or ``encoding`` for ``.txt``.

    Raises:
        ValueError: If the suffix of ``path`` is not supported.
        FileExistsError: If the file already exists.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from coola.io import load, save
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("data.txt")
        ...     save("abc", path)
        ...     load(path)
        ...
        'abc'

        ```
    """
    suffix = Path(path).suffix.lower()
    validate_in(suffix, tuple(_SAVERS), name="suffix of the path")
    import_object(_SAVERS[suffix])(**kwargs).save(to_save, path, exist_ok=exist_ok, fsync=fsync)
