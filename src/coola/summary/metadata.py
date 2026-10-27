r"""Implement the base class of the summarizers that describe the
metadata of an object."""

from __future__ import annotations

__all__ = ["BaseMetadataSummarizer", "Field", "format_fields"]

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from coola.summary.base import BaseSummarizer
from coola.utils.safe import safe_repr

if TYPE_CHECKING:
    from collections.abc import Iterable

    from coola.summary.config import SummaryConfig

T = TypeVar("T")

Field: TypeAlias = "str | tuple[str, Iterable[tuple[Any, Any]]]"
r"""A field of the metadata: a text such as ``"shape=(2, 3)"``, or a
name and the ``(name, type)`` pairs of the columns, which are formatted
with ``format_fields`` and limited to ``max_items``."""


def format_fields(fields: Iterable[tuple[Any, Any]], max_fields: int | None) -> str:
    r"""Format ``(name, type)`` pairs as ``name: type``, with at most
    ``max_fields`` of them.

    Args:
        fields: The ``(name, type)`` pairs, for example the columns of a
            data frame and their types.
        max_fields: The maximum number of pairs to show. The other ones
            are replaced by ``... (+N more)``. ``None`` shows all of
            them.

    Returns:
        The formatted pairs, separated by a comma.

    Example:
        ```pycon
        >>> from coola.summary import format_fields
        >>> format_fields([("a", "int64"), ("b", "float64")], 5)
        'a: int64, b: float64'
        >>> format_fields([("a", "int64"), ("b", "float64"), ("c", "object")], 1)
        'a: int64, ... (+2 more)'

        ```
    """
    items = [f"{name}: {dtype}" for name, dtype in fields]
    if max_fields is None or len(items) <= max_fields:
        return ", ".join(items)
    hidden = len(items) - max_fields
    return ", ".join([*items[:max_fields], f"... (+{hidden} more)"])


class BaseMetadataSummarizer(BaseSummarizer[T]):
    r"""Define the base class of the summarizers of objects that are
    described by their metadata (arrays, tensors, data frames, ...).

    A subclass implements ``_describe`` and, if the type needs an
    optional library, ``_check_dependencies``. The summary is a leaf:
    the type followed by the fields, separated by ``|``. If
    ``config.show_data`` is ``True``, it is the ``repr`` of the data
    (cut by the engine at ``max_characters``).

    Raises:
        MissingPackageError: If the optional library of the type is not
            installed (raised when the summarizer is created).

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.summary import NDArraySummarizer, SummaryConfig
        >>> summarizer = NDArraySummarizer()
        >>> print(summarizer.summarize(np.ones((2, 3)), SummaryConfig(), compact=False))
        <class 'numpy.ndarray'> | shape=(2, 3) | dtype=float64
        >>> print(summarizer.summarize(np.ones(2), SummaryConfig(show_data=True), compact=False))
        array([1., 1.])

        ```
    """

    def __init__(self) -> None:
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        r"""Check that the optional library of the type is installed.

        Raises:
            MissingPackageError: If the library is not installed. By
                default, nothing is checked.
        """

    @abstractmethod
    def _describe(self, data: T) -> list[Field]:
        r"""Describe the metadata of the data.

        Args:
            data: The data to describe.

        Returns:
            The fields of the metadata, for example
                ``["shape=(2, 3)", "dtype=float64"]``. A table returns
                its columns as ``("columns", pairs)``, so the engine
                limits them to ``max_items``.
        """

    def summarize(self, data: T, config: SummaryConfig, *, compact: bool) -> str:  # noqa: ARG002
        if config.show_data:
            return safe_repr(data)
        parts = [f"{type(data)}"]
        for field in self._describe(data):
            if isinstance(field, str):
                parts.append(field)
            else:
                name, pairs = field
                parts.append(f"{name}={format_fields(pairs, config.max_items)}")
        return " | ".join(parts)
