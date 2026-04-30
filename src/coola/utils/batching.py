r"""Contain utility functions for batching."""

from __future__ import annotations

__all__ = ["batchify"]

from itertools import islice
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

T = TypeVar("T")


def batchify(items: Iterable[T], *, size: int) -> Iterator[tuple[T, ...]]:
    r"""Generate batches of items from an iterable.

    Splits ``items`` into consecutive non-overlapping tuples of length
    ``size``. The last batch may be shorter than ``size`` if the
    number of items is not evenly divisible.

    Args:
        items: The iterable of items to batch.
        size: The maximum number of items per batch. Must be >= 1.

    Yields:
        Tuples of at most ``size`` items each. Nothing is yielded
            when ``items`` is empty.

    Raises:
        ValueError: If ``size`` is less than 1.

    Example:
        ```pycon
        >>> from coola.utils.batching import batchify
        >>> list(batchify([1, 2, 3, 4, 5], size=2))
        [(1, 2), (3, 4), (5,)]
        >>> list(batchify([], size=3))
        []
        >>> list(batchify(iter(range(6)), size=3))
        [(0, 1, 2), (3, 4, 5)]

        ```
    """
    if size < 1:
        msg = "size must be >= 1"
        raise ValueError(msg)
    iterator = iter(items)
    while batch := tuple(islice(iterator, size)):
        yield batch
