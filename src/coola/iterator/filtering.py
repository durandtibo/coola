r"""Type-based filtering utilities for iterators.

This module provides utilities for filtering iterators based on type
checks, allowing you to extract only values matching specific types from
heterogeneous sequences.
"""

from __future__ import annotations

__all__ = ["filter_by_type"]

from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

T = TypeVar("T")


def filter_by_type(iterator: Iterable[Any], types: type[T] | tuple[type, ...]) -> Iterator[T]:
    """Filter an iterator to yield only values of specified types.

    This function acts as a type-safe filter that passes through only values
    matching the specified type(s). It's particularly useful when working with
    heterogeneous collections where you need to extract elements of specific
    types.

    Args:
        iterator: An iterator or iterable to filter. Can contain values of any type.
        types: A single type or tuple of types to filter for. Only values that are
            instances of these types will be yielded. Follows the same semantics
            as the built-in isinstance() function.

    Yields:
        Values from the input iterator that are instances of any of the specified
        types, preserving the original order of elements.

    Example:
        Filter mixed-type list to get only integers:

        ```pycon
        >>> from coola.iterator import filter_by_type
        >>> list(filter_by_type([1, "hello", 2, 3.14, "world", 4], int))
        [1, 2, 4]

        ```

        Filter for multiple types:

        ```pycon
        >>> from coola.iterator import filter_by_type
        >>> # Note: bool is a subclass of int
        >>> list(filter_by_type([1, "hello", 2.5, True, None, [1, 2]], (int, float)))
        [1, 2.5, True]

        ```

    Notes:
        - This function uses isinstance() internally, so subclass relationships
          are respected (e.g., bool values will match int type).
        - The input iterator is consumed as items are yielded.
        - For empty iterators or when no items match, the generator yields nothing.
    """
    for value in iterator:
        if isinstance(value, types):
            yield value
