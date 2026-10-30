r"""Define a bounded, dict-like LRU (least-recently-used) cache.

A standalone utility for the data structures that need a bounded lookup
cache: it caps memory usage while keeping the most recently used entries
around. It is not used by the registries of ``coola`` (``TypeRegistry``
uses a ``WeakKeyDictionary`` as cache). This class is not thread-safe on
its own — callers that share an instance across threads are expected to
guard access with their own lock.
"""

from __future__ import annotations

__all__ = ["LRUCache"]

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from coola.validation import validate_gt

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, ValuesView

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class LRUCache(Generic[K, V]):
    r"""A bounded, dict-like cache that evicts the least-recently-used
    entry once it grows past ``maxsize``.

    Both reading (``__getitem__``/``get``) and writing (``__setitem__``)
    an existing key mark it as most-recently-used. The class supports the
    common ``dict``-like operations (``__contains__``, ``__len__``,
    ``__iter__``, ``__delitem__``, ``keys``/``values``/``items``,
    ``clear``, ``pop``) needed to act as a drop-in cache store.

    Warning:
        The cache is **not thread-safe**, unlike the registries of
        ``coola``: even a read (``__getitem__``, ``get``) modifies the
        order of the entries. Protect it with a lock if it is shared
        between threads.

    Args:
        maxsize: The maximum number of entries the cache holds. Must be
            a positive integer. Once a new entry would push the cache
            past this size, the least-recently-used entry is evicted.

    Raises:
        ValueError: if ``maxsize`` is not a positive integer.

    Example:
        ```pycon
        >>> from coola.utils.lru import LRUCache
        >>> cache = LRUCache[str, int](maxsize=2)
        >>> cache["a"] = 1
        >>> cache["b"] = 2
        >>> cache["a"]  # accessing "a" marks it as most-recently-used
        1
        >>> cache["c"] = 3  # evicts "b", the least-recently-used entry
        >>> list(cache)
        ['a', 'c']

        ```
    """

    def __init__(self, maxsize: int) -> None:
        validate_gt(maxsize, 0, name="maxsize")
        self._maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()

    def __repr__(self) -> str:
        r"""Return the representation, with the entries.

        Returns:
            The representation of the cache.
        """
        return f"{self.__class__.__qualname__}(maxsize={self._maxsize}, data={self._data})"

    def __str__(self) -> str:
        r"""Return the string, with the number of entries.

        Returns:
            The string of the cache.
        """
        return f"{self.__class__.__qualname__}(maxsize={self._maxsize}, size={len(self._data)})"

    def __len__(self) -> int:
        r"""Return the number of entries.

        Returns:
            The number of entries.
        """
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        r"""Indicate if a key is in the cache, without changing the
        order.

        Args:
            key: The key.

        Returns:
            ``True`` if the key is in the cache, otherwise ``False``.
        """
        return key in self._data

    def __iter__(self) -> Iterator[K]:
        r"""Iterate over the keys, from the least to the most recently
        used.

        Returns:
            An iterator over the keys.
        """
        return iter(self._data)

    def __getitem__(self, key: K) -> V:
        r"""Get the value associated with ``key`` and mark it as most-
        recently-used.

        Args:
            key: The key to look up.

        Returns:
            The value associated with ``key``.

        Raises:
            KeyError: if ``key`` is not in the cache.
        """
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key: K, value: V) -> None:
        r"""Set ``value`` for ``key``, marking it as most-recently-used,
        and evict the least-recently-used entry if the cache is now over
        capacity.

        Args:
            key: The key to set.
            value: The value to associate with ``key``.
        """
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __delitem__(self, key: K) -> None:
        r"""Remove a key.

        Args:
            key: The key.

        Raises:
            KeyError: If the key is not in the cache.
        """
        del self._data[key]

    def __hash__(self) -> int:
        r"""Raise an error because the cache is mutable.

        Raises:
            TypeError: Always.
        """
        # Mutable and compares by content, like dict: not hashable.
        msg = f"unhashable type: {type(self).__name__!r}"
        raise TypeError(msg)

    def __eq__(self, other: object) -> bool:
        r"""Indicate if the cache is equal to another cache or a
        dictionary.

        Args:
            other: The other object.

        Returns:
            ``True`` if ``other`` is a cache or a dictionary with the
                same entries, ``NotImplemented`` for another type,
                otherwise ``False``.
        """
        if isinstance(other, LRUCache):
            return self._data == cast("LRUCache[Any, Any]", other)._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented

    @property
    def maxsize(self) -> int:
        r"""The maximum number of entries the cache holds.

        Returns:
            The maximum number of entries.
        """
        return self._maxsize

    def get(self, key: K, default: V | None = None) -> V | None:
        r"""Get the value associated with ``key``, or ``default`` if the
        key is missing, marking a hit as most-recently-used.

        Args:
            key: The key to look up.
            default: The value returned if ``key`` is missing.

        Returns:
            The value associated with ``key``, or ``default`` if
                ``key`` is missing.
        """
        if key not in self._data:
            return default
        return self[key]

    def pop(self, key: K, *args: V) -> V:
        r"""Remove and return the value associated with ``key``.

        Args:
            key: The key to remove.
            *args: An optional default value returned instead of raising
                ``KeyError`` when ``key`` is missing, mirroring
                ``dict.pop``.

        Returns:
            The value that was associated with ``key``, or the default
                value if ``key`` is missing.

        Raises:
            KeyError: if ``key`` is missing and no default is given.
        """
        return self._data.pop(key, *args)

    def clear(self) -> None:
        r"""Remove all entries from the cache."""
        self._data.clear()

    def keys(self) -> KeysView[K]:
        r"""Return a view of the cache's keys, in least- to most-
        recently-used order.

        Returns:
            A view of the keys.
        """
        return self._data.keys()

    def values(self) -> ValuesView[V]:
        r"""Return a view of the cache's values, in least- to most-
        recently-used order.

        Returns:
            A view of the values.
        """
        return self._data.values()

    def items(self) -> ItemsView[K, V]:
        r"""Return a view of the cache's ``(key, value)`` pairs, in
        least- to most-recently-used order.

        Returns:
            A view of the ``(key, value)`` pairs.
        """
        return self._data.items()
