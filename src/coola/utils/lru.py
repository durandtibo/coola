r"""Define a bounded, dict-like LRU (least-recently-used) cache.

Used by ``TypeRegistry`` (and other data structures that need a bounded
lookup cache) to cap memory usage while keeping the most recently used
entries around. This class is not thread-safe on its own — callers
that share an instance across threads (e.g. ``TypeRegistry``) are
expected to guard access with their own lock.
"""

from __future__ import annotations

__all__ = ["LRUCache"]

from collections import OrderedDict
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

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

    # Mutable and compares by content, like dict: not hashable.
    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            msg = f"maxsize must be greater than 0, but received {maxsize}"
            raise ValueError(msg)
        self._maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(maxsize={self._maxsize}, data={self._data})"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(maxsize={self._maxsize}, size={len(self._data)})"

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __getitem__(self, key: K) -> V:
        r"""Get the value associated with ``key`` and mark it as most-
        recently-used.

        Raises:
            KeyError: if ``key`` is not in the cache.
        """
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key: K, value: V) -> None:
        r"""Set ``value`` for ``key``, marking it as most-recently-used,
        and evict the least-recently-used entry if the cache is now over
        capacity."""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __delitem__(self, key: K) -> None:
        del self._data[key]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LRUCache):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented

    @property
    def maxsize(self) -> int:
        r"""The maximum number of entries the cache holds."""
        return self._maxsize

    def get(self, key: K, default: V | None = None) -> V | None:
        r"""Get the value associated with ``key``, or ``default`` if the
        key is missing, marking a hit as most-recently-used."""
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

        Raises:
            KeyError: if ``key`` is missing and no default is given.
        """
        return self._data.pop(key, *args)

    def clear(self) -> None:
        r"""Remove all entries from the cache."""
        self._data.clear()

    def keys(self) -> KeysView[K]:
        r"""Return a view of the cache's keys, in least- to most-
        recently-used order."""
        return self._data.keys()

    def values(self) -> ValuesView[V]:
        r"""Return a view of the cache's values, in least- to most-
        recently-used order."""
        return self._data.values()

    def items(self) -> ItemsView[K, V]:
        r"""Return a view of the cache's ``(key, value)`` pairs, in
        least- to most-recently-used order."""
        return self._data.items()
