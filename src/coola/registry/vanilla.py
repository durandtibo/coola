r"""Define a thread-safe registry."""

from __future__ import annotations

__all__ = ["Registry"]

import threading
from typing import TYPE_CHECKING, Generic, TypeVar

from coola.comparison import objects_are_equal
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class Registry(Generic[K, V]):
    r"""A thread-safe generic key-value registry for storing and managing
    typed mappings.


    The Registry class provides a type-safe container for registering and
    retrieving values by key. It supports all standard dictionary operations
    through operator overloading and provides additional methods for safe
    registration and querying. All operations are protected by a lock to
    ensure thread safety in concurrent environments.

    Args:
        initial_state: An optional dictionary to initialize the registry with.
            If provided, a copy is made to prevent external modifications.
            Defaults to None, which creates an empty registry.

    Attributes:
        _state: Internal dictionary storing the key-value pairs.
        _lock: Threading lock for synchronizing access to the registry.

    Example:
        Basic usage with registration and retrieval:

        ```pycon
        >>> from coola.registry import Registry
        >>> registry = Registry[str, int]()
        >>> registry.register("key1", 42)
        >>> registry.get("key1")
        42
        >>> registry
        Registry(
          (key1): 42
        )

        ```

        Using dictionary-style operations:

        ```pycon
        >>> from coola.registry import Registry
        >>> registry = Registry[str, int]()
        >>> registry["key2"] = 100
        >>> "key2" in registry
        True
        >>> del registry["key2"]

        ```

        Initializing with existing data:

        ```pycon
        >>> from coola.registry import Registry
        >>> registry = Registry[str, int](initial_state={"a": 1, "b": 2})
        >>> len(registry)
        2

        ```
    """

    def __init__(self, initial_state: dict[K, V] | None = None) -> None:
        self._state: dict[K, V] = initial_state.copy() if initial_state else {}
        self._lock: threading.RLock = threading.RLock()  # RLock allows re-entrant locking

    def __contains__(self, key: K) -> bool:
        return self.has(key)

    def __getitem__(self, key: K) -> V:
        with self._lock:
            if key not in self._state:
                msg = f"Key '{key}' is not registered"
                raise KeyError(msg)
            return self._state[key]

    def __setitem__(self, key: K, value: V) -> None:
        self.register(key, value, exist_ok=True)

    def __delitem__(self, key: K) -> None:
        self.unregister(key)

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            return iter(self._state.copy())

    def __len__(self) -> int:
        with self._lock:
            return len(self._state)

    def __repr__(self) -> str:
        with self._lock:
            snapshot = self._state.copy()
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(snapshot))}\n)"

    def __str__(self) -> str:
        with self._lock:
            snapshot = self._state.copy()
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(snapshot))}\n)"

    def clear(self) -> None:
        """Remove all entries from the registry.

        This method empties the registry, leaving it in the same state as a
        newly created empty registry. This operation cannot be undone.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]({"key1": 42, "key2": 100})
            >>> len(registry)
            2
            >>> registry.clear()
            >>> len(registry)
            0
            >>> registry.has("key1")
            False

            ```
        """
        with self._lock:
            self._state.clear()

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry1 = Registry[str, int]({"key1": 42, "key2": 100})
            >>> registry2 = Registry[str, int]({"key1": 42, "key2": 100})
            >>> registry3 = Registry[str, int]({"key1": 42})
            >>> registry1.equal(registry2)
            True
            >>> registry1.equal(registry3)
            False

            ```
        """
        if type(other) is not type(self):
            return False

        # Acquire locks in a consistent order based on object id to avoid deadlock
        first, second = (self, other) if id(self) < id(other) else (other, self)
        with first._lock, second._lock:
            return objects_are_equal(self._state, other._state, equal_nan=equal_nan)

    def get(self, key: K, default: V | None = None) -> V | None:
        """Retrieve the value associated with a key.

        This method performs a lookup in the registry and returns the
        corresponding value.

        Args:
            key: The key whose value should be retrieved.
            default: Value to return if the key does not exist.

        Returns:
            The value associated with the specified key.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]({"key1": 42, "key2": 100})
            >>> registry.get("key1")
            42
            >>> registry.get("missing")
            None

            ```
        """
        with self._lock:
            return self._state.get(key, default)

    def has(self, key: K) -> bool:
        """Check whether a key is registered in the registry.

        This method provides a safe way to test for key existence without
        risking a KeyError exception.

        Args:
            key: The key to check for existence.

        Returns:
            True if the key exists in the registry, False otherwise.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]({"key1": 42, "key2": 100})
            >>> registry.has("key1")
            True
            >>> registry.has("missing")
            False

            ```
        """
        with self._lock:
            return key in self._state

    def register(self, key: K, value: V, exist_ok: bool = False) -> None:
        """Register a new key-value pair in the registry.

        By default, this method raises an error if you try to register a key
        that already exists. This prevents accidental overwriting of values.
        Set exist_ok=True to allow overwriting.

        Args:
            key: The key to register. Must be hashable.
            value: The value to associate with the key.
            exist_ok: Controls behavior when the key already exists.
                If False (default), raises RuntimeError for duplicate keys.
                If True, silently overwrites the existing value.

        Raises:
            RuntimeError: If the key is already registered and exist_ok is False.
                The error message provides guidance on how to resolve the conflict.

        Example:
            Basic registration:

            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]()
            >>> registry.register("key1", 42)
            >>> registry.get("key1")
            42

            ```

            Attempting to register a duplicate key:

            ```pycon
            >>> registry.register("key1", 100)  # doctest: +SKIP
            RuntimeError: A value is already registered for 'key1'...

            ```

            Overwriting with exist_ok:

            ```pycon
            >>> registry.register("key1", 100, exist_ok=True)
            >>> registry.get("key1")
            100

            ```
        """
        with self._lock:
            if key in self._state and not exist_ok:
                msg = (
                    f"A value is already registered for '{key}'. "
                    "Use a different key or set exist_ok=True to override."
                )
                raise RuntimeError(msg)
            self._state[key] = value

    def register_many(self, mapping: Mapping[K, V], exist_ok: bool = False) -> None:
        """Register multiple key-value pairs in a single operation.

        This is a convenience method for bulk registration. It iterates through
        the provided mapping and registers each key-value pair. All registrations
        follow the same exist_ok policy. The operation is atomic when exist_ok
        is False - if any key already exists, no changes are made.

        Args:
            mapping: A dictionary or mapping containing the key-value pairs
                to register. The keys and values must match the registry's
                type parameters.
            exist_ok: Controls behavior when any key already exists.
                If False (default), raises error on the first duplicate key.
                If True, overwrites all existing values without error.

        Raises:
            RuntimeError: If exist_ok is False and any key in the mapping
                is already registered. The error occurs on the first duplicate
                encountered, and no partial registration occurs.

        Example:
            Registering multiple entries at once:

            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]()
            >>> registry.register_many({"key1": 42, "key2": 100, "key3": 7})
            >>> registry.get("key1")
            42
            >>> registry.get("key2")
            100
            >>> len(registry)
            3

            ```

            Bulk update with exist_ok:

            ```pycon
            >>> registry.register_many({"key1": 1, "key4": 4}, exist_ok=True)
            >>> registry.get("key1")
            1
            >>> registry.get("key4")
            4

            ```
        """
        with self._lock:
            # Check all keys first if exist_ok is False
            if not exist_ok and (duplicates := set(mapping) & set(self._state)):
                msg = (
                    f"Keys already registered: {', '.join(map(str, duplicates))}. "
                    "Use different keys or set exist_ok=True to override."
                )
                raise RuntimeError(msg)
            self._state.update(mapping)

    def unregister(self, key: K) -> V:
        """Remove a key-value pair from the registry and return the
        value.

        This method removes the specified key from the registry and returns
        the value that was associated with it. This allows you to retrieve
        the value one last time before it's removed.

        Args:
            key: The key to unregister and remove from the registry.

        Returns:
            The value that was associated with the key before removal.

        Raises:
            KeyError: If the key is not registered. The error message
                includes the key that was not found.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]({"key1": 42, "key2": 100})
            >>> registry.has("key1")
            True
            >>> value = registry.unregister("key1")
            >>> value
            42
            >>> registry.has("key1")
            False

            ```
        """
        with self._lock:
            if key not in self._state:
                msg = f"Key '{key}' is not registered"
                raise KeyError(msg)
            return self._state.pop(key)

    def items(self) -> ItemsView[K, V]:
        """Return key-value pairs.

        Returns:
            The key-value pairs.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int](initial_state={"a": 1, "b": 2})
            >>> registry.items()
            dict_items([('a', 1), ('b', 2)])

            ```
        """
        with self._lock:
            return self._state.copy().items()

    def keys(self) -> KeysView[K]:
        """Return registered keys.

        Returns:
            The registered keys.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int](initial_state={"a": 1, "b": 2})
            >>> registry.keys()
            dict_keys(['a', 'b'])

            ```
        """
        with self._lock:
            return self._state.copy().keys()

    def values(self) -> ValuesView[V]:
        """Return registered values.

        Returns:
            The registered values.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int](initial_state={"a": 1, "b": 2})
            >>> registry.values()
            dict_values([1, 2])

            ```
        """
        with self._lock:
            return self._state.copy().values()
