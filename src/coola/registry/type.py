r"""Define a thread-safe type-based registry with an internal cache."""

from __future__ import annotations

__all__ = ["TypeRegistry"]

import threading
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from coola.comparison import objects_are_equal
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping

T = TypeVar("T")


class TypeRegistry(Generic[T]):
    r"""A thread-safe generic key-value registry for storing and managing
    typed mappings.

    The Registry class provides a type-safe container for registering and
    retrieving values by key. It supports all standard dictionary operations
    through operator overloading and provides additional methods for safe
    registration and querying. All operations are protected by a lock to
    ensure thread safety in concurrent environments.

    This registry maintains a mapping from Python types to transformer instances
    and uses the Method Resolution Order (MRO) for type lookup. When transforming
    data, it automatically selects the most specific registered transformer for
    the data's type, falling back to parent types or a default transformer if needed.

    The registry includes an LRU cache for type lookups to optimize performance
    in applications that repeatedly transform similar data structures.

    Args:
        initial_state: An optional dictionary to initialize the registry with.
            If provided, a copy is made to prevent external modifications.
            Defaults to None, which creates an empty registry.

    Attributes:
        _state: Internal dictionary storing the key-value pairs.
        _cache: Cached version of the lookup to speed up.
        _lock_state: Threading lock for synchronizing access to the state.
        _lock_cache: Threading lock for synchronizing access to the cache.

    Examples:
        Basic usage with registration and retrieval:

        ```pycon
        >>> from coola.registry import TypeRegistry
        >>> registry = TypeRegistry[str]()
        >>> registry.register(int, "I am a integer")
        >>> registry.get(int)
        'I am a integer'
        >>> registry
        TypeRegistry(
          (<class 'int'>): I am a integer
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

    def __init__(self, initial_state: dict[type, T] | None = None) -> None:
        self._state: dict[type, T] = initial_state.copy() if initial_state else {}
        # cache for type lookups - improves performance for repeated transforms
        self._cache: dict[type, T] = {}

        self._lock_state = threading.RLock()  # RLock allows re-entrant locking
        self._lock_cache = threading.RLock()  # RLock allows re-entrant locking

    def __contains__(self, dtype: type) -> bool:
        return self.has(dtype)

    def __getitem__(self, dtype: type) -> T:
        return self.get(dtype)

    def __setitem__(self, dtype: type, value: T) -> None:
        self.register(dtype, value, exist_ok=True)

    def __delitem__(self, dtype: type) -> None:
        self.unregister(dtype)

    def __len__(self) -> int:
        with self._lock_state:
            return len(self._state)

    def __repr__(self) -> str:
        with self._lock_state:
            snapshot = self._state.copy()
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(snapshot))}\n)"

    def __str__(self) -> str:
        with self._lock_state:
            snapshot = self._state.copy()
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(snapshot))}\n)"

    def clear(self) -> None:
        """Remove all entries from the registry.

        This method empties the registry, leaving it in the same state as a
        newly created empty registry. This operation cannot be undone.

        Examples:
            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]({int: "I am an integer", float: "I am a float"})
            >>> len(registry)
            2
            >>> registry.clear()
            >>> len(registry)
            0
            >>> registry.has(int)
            False
            >>> registry.has(float)
            False

            ```
        """
        with self._lock_state, self._lock_cache:
            self._state.clear()
            self._cache.clear()

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> registry1 = TypeRegistry[str]({int: "I am an integer", float: "I am a float"})
            >>> registry2 = TypeRegistry[str]({int: "I am an integer", float: "I am a float"})
            >>> registry3 = TypeRegistry[str]({int: "I am an integer"})
            >>> registry1.equal(registry2)
            True
            >>> registry1.equal(registry3)
            False

            ```
        """
        if type(other) is not type(self):
            return False
        with self._lock_state, other._lock_state:
            return objects_are_equal(self._state, other._state, equal_nan=equal_nan)

    def get(self, key: type) -> T:
        """Retrieve the value associated with a key.

        This method performs a lookup in the registry and returns the
        corresponding value. If the key doesn't exist, a KeyError is raised
        with a descriptive message.

        Args:
            key: The key whose value should be retrieved.

        Returns:
            The value associated with the specified key.

        Raises:
            KeyError: If the key has not been registered. The error message
                includes the key that was not found.

        Examples:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]()
            >>> registry.register("key1", 42)
            >>> registry.get("key1")
            42
            >>> registry.get("missing")  # doctest: +SKIP
            KeyError: "Key 'missing' is not registered"

            ```
        """
        with self._lock_state:
            if key not in self._state:
                msg = f"Key '{key}' is not registered"
                raise KeyError(msg)
            return self._state[key]

    def has(self, dtype: type) -> bool:
        """Check whether a data type is registered in the registry.

        This method provides a safe way to test for data type existence without
        risking a KeyError exception.

        Args:
            dtype: The data type to check for existence.

        Returns:
            True if the dtype exists in the registry, False otherwise.

        Examples:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]()
            >>> registry.register("dtype1", 42)
            >>> registry.has("dtype1")
            True
            >>> registry.has("missing")
            False

            ```
        """
        with self._lock_state:
            return dtype in self._state

    def register(self, key: type, value: T, exist_ok: bool = False) -> None:
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

        Examples:
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
        with self._lock_state, self._lock_cache:
            if key in self._state and not exist_ok:
                msg = (
                    f"A value is already registered for '{key}'. "
                    "Use a different key or set exist_ok=True to override."
                )
                raise RuntimeError(msg)
            self._state[key] = value
            # Clear cache when registry changes to ensure new registrations are used
            self._cache.clear()

    def register_many(self, mapping: Mapping[type, T], exist_ok: bool = False) -> None:
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

        Examples:
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
        with self._lock_state, self._lock_cache:
            # Check all keys first if exist_ok is False
            if not exist_ok and (duplicates := set(mapping) & set(self._state)):
                msg = (
                    f"Types already registered: {', '.join(map(str, duplicates))}"
                    "Use different types or set exist_ok=True to override."
                )
                raise RuntimeError(msg)
            self._state.update(mapping)
            # Clear cache when registry changes to ensure new registrations are used
            self._cache.clear()

    def resolve(self, dtype: type) -> T:
        with self._lock_cache:
            if dtype not in self._cache:
                self._cache[dtype] = self._resolve_uncached(dtype)
            return self._cache[dtype]

    def unregister(self, dtype: type) -> T:
        """Remove a data type-value pair from the registry and return
        the value.

        This method removes the specified data type from the registry and returns
        the value that was associated with it. This allows you to retrieve
        the value one last time before it's removed.

        Args:
            dtype: The data type to unregister and remove from the registry.

        Returns:
            The value that was associated with the data type before removal.

        Raises:
            KeyError: If the data type is not registered. The error message
                includes the data type that was not found.

        Examples:
            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]({int: "I am an integer"})
            >>> registry.has(int)
            True
            >>> value = registry.unregister(int)
            >>> value
            'I am an integer'
            >>> registry.has(int)
            False

            ```
        """
        with self._lock_state, self._lock_cache:
            if dtype not in self._state:
                msg = f"Type '{dtype}' is not registered"
                raise KeyError(msg)
            # Clear cache when registry changes to ensure new registrations are used
            self._cache.clear()
            return self._state.pop(dtype)

    def _resolve_uncached(self, dtype: type) -> T:
        """Find value using MRO (uncached version).

        This is the internal implementation that performs the actual lookup.
        It first checks for a direct match, then walks the MRO to find the
        most specific registered transformer, and finally falls back to the
        default transformer.

        Args:
            dtype: The type to find a transformer for

        Returns:
            The appropriate transformer instance
        """
        with self._lock_state:
            # Direct lookup first (most common case, O(1))
            if dtype in self._state:
                return self._state[dtype]

            # MRO lookup for inheritance - finds the most specific parent type
            for base_type in dtype.__mro__:
                if base_type in self._state:
                    return self._state[base_type]

        msg = "Could not find a registered type"
        raise KeyError(msg)
