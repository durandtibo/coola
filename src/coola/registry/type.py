r"""Define a thread-safe type-based registry with an internal cache."""

from __future__ import annotations

__all__ = ["TypeRegistry"]

import threading
from typing import TYPE_CHECKING, Generic, TypeVar

from coola.comparison import objects_are_equal
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView

T = TypeVar("T")


class TypeRegistry(Generic[T]):
    r"""A thread-safe type-based registry for storing and retrieving
    values.

    The TypeRegistry class provides a thread-safe container for mapping Python
    types to values. It supports standard dictionary operations through operator
    overloading and provides methods for safe registration and querying.

    The registry uses the Method Resolution Order (MRO) for type lookup through
    the `resolve()` method. When resolving a type, it automatically selects the
    most specific registered type, walking up the inheritance hierarchy if needed.
    This makes it ideal for type-based dispatching systems.

    The registry includes an internal cache for type resolution to optimize
    performance when repeatedly resolving the same types.

    Args:
        initial_state: An optional dictionary to initialize the registry with.
            If provided, a copy is made to prevent external modifications.
            Defaults to None, which creates an empty registry.

    Attributes:
        _state: Internal dictionary storing the type-value pairs.
        _cache: Cached version of type resolution lookups for performance.
        _lock: Threading lock for synchronizing access to both state and cache.

    Example:
        Basic usage with registration and retrieval:

        ```pycon
        >>> from coola.registry import TypeRegistry
        >>> registry = TypeRegistry[str]()
        >>> registry.register(int, "I am an integer")
        >>> registry.get(int)
        'I am an integer'
        >>> registry
        TypeRegistry(
          (<class 'int'>): I am an integer
        )

        ```

        Using dictionary-style operations:

        ```pycon
        >>> from coola.registry import TypeRegistry
        >>> registry = TypeRegistry[str]()
        >>> registry[str] = "I am a string"
        >>> str in registry
        True
        >>> registry[str]
        'I am a string'
        >>> del registry[str]
        >>> str in registry
        False

        ```

        Initializing with existing data:

        ```pycon
        >>> from coola.registry import TypeRegistry
        >>> registry = TypeRegistry[int](initial_state={str: 100, float: 200})
        >>> len(registry)
        2
        >>> registry.get(str)
        100

        ```

        Using resolve() with inheritance (MRO lookup):

        ```pycon
        >>> from coola.registry import TypeRegistry
        >>> registry = TypeRegistry[str]()
        >>> registry.register(object, "I am an object")
        >>> registry.register(int, "I am an integer")
        >>> # Direct match
        >>> registry.resolve(int)
        'I am an integer'
        >>> # Falls back to parent type via MRO
        >>> registry.resolve(bool)  # bool inherits from int
        'I am an integer'
        >>> # Falls back to object
        >>> registry.resolve(str)
        'I am an object'

        ```

        Bulk registration:

        ```pycon
        >>> from coola.registry import TypeRegistry
        >>> registry = TypeRegistry[str]()
        >>> registry.register_many({int: "integer", float: "float", str: "string"})
        >>> len(registry)
        3
        >>> registry.get(float)
        'float'

        ```
    """

    def __init__(self, initial_state: dict[type, T] | None = None) -> None:
        self._state: dict[type, T] = initial_state.copy() if initial_state else {}
        # cache for type lookups - improves performance for repeated transforms
        self._cache: dict[type, T] = {}

        self._lock: threading.RLock = threading.RLock()  # RLock allows re-entrant locking

    def __contains__(self, dtype: type) -> bool:
        return self.has(dtype)

    def __getitem__(self, dtype: type) -> T:
        with self._lock:
            if dtype not in self._state:
                msg = f"Type '{dtype}' is not registered"
                raise KeyError(msg)
            return self._state[dtype]

    def __setitem__(self, dtype: type, value: T) -> None:
        self.register(dtype, value, exist_ok=True)

    def __delitem__(self, dtype: type) -> None:
        self.unregister(dtype)

    def __iter__(self) -> Iterator[type]:
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
        with self._lock:
            self._state.clear()
            self._cache.clear()

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        r"""Indicate if two registries are equal.

        Two registries are considered equal if they contain the same type-value
        mappings. The comparison is order-independent since dictionaries are
        unordered collections.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN`` values will be
                considered equal when comparing values.

        Returns:
            ``True`` if the two registries are equal, otherwise ``False``.

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

        # Acquire locks in a consistent order based on object id to avoid deadlock
        first, second = (self, other) if id(self) < id(other) else (other, self)
        with first._lock, second._lock:
            return objects_are_equal(self._state, other._state, equal_nan=equal_nan)

    def get(self, dtype: type, default: T | None = None) -> T:
        """Retrieve the value associated with a type.

        This method performs a direct lookup in the registry and returns the
        corresponding value. If the type doesn't exist, a KeyError is raised
        with a descriptive message. For inheritance-based lookup, use the
        `resolve()` method instead.

        Args:
            dtype: The type whose value should be retrieved.
            default: Value to return if the key does not exist.

        Returns:
            The value associated with the specified type.

        Example:
            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[int]()
            >>> registry.register(str, 42)
            >>> registry.get(str)
            42
            >>> registry.get(int)  # doctest: +SKIP
            KeyError: "Key <class 'int'> is not registered"

            ```
        """
        with self._lock:
            return self._state.get(dtype, default)

    def has(self, dtype: type) -> bool:
        """Check whether a type is registered in the registry.

        This method provides a safe way to test for type existence without
        risking a KeyError exception. It only checks for direct type matches;
        it does not check parent types via MRO.

        Args:
            dtype: The type to check for existence.

        Returns:
            True if the type exists in the registry, False otherwise.

        Example:
            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[int]()
            >>> registry.register(str, 42)
            >>> registry.has(str)
            True
            >>> registry.has(int)
            False

            ```
        """
        with self._lock:
            return dtype in self._state

    def register(self, dtype: type, value: T, exist_ok: bool = False) -> None:
        """Register a new type-value pair in the registry.

        By default, this method raises an error if you try to register a type
        that already exists. This prevents accidental overwriting of values.
        Set exist_ok=True to allow overwriting.

        Registering a new type clears the internal resolution cache to ensure
        that subsequent `resolve()` calls use the updated registry state.

        Args:
            dtype: The type to register. Must be a Python type object.
            value: The value to associate with the type.
            exist_ok: Controls behavior when the type already exists.
                If False (default), raises RuntimeError for duplicate types.
                If True, silently overwrites the existing value.

        Raises:
            RuntimeError: If the type is already registered and exist_ok is False.
                The error message provides guidance on how to resolve the conflict.

        Example:
            Basic registration:

            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[int]()
            >>> registry.register(str, 42)
            >>> registry.get(str)
            42

            ```

            Attempting to register a duplicate type:

            ```pycon
            >>> registry.register(str, 100)  # doctest: +SKIP
            RuntimeError: A value is already registered for '<class 'str'>'...

            ```

            Overwriting with exist_ok:

            ```pycon
            >>> registry.register(str, 100, exist_ok=True)
            >>> registry.get(str)
            100

            ```
        """
        with self._lock:
            if dtype in self._state and not exist_ok:
                msg = (
                    f"A value is already registered for {dtype}. "
                    "Use a different type or set exist_ok=True to override."
                )
                raise RuntimeError(msg)
            self._state[dtype] = value
            # Clear cache when registry changes to ensure new registrations are used
            self._cache.clear()

    def register_many(self, mapping: Mapping[type, T], exist_ok: bool = False) -> None:
        """Register multiple type-value pairs in a single operation.

        This is a convenience method for bulk registration. It iterates through
        the provided mapping and registers each type-value pair. All registrations
        follow the same exist_ok policy. The operation is atomic when exist_ok
        is False - if any type already exists, no changes are made.

        Registering new types clears the internal resolution cache to ensure
        that subsequent `resolve()` calls use the updated registry state.

        Args:
            mapping: A dictionary or mapping containing the type-value pairs
                to register. The keys must be Python type objects and values
                must match the registry's type parameter.
            exist_ok: Controls behavior when any type already exists.
                If False (default), raises error on the first duplicate type.
                If True, overwrites all existing values without error.

        Raises:
            RuntimeError: If exist_ok is False and any type in the mapping
                is already registered. The error occurs before any registration
                is performed, ensuring no partial updates.

        Example:
            Registering multiple entries at once:

            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[int]()
            >>> registry.register_many({str: 42, float: 100, list: 7})
            >>> registry.get(str)
            42
            >>> registry.get(float)
            100
            >>> len(registry)
            3

            ```

            Bulk update with exist_ok:

            ```pycon
            >>> registry.register_many({str: 1, dict: 4}, exist_ok=True)
            >>> registry.get(str)
            1
            >>> registry.get(dict)
            4

            ```
        """
        with self._lock:
            # Check all keys first if exist_ok is False
            if not exist_ok and (duplicates := set(mapping) & set(self._state)):
                msg = (
                    f"Types already registered: {', '.join(map(str, duplicates))}. "
                    "Use different types or set exist_ok=True to override."
                )
                raise RuntimeError(msg)
            self._state.update(mapping)
            # Clear cache when registry changes to ensure new registrations are used
            self._cache.clear()

    def resolve(self, dtype: type) -> T:
        """Resolve a type to its associated value using MRO lookup.

        This method finds the most appropriate value for a given type by
        walking the Method Resolution Order (MRO). It first checks for a
        direct match, then searches through parent types in MRO order to
        find the most specific registered type.

        Results are cached internally to optimize performance for repeated
        lookups of the same type.

        Args:
            dtype: The type to resolve.

        Returns:
            The value associated with the type or its nearest registered
            parent type in the MRO.

        Raises:
            KeyError: If no matching type is found in the registry, including
                parent types in the MRO.

        Example:
            Basic resolution with inheritance:

            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]()
            >>> registry.register(object, "base")
            >>> registry.register(int, "integer")
            >>> # Direct match
            >>> registry.resolve(int)
            'integer'
            >>> # bool inherits from int, so resolves to int's value
            >>> registry.resolve(bool)
            'integer'
            >>> # str inherits from object, so resolves to object's value
            >>> registry.resolve(str)
            'base'

            ```

            Resolution with custom classes:

            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> class Animal:
            ...     pass
            ...
            >>> class Dog(Animal):
            ...     pass
            ...
            >>> class Poodle(Dog):
            ...     pass
            ...
            >>> registry = TypeRegistry[str]()
            >>> registry.register(Animal, "animal")
            >>> registry.register(Dog, "dog")
            >>> registry.resolve(Dog)
            'dog'
            >>> registry.resolve(Poodle)  # Resolves to parent Dog
            'dog'

            ```
        """
        with self._lock:
            if dtype not in self._cache:
                self._cache[dtype] = self._resolve_uncached(dtype)
            return self._cache[dtype]

    def unregister(self, dtype: type) -> T:
        """Remove a type-value pair from the registry and return the
        value.

        This method removes the specified type from the registry and returns
        the value that was associated with it. This allows you to retrieve
        the value one last time before it's removed.

        Unregistering a type clears the internal resolution cache to ensure
        that subsequent `resolve()` calls reflect the updated registry state.

        Args:
            dtype: The type to unregister and remove from the registry.

        Returns:
            The value that was associated with the type before removal.

        Raises:
            KeyError: If the type is not registered. The error message
                includes the type that was not found.

        Example:
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
        with self._lock:
            if dtype not in self._state:
                msg = f"Type {dtype} is not registered"
                raise KeyError(msg)
            # Clear cache when registry changes to ensure new registrations are used
            self._cache.clear()
            return self._state.pop(dtype)

    def items(self) -> ItemsView[type, T]:
        """Return key-value pairs.

        Returns:
            The key-value pairs.

        Example:
            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]({int: "I am an integer", float: "I am a float"})
            >>> registry.items()
            dict_items([(<class 'int'>, 'I am an integer'), (<class 'float'>, 'I am a float')])

            ```
        """
        with self._lock:
            return self._state.copy().items()

    def keys(self) -> KeysView[type]:
        """Return registered keys.

        Returns:
            The registered keys.

        Example:
            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]({int: "I am an integer", float: "I am a float"})
            >>> registry.keys()
            dict_keys([<class 'int'>, <class 'float'>])

            ```
        """
        with self._lock:
            return self._state.copy().keys()

    def values(self) -> ValuesView[T]:
        """Return registered values.

        Returns:
            The registered values.

        Example:
            ```pycon
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]({int: "I am an integer", float: "I am a float"})
            >>> registry.values()
            dict_values(['I am an integer', 'I am a float'])

            ```
        """
        with self._lock:
            return self._state.copy().values()

    def _resolve_uncached(self, dtype: type) -> T:
        """Find value using MRO lookup (uncached version).

        This is the internal implementation that performs the actual type
        resolution. It first checks for a direct match, then walks the MRO
        to find the most specific registered parent type.

        This method should only be called while holding self._lock.

        Args:
            dtype: The type to find a value for.

        Returns:
            The appropriate value for the type or its nearest parent type.

        Raises:
            KeyError: If no matching type is found in the registry.
        """
        # Direct lookup first (most common case, O(1))
        if dtype in self._state:
            return self._state[dtype]

        # MRO lookup for inheritance - finds the most specific parent type
        for base_type in dtype.__mro__:
            if base_type in self._state:
                return self._state[base_type]

        msg = "Could not find a registered type"
        raise KeyError(msg)
