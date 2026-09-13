r"""Define a thread-safe type-based registry with an internal cache."""

from __future__ import annotations

__all__ = ["TypeRegistry"]

from typing import Generic, TypeVar

from coola.registry.base import BaseRegistry
from coola.utils.lru import LRUCache

T = TypeVar("T")

#: Maximum number of entries kept in the ``resolve()`` LRU cache.
_MAX_CACHE_SIZE = 1024


class TypeRegistry(BaseRegistry[type, T], Generic[T]):
    r"""A thread-safe type-based registry for storing and retrieving
    values.

    The TypeRegistry class provides a thread-safe container for mapping Python
    types to values. It supports standard dictionary operations through operator
    overloading and provides methods for safe registration and querying.

    The registry uses the Method Resolution Order (MRO) for type lookup through
    the `resolve()` method. When resolving a type, it automatically selects the
    most specific registered type, walking up the inheritance hierarchy if needed.
    This makes it ideal for type-based dispatching systems.

    The registry includes an internal LRU (least-recently-used) cache for
    type resolution, bounded to ``_MAX_CACHE_SIZE`` (1024) entries, to
    optimize performance when repeatedly resolving the same types without
    growing unbounded.

    Args:
        initial_state: An optional dictionary to initialize the registry with.
            If provided, a copy is made to prevent external modifications.
            Defaults to None, which creates an empty registry.

    Attributes:
        _state: Internal dictionary storing the type-value pairs.
        _cache: Bounded LRU cache of type resolution lookups for performance.
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
        super().__init__(initial_state=initial_state)
        # bounded LRU cache for type lookups - improves performance for
        # repeated transforms without growing unbounded
        self._cache: LRUCache[type, T] = LRUCache(maxsize=_MAX_CACHE_SIZE)

    def _on_change(self) -> None:
        # Clear cache when registry changes to ensure new registrations are used
        self._cache.clear()

    def _not_registered_msg(self, key: type) -> str:
        return f"Type '{key}' is not registered"

    def _already_registered_msg(self, key: type) -> str:
        return (
            f"A value is already registered for {key}. "
            "Use a different type or set exist_ok=True to override."
        )

    def _many_already_registered_msg(self, duplicates: set[type]) -> str:
        return (
            f"Types already registered: {', '.join(map(str, duplicates))}. "
            "Use different types or set exist_ok=True to override."
        )

    def resolve(self, dtype: type) -> T:
        r"""Resolve a type to its associated value using MRO lookup.

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
            if dtype in self._cache:
                return self._cache[dtype]
            value = self._resolve_uncached(dtype)
            self._cache[dtype] = value
            return value

    def _resolve_uncached(self, dtype: type) -> T:
        r"""Find value using MRO lookup (uncached version).

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

        raise KeyError(self._not_registered_msg(dtype))
