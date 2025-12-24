r"""Define the iterator registry."""

from __future__ import annotations

__all__ = ["IteratorRegistry"]

from typing import TYPE_CHECKING, Any

from coola.iterator.dfs.default import DefaultIterator
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from coola.iterator.dfs.base import BaseIterator


class IteratorRegistry:
    """Registry that manages and dispatches iterators based on data
    type.

    This registry maintains a mapping from Python types to iterator instances
    and uses the Method Resolution Order (MRO) for type lookup. When iterating
    over data, it automatically selects the most specific registered iterator for
    the data's type, falling back to parent types or a default iterator if needed.

    The registry includes a cache for type lookups to optimize performance
    in applications that repeatedly iterate over similar data structures.

    Args:
        registry: Optional initial mapping of types to iterators. If provided,
            the registry is copied to prevent external mutations.

    Attributes:
        _registry: Internal mapping of registered types to iterators
        _default_iterator: Fallback iterator for unregistered types
        _iterator_cache: Cache for type lookups to improve performance

    Example:
        Basic usage:

        ```pycon
        >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator
        >>> registry = IteratorRegistry({list: IterableIterator()})
        >>> registry
        IteratorRegistry(
          (<class 'list'>): IterableIterator()
        )
        >>> list(registry.iterate([1, 2, 3]))
        [1, 2, 3]

        ```

        Working with nested structures:

        ```pycon
        >>> from coola.iterator.dfs import get_default_registry
        >>> registry = get_default_registry()
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> list(registry.iterate(data))
        [1, 2, 3, 4]

        ```
    """

    def __init__(self, registry: dict[type, BaseIterator[Any]] | None = None) -> None:
        self._registry: dict[type, BaseIterator[Any]] = registry.copy() if registry else {}
        self._default_iterator: BaseIterator[Any] = DefaultIterator()

        # Cache for type lookups - improves performance for repeated iterations
        self._iterator_cache: dict[type, BaseIterator[Any]] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(self._registry))}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self._registry))}\n)"

    def register(
        self,
        data_type: type,
        iterator: BaseIterator[Any],
        exist_ok: bool = False,
    ) -> None:
        """Register an iterator for a given data type.

        This method associates an iterator instance with a specific Python type.
        When data of this type is iterated over, the registered iterator will be used.
        The cache is automatically cleared after registration to ensure consistency.

        Args:
            data_type: The Python type to register (e.g., list, dict, custom classes)
            iterator: The iterator instance that handles this type
            exist_ok: If False (default), raises an error if the type is already
                registered. If True, overwrites the existing registration silently.

        Raises:
            RuntimeError: If the type is already registered and exist_ok is False

        Example:
        ```pycon
        >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator
        >>> registry = IteratorRegistry()
        >>> registry.register(list, IterableIterator())
        >>> registry.has_iterator(list)
        True

        ```
        """
        if data_type in self._registry and not exist_ok:
            msg = (
                f"Iterator {self._registry[data_type]} already registered "
                f"for {data_type}. Use exist_ok=True to overwrite."
            )
            raise RuntimeError(msg)
        self._registry[data_type] = iterator
        # Clear cache when registry changes to ensure new registrations are used
        self._iterator_cache.clear()

    def register_many(
        self,
        mapping: Mapping[type, BaseIterator[Any]],
        exist_ok: bool = False,
    ) -> None:
        """Register multiple iterators at once.

        This is a convenience method for bulk registration that internally calls
        register() for each type-iterator pair.

        Args:
            mapping: Dictionary mapping Python types to iterator instances
            exist_ok: If False (default), raises an error if any type is already
                registered. If True, overwrites existing registrations silently.

        Raises:
            RuntimeError: If any type is already registered and exist_ok is False

        Example:
        ```pycon
        >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator, MappingIterator
        >>> registry = IteratorRegistry()
        >>> registry.register_many({list: IterableIterator(), dict: MappingIterator()})
        >>> registry
        IteratorRegistry(
          (<class 'list'>): IterableIterator()
          (<class 'dict'>): MappingIterator()
        )

        ```
        """
        for typ, iterator in mapping.items():
            self.register(typ, iterator, exist_ok=exist_ok)

    def has_iterator(self, data_type: type) -> bool:
        """Check if an iterator is explicitly registered for the given
        type.

        Note that this only checks for direct registration. Even if this returns
        False, find_iterator() may still return an iterator via MRO lookup
        or the default iterator.

        Args:
            data_type: The type to check

        Returns:
            True if an iterator is explicitly registered for this type,
            False otherwise

        Example:
        ```pycon
        >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator
        >>> registry = IteratorRegistry({list: IterableIterator()})
        >>> registry.has_iterator(list)
        True
        >>> registry.has_iterator(tuple)
        False

        ```
        """
        return data_type in self._registry

    def _find_iterator_uncached(self, data_type: type) -> BaseIterator[Any]:
        """Find iterator using MRO (uncached version).

        This is the internal implementation that performs the actual lookup.
        It first checks for a direct match, then walks the MRO to find the
        most specific registered iterator, and finally falls back to the
        default iterator.

        Args:
            data_type: The type to find an iterator for

        Returns:
            The appropriate iterator instance
        """
        # Direct lookup first (most common case, O(1))
        if data_type in self._registry:
            return self._registry[data_type]

        # MRO lookup for inheritance - finds the most specific parent type
        for base_type in data_type.__mro__:
            if base_type in self._registry:
                return self._registry[base_type]

        # Fall back to default iterator for unregistered types
        return self._default_iterator

    def find_iterator(self, data_type: type) -> BaseIterator[Any]:
        """Find the appropriate iterator for a given type.

        Uses the Method Resolution Order (MRO) to find the most specific
        registered iterator. For example, if you register an iterator
        for Sequence but not for list, lists will use the Sequence iterator.

        Results are cached for performance, as iterator lookup is a hot path
        during iteration operations.

        Args:
            data_type: The Python type to find an iterator for

        Returns:
            The most specific registered iterator for this type, a parent
            type's iterator via MRO, or the default iterator

        Example:
        ```pycon
        >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator
        >>> registry = IteratorRegistry({list: IterableIterator()})
        >>> registry.find_iterator(list)
        IterableIterator()
        >>> registry.find_iterator(tuple)
        DefaultIterator()

        ```
        """
        if data_type not in self._iterator_cache:
            self._iterator_cache[data_type] = self._find_iterator_uncached(data_type)
        return self._iterator_cache[data_type]

    def iterate(self, data: Any) -> Generator[Any]:
        """Perform depth-first iteration over a data structure.

        This method finds the appropriate iterator for the data's type and
        delegates to it. The iterator will recursively traverse the data
        structure, yielding elements according to the iterator's implementation.

        Args:
            data: The data structure to iterate over. Can be any type with
                a registered iterator.

        Yields:
            Elements from the data structure as determined by the registered iterator.

        Example:
        ```pycon
        >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator, MappingIterator
        >>> registry = IteratorRegistry({list: IterableIterator(), dict: MappingIterator()})
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> list(registry.iterate(data))
        [1, 2, 3, 4]

        ```
        """
        iterator = self.find_iterator(type(data))
        yield from iterator.iterate(data, self)
