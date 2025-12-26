r"""Define the iterator registry for managing iterators based on data
types."""

from __future__ import annotations

__all__ = ["IteratorRegistry"]

from typing import TYPE_CHECKING, Any

from coola.iterator.dfs.default import DefaultIterator
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from coola.iterator.dfs.base import BaseIterator


class IteratorRegistry:
    r"""Registry that manages iterators for different data types.

    This registry stores iterators for various data types and handles
    the dispatching of the appropriate iterator based on the data
    type during iteration. It uses Method Resolution Order (MRO) to
    resolve the most specific iterator for a given data type.
    It also supports caching of iterators for performance optimization
    in repetitive iteration tasks.

    Args:
        registry: An optional dictionary mapping types to iterators.
            If provided, the registry is initialized with this mapping.

    Attributes:
        _registry: Internal mapping of registered data types to iterators.
        _default_iterator: The fallback iterator used for types not explicitly registered.
        _iterator_cache: Cache to speed up iterator lookups.

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
        r"""Register an iterator for a given data type.

        This method associates a specific iterator with a type.
        If data of this type is iterated, the registered iterator
        will be used. The cache is cleared after a registration to
        ensure consistency.

        Args:
            data_type: The Python type to register (e.g., `list`, `dict`, custom types).
            iterator: The iterator instance that handles this type.
            exist_ok: If `True`, allows overwriting an existing registration.
                If `False`, raises an error.

        Raises:
            RuntimeError: If the type is already registered and `exist_ok` is `False`.

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
                f"Iterator {self._registry[data_type]} already registered for {data_type}. "
                f"Use exist_ok=True to overwrite."
            )
            raise RuntimeError(msg)
        self._registry[data_type] = iterator
        self._iterator_cache.clear()

    def register_many(
        self,
        mapping: Mapping[type, BaseIterator[Any]],
        exist_ok: bool = False,
    ) -> None:
        r"""Register multiple iterators at once.

        This method allows for bulk registration of iterators for multiple data types.

        Args:
            mapping: A dictionary mapping Python types to their respective iterators.
            exist_ok: If `True`, allows overwriting existing registrations.

        Raises:
            RuntimeError: If any type is already registered and `exist_ok` is `False`.

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
        r"""Check if an iterator is registered for a given data type.

        This method checks for direct registration. Even if this method returns `False`,
        a suitable iterator might still be found using the MRO lookup.

        Args:
            data_type: The type to check.

        Returns:
            `True` if an iterator is registered for the type, `False` otherwise.

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
        r"""Find the iterator for a data type without using cache.

        This method looks up the most specific iterator for the given type, starting
        with direct matches and then walking up the MRO to find an appropriate iterator.

        Args:
            data_type: The data type for which to find an iterator.

        Returns:
            The matching iterator instance for the type or a fallback iterator.
        """
        if data_type in self._registry:
            return self._registry[data_type]

        for base_type in data_type.__mro__:
            if base_type in self._registry:
                return self._registry[base_type]

        return self._default_iterator

    def find_iterator(self, data_type: type) -> BaseIterator[Any]:
        r"""Find the appropriate iterator for a given type.

        This method uses the MRO to find the most specific iterator. It caches the result
        for performance, so subsequent lookups are faster.

        Args:
            data_type: The data type for which to find an iterator.

        Returns:
            The appropriate iterator for the data type.

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

    def iterate(self, data: Any) -> Iterator[Any]:
        r"""Perform depth-first iteration over a data structure.

        This method uses the appropriate iterator for the data type, which may be
        retrieved via the registry. The iterator will recursively traverse the data
        structure, yielding elements based on its specific implementation.

        Args:
            data: The data structure to iterate over.

        Yields:
            The elements of the data structure according to the
                appropriate iterator's traversal logic.

        Example:
            ```pycon
            >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator, MappingIterator
            >>> registry = IteratorRegistry({list: IterableIterator(), dict: MappingIterator()})
            >>> list(registry.iterate({"a": [1, 2], "b": [3, 4]}))
            [1, 2, 3, 4]

            ```
        """
        iterator = self.find_iterator(type(data))
        yield from iterator.iterate(data, self)
