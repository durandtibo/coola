r"""Define the hasher registry for recursive data hashing.

This module provides a registry system that manages and dispatches
hashers based on data types, enabling recursive hashing of nested data
structures.
"""

from __future__ import annotations

__all__ = ["HasherRegistry"]

from typing import TYPE_CHECKING, Any

from coola.hashing.base import BaseHasher
from coola.registry import TypeRegistry
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping


class HasherRegistry:
    """Registry that manages and dispatches hashers based on data type.

    This registry maintains a mapping from Python types to hasher instances
    and uses the Method Resolution Order (MRO) for type lookup. When hashing
    data, it automatically selects the most specific registered hasher for
    the data's type, falling back to parent types or a default hasher if needed.

    Args:
        initial_state: Optional initial mapping of types to hashers.
            If provided, the state is copied to prevent external mutations.

    Example:
        Basic usage with a sequence hasher:

        ```pycon
        >>> from coola.hashing import HasherRegistry, SequenceHasher, StrHasher
        >>> registry = HasherRegistry({object: StrHasher(), list: SequenceHasher()})
        >>> registry
        HasherRegistry(
          (state): TypeRegistry(
              (<class 'object'>): StrHasher()
              (<class 'list'>): SequenceHasher()
            )
        )
        >>> registry.hash([1, 2, 3])

        ```

        Registering custom hashers:

        ```pycon
        >>> from coola.hashing import HasherRegistry, SequenceHasher, StrHasher
        >>> registry = HasherRegistry({object: StrHasher()})
        >>> registry.register(list, SequenceHasher())
        >>> registry.hash([1, 2, 3])

        ```

        Working with nested structures:

        ```pycon
        >>> from coola.hashing import get_default_registry
        >>> registry = get_default_registry()
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> registry.hash(data)

        ```
    """

    def __init__(self, initial_state: dict[type, BaseHasher[Any]] | None = None) -> None:
        self._state: TypeRegistry[BaseHasher] = TypeRegistry[BaseHasher](initial_state)

    def __repr__(self) -> str:
        state = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {state}\n)"

    def __str__(self) -> str:
        state = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {state}\n)"

    def register(
        self,
        data_type: type,
        hasher: BaseHasher[Any],
        exist_ok: bool = False,
    ) -> None:
        """Register a hasher for a given data type.

        Args:
            data_type: The Python type to register (e.g., ``list``, ``dict``,
                custom classes).
            hasher: The hasher instance that handles this type.
            exist_ok: If ``False`` (default), raises an error if the type is
                already registered. If ``True``, overwrites the existing
                registration silently.

        Raises:
            RuntimeError: If the type is already registered and ``exist_ok``
                is ``False``.

        Example:
            ```pycon
            >>> from coola.hashing import HasherRegistry, SequenceHasher
            >>> registry = HasherRegistry()
            >>> registry.register(list, SequenceHasher())
            >>> registry.has_hasher(list)
            True

            ```
        """
        self._state.register(data_type, hasher, exist_ok=exist_ok)

    def register_many(
        self,
        mapping: Mapping[type, BaseHasher[Any]],
        exist_ok: bool = False,
    ) -> None:
        """Register multiple hashers at once.

        This is a convenience method for bulk registration that internally
        calls ``register`` for each type-hasher pair.

        Args:
            mapping: Dictionary mapping Python types to hasher instances.
            exist_ok: If ``False`` (default), raises an error if any type is
                already registered. If ``True``, overwrites existing
                registrations silently.

        Raises:
            RuntimeError: If any type is already registered and ``exist_ok``
                is ``False``.

        Example:
            ```pycon
            >>> from coola.hashing import HasherRegistry, SequenceHasher, MappingHasher
            >>> registry = HasherRegistry()
            >>> registry.register_many(
            ...     {
            ...         list: SequenceHasher(),
            ...         dict: MappingHasher(),
            ...     }
            ... )
            >>> registry
            HasherRegistry(
              (state): TypeRegistry(
                  (<class 'list'>): SequenceHasher()
                  (<class 'dict'>): MappingHasher()
                )
            )

            ```
        """
        self._state.register_many(mapping, exist_ok=exist_ok)

    def has_hasher(self, data_type: type) -> bool:
        """Check if a hasher is explicitly registered for the given
        type.

        Note that this only checks for direct registration. Even if this
        returns ``False``, ``find_hasher`` may still return a hasher via
        MRO lookup or the default hasher.

        Args:
            data_type: The type to check.

        Returns:
            ``True`` if a hasher is explicitly registered for this type,
                ``False`` otherwise.

        Example:
            ```pycon
            >>> from coola.hashing import HasherRegistry, SequenceHasher
            >>> registry = HasherRegistry()
            >>> registry.register(list, SequenceHasher())
            >>> registry.has_hasher(list)
            True
            >>> registry.has_hasher(tuple)
            False

            ```
        """
        return data_type in self._state

    def find_hasher(self, data_type: type) -> BaseHasher[Any]:
        """Find the appropriate hasher for a given type.

        Uses the Method Resolution Order (MRO) to find the most specific
        registered hasher. For example, if a hasher is registered for
        ``Sequence`` but not for ``list``, lists will use the ``Sequence``
        hasher.

        Args:
            data_type: The Python type to find a hasher for.

        Returns:
            The most specific registered hasher for this type, resolved
            via MRO, or the default hasher if no match is found.

        Example:
            ```pycon
            >>> from collections.abc import Sequence
            >>> from coola.hashing import HasherRegistry, SequenceHasher, StrHasher
            >>> registry = HasherRegistry({object: StrHasher()})
            >>> registry.register(Sequence, SequenceHasher())
            >>> hasher = registry.find_hasher(list)
            >>> hasher
            StrHasher()

            ```
        """
        return self._state.resolve(data_type)

    def hash(self, data: object, length: int = 64) -> str:
        """Hash the given data by recursively traversing its structure.

        This is the main entry point for hashing. It automatically:

        1. Determines the data's type.
        2. Finds the appropriate hasher via ``find_hasher``.
        3. Delegates to that hasher's ``hash`` method, which recursively
           processes any nested structures.

        Args:
            data: The data to hash. Can be a nested structure such as
                a ``list``, ``dict``, or ``tuple``.
            length: The desired length of the returned hex string. Must be an
                even number between 2 and 128 inclusive, since each byte of the
                digest encodes as two hex characters.

        Returns:
            A string representing the hash of the input data.

        Example:
            ```pycon
            >>> from coola.hashing import get_default_registry
            >>> registry = get_default_registry()
            >>> registry.hash({"scores": [95, 87, 92], "name": "test"})

            ```
        """
        hasher = self.find_hasher(type(data))
        return hasher.hash(data, registry=self, length=length)
