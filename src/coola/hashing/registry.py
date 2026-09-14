r"""Define the hasher registry for recursive data hashing.

This module provides a registry system that manages and dispatches
hashers based on data types, enabling recursive hashing of nested data
structures.
"""

from __future__ import annotations

__all__ = ["HasherRegistry"]

from typing import Any

from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_string
from coola.registry import BaseTypeDispatchRegistry


class HasherRegistry(BaseTypeDispatchRegistry[BaseHasher[Any]]):
    r"""Registry that manages and dispatches hashers based on data type.

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
        'e30f3d309eab8b8216b15ef153005972ce61c8c64c55f78075630089aed023de'

        ```

        Registering custom hashers:

        ```pycon
        >>> from coola.hashing import HasherRegistry, SequenceHasher, StrHasher
        >>> registry = HasherRegistry({object: StrHasher()})
        >>> registry.register(list, SequenceHasher())
        >>> registry.hash([1, 2, 3])
        'e30f3d309eab8b8216b15ef153005972ce61c8c64c55f78075630089aed023de'

        ```

        Working with nested structures:

        ```pycon
        >>> from coola.hashing import get_default_registry
        >>> registry = get_default_registry()
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> registry.hash(data)
        'fe7eca5d3348be5060774aab9a95169595884dbb3d1fb7ddc318b1123eadc32b'

        ```
    """

    def has_hasher(self, data_type: type) -> bool:
        """Type-specific alias for :meth:`has`: check if a hasher is
        explicitly registered for the given type.

        See :meth:`BaseTypeDispatchRegistry.has` for the full behavior
        description.

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
        return self.has(data_type)

    def find_hasher(self, data_type: type) -> BaseHasher[Any]:
        """Type-specific alias for :meth:`find`: find the appropriate
        hasher for a given type.

        See :meth:`BaseTypeDispatchRegistry.find` for the full behavior
        description (MRO resolution, caching, and the ``KeyError`` on no
        match).

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
        return self.find(data_type)

    def hash(self, data: object, length: int = 64, ignore_unhashable: bool = False) -> str:
        r"""Hash the given data by recursively traversing its structure.

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
            ignore_unhashable: If ``True``, objects for which no hasher is
                registered (including their parent types via MRO) are
                replaced by a deterministic placeholder hash instead of
                raising a ``KeyError``. This also applies to nested
                objects encountered while recursing into sequences or
                mappings. Defaults to ``False``, which raises an error.

        Returns:
            A string representing the hash of the input data.

        Raises:
            KeyError: If no hasher is registered for ``data``'s type
                (or any of its parent types) and ``ignore_unhashable``
                is ``False``.

        Example:
            ```pycon
            >>> from coola.hashing import get_default_registry
            >>> registry = get_default_registry()
            >>> registry.hash({"scores": [95, 87, 92], "name": "test"})
            '3f77b02a675a351ea0db656ace3425998b742702c03daf3694ab66d5cb67b729'

            ```
        """
        try:
            hasher = self.find_hasher(type(data))
        except KeyError:
            if ignore_unhashable:
                return hash_string(f"<unhashable:{type(data)!r}>", length=length)
            raise
        return hasher.hash(data, registry=self, length=length, ignore_unhashable=ignore_unhashable)
