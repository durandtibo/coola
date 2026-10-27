r"""Define the hasher registry for recursive data hashing.

This module provides a registry system that manages and dispatches
hashers based on data types, enabling recursive hashing of nested data
structures.
"""

from __future__ import annotations

__all__ = ["HasherRegistry"]

from enum import Enum
from typing import Any

from coola.hashing.base import Hasher
from coola.hashing.engine import hash_hex
from coola.registry import TypeRegistry


class HasherRegistry(TypeRegistry[Hasher[Any]]):
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
          (<class 'object'>): StrHasher()
          (<class 'list'>): SequenceHasher(tag='sequence')
        )
        >>> len(registry.hash([1, 2, 3]))
        64

        ```

        Registering custom hashers:

        ```pycon
        >>> from coola.hashing import HasherRegistry, SequenceHasher, StrHasher
        >>> registry = HasherRegistry({object: StrHasher()})
        >>> registry.register(list, SequenceHasher())
        >>> len(registry.hash([1, 2, 3]))
        64

        ```

        Working with nested structures:

        ```pycon
        >>> from coola.hashing import get_default_registry
        >>> registry = get_default_registry()
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> len(registry.hash(data))
        64

        ```
    """

    def has_hasher(self, data_type: type) -> bool:
        """Type-specific alias for ``has``: check if a hasher is
        explicitly registered for the given type.

        See ``Registry.has`` for the full behavior
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

    def find_hasher(self, data_type: type) -> Hasher[Any]:
        """Type-specific alias for ``find``: find the appropriate
        hasher for a given type.

        See ``TypeRegistry.resolve`` for the full behavior
        description (MRO resolution, caching, and the ``KeyError`` on no
        match). If no hasher is registered for the type or its parent
        types, the fallbacks (see ``register_fallback``) are tried.

        A subclass of ``Enum`` resolves to the hasher of ``Enum`` unless
        a hasher is registered for the enum class itself (or for a
        parent that is not a builtin base of the mixin). A member of
        ``class C(bytes, Enum)`` would otherwise hash as the plain
        value, like a member of a different enum with the same value.

        Args:
            data_type: The Python type to find a hasher for.

        Returns:
            The most specific registered hasher for this type, resolved
            via MRO, or the hasher of the first matching fallback.

        Raises:
            TypeNotRegisteredError: If no hasher is found.

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
        if issubclass(data_type, Enum) and self.has(Enum):
            # A hasher registered for the enum itself (or for ``Enum``) is the most specific one.
            base = next((base for base in data_type.__mro__ if self.has(base)), Enum)
            if issubclass(base, Enum):
                return self.resolve(base)
            return self.resolve(Enum)
        return self.resolve(data_type)

    def hash(self, data: object, *, length: int = 64, ignore_unhashable: bool = False) -> str:
        r"""Hash the given data.

        This is the registry-level entry point for hashing. It finds the
        hasher of each object with ``find_hasher`` and walks the nested
        structure with the engine (see ``coola.hashing.engine``). The
        hash is the first ``length`` characters of the 128-character
        hex digest, so a hash is the start of the longer ones.

        Args:
            data: The data to hash. Can be a nested structure such as
                a ``list``, ``dict``, or ``tuple``.
            length: The length of the returned hex string: an even
                ``int`` between 2 and 128 inclusive.
            ignore_unhashable: If ``True``, objects for which no hasher is
                registered (including their parent types via MRO) are
                replaced by a deterministic placeholder hash instead of
                raising a ``TypeNotRegisteredError``. This also applies to
                nested objects. The placeholder only depends on the type,
                so any two unsupported objects of the same type hash the
                same; a ``UserWarning`` is emitted once per type.
                Defaults to ``False``, which raises an error.

        Returns:
            A string representing the hash of the input data.

        Raises:
            TypeNotRegisteredError: If no hasher is registered for
                the type of ``data`` or of a nested object (or any of
                their parent types) and ``ignore_unhashable`` is
                ``False``. It is a subclass of ``KeyError``.
            ValueError: If ``length`` is not valid, or if ``data``
                contains itself (a self-referencing container).

        Example:
            ```pycon
            >>> from coola.hashing import get_default_registry
            >>> registry = get_default_registry()
            >>> len(registry.hash({"scores": [95, 87, 92], "name": "test"}))
            64

            ```
        """
        return hash_hex(
            data, self, length=length, ignore_unhashable=ignore_unhashable, stacklevel=2
        )
