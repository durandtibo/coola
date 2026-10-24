r"""Define the hasher registry for recursive data hashing.

This module provides a registry system that manages and dispatches
hashers based on data types, enabling recursive hashing of nested data
structures.
"""

from __future__ import annotations

__all__ = ["HasherRegistry"]

import threading
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from coola.hashing.base import Hasher
from coola.hashing.string import hash_string
from coola.registry import BaseTypeDispatchRegistry, TypeNotRegisteredError
from coola.utils.fork import track_fork

if TYPE_CHECKING:
    from collections.abc import Callable

_state = threading.local()


class HasherRegistry(BaseTypeDispatchRegistry[Hasher[Any]]):
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
              (<class 'list'>): SequenceHasher(tag='sequence')
            )
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

    def __init__(self, initial_state: dict[type, Hasher[Any]] | None = None) -> None:
        super().__init__(initial_state=initial_state)
        # Copy-on-write tuple: readers do not need a lock.
        self._fallbacks: tuple[tuple[Callable[[type], bool], Hasher[Any]], ...] = ()
        self._fallbacks_lock = threading.Lock()
        track_fork(self)

    def _reset_after_fork(self) -> None:
        r"""Recreate the lock of the fallbacks in a forked child process.

        The lock may have been held by a thread that does not exist in
        the child process, which would block the registry forever. The
        registered hashers and fallbacks are kept. The lock of the
        internal type registry is recreated by the type registry.
        """
        self._fallbacks_lock = threading.Lock()

    def _display_fields(self) -> dict[str, Any]:
        kwargs = super()._display_fields()
        if self._fallbacks:
            kwargs["fallbacks"] = [hasher for _, hasher in self._fallbacks]
        return kwargs

    def register_fallback(self, predicate: Callable[[type], bool], hasher: Hasher[Any]) -> None:
        r"""Register a hasher for the types that satisfy a predicate.

        A type-keyed registry cannot express "any dataclass" or "any
        class that defines a ``hash`` method". A fallback is used only
        if no hasher is registered for the type or for one of its
        parent types, so a hasher registered for a type always wins.
        If several predicates match, the first registered one is used.
        Note that registering a hasher for ``object`` makes the
        fallbacks unreachable, since ``object`` matches every type.

        Args:
            predicate: A function that takes a type and returns ``True``
                if ``hasher`` can hash the instances of this type.
            hasher: The hasher to use for these types.

        Example:
            ```pycon
            >>> from dataclasses import dataclass, is_dataclass
            >>> from coola.hashing import DataclassHasher, HasherRegistry, SequenceHasher
            >>> registry = HasherRegistry({list: SequenceHasher()})
            >>> registry.register_fallback(is_dataclass, DataclassHasher())
            >>> @dataclass
            ... class Point:
            ...     x: int
            ...
            >>> registry.find_hasher(Point)
            DataclassHasher()

            ```
        """
        with self._fallbacks_lock:
            self._fallbacks = (*self._fallbacks, (predicate, hasher))

    def unregister_fallback(self, predicate: Callable[[type], bool]) -> None:
        r"""Remove the fallback registered for a predicate.

        Args:
            predicate: The predicate that was given to
                ``register_fallback``.

        Raises:
            KeyError: If no fallback is registered for this predicate.

        Example:
            ```pycon
            >>> from dataclasses import dataclass, is_dataclass
            >>> from coola.hashing import DataclassHasher, HasherRegistry
            >>> registry = HasherRegistry()
            >>> registry.register_fallback(is_dataclass, DataclassHasher())
            >>> registry.unregister_fallback(is_dataclass)
            >>> @dataclass
            ... class Point:
            ...     x: int
            ...
            >>> registry.has_hasher(Point)
            False

            ```
        """
        with self._fallbacks_lock:
            remaining = tuple(item for item in self._fallbacks if item[0] != predicate)
            if len(remaining) == len(self._fallbacks):
                msg = f"No fallback is registered for {predicate!r}"
                raise KeyError(msg)
            self._fallbacks = remaining

    def clear(self) -> None:
        r"""Remove all the registered hashers and fallbacks."""
        super().clear()
        with self._fallbacks_lock:
            self._fallbacks = ()

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        r"""Indicate if two registries are equal or not.

        Two registries are equal if they have the same hashers and the
        same fallbacks (same predicates, in the same order, with equal
        hashers).

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

        Returns:
            ``True`` if ``other`` has the same type, the same
                registered hashers and the same fallbacks, otherwise
                ``False``.

        Example:
            ```pycon
            >>> from dataclasses import is_dataclass
            >>> from coola.hashing import DataclassHasher, HasherRegistry
            >>> registry1 = HasherRegistry()
            >>> registry2 = HasherRegistry()
            >>> registry1.equal(registry2)
            True
            >>> registry2.register_fallback(is_dataclass, DataclassHasher())
            >>> registry1.equal(registry2)
            False

            ```
        """
        # Local import to avoid a cyclic import (see ``BaseRegistry.equal``).
        from coola.equality.interface import objects_are_equal  # noqa: PLC0415

        if not super().equal(other, equal_nan):
            return False
        fallbacks, other_fallbacks = self._fallbacks, cast("HasherRegistry", other)._fallbacks
        return len(fallbacks) == len(other_fallbacks) and all(
            predicate == other_predicate
            and objects_are_equal(hasher, other_hasher, equal_nan=equal_nan)
            for (predicate, hasher), (other_predicate, other_hasher) in zip(
                fallbacks, other_fallbacks, strict=True
            )
        )

    def has_hasher(self, data_type: type) -> bool:
        """Type-specific alias for ``has``: check if a hasher is
        explicitly registered for the given type.

        See ``BaseTypeDispatchRegistry.has`` for the full behavior
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

        See ``BaseTypeDispatchRegistry.find`` for the full behavior
        description (MRO resolution, caching, and the ``KeyError`` on no
        match). If no hasher is registered for the type or its parent
        types, the fallbacks (see ``register_fallback``) are tried.

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
        # ``resolve_or`` does not build the error message of a failing lookup, which is
        # costly for the types that are hashed by a fallback (every dataclass).
        hasher = self._state.resolve_or(data_type)
        if hasher is not None:
            return hasher
        for predicate, fallback in self._fallbacks:
            if predicate(data_type):
                return fallback
        raise TypeNotRegisteredError(self._state._not_registered_msg(data_type))

    def _find_enum_hasher(self, data_type: type) -> Hasher[Any] | None:
        """Return the ``Enum`` hasher for an enum mixed with another
        type, or ``None`` if the regular lookup should be used.

        A member of ``class C(bytes, Enum)`` would otherwise resolve to
        the hasher of ``bytes`` and hash as the plain value, like a
        member of a different enum with the same value.
        """
        if not self.has(Enum):
            return None
        # A hasher registered for the enum itself (or for ``Enum``) is the
        # most specific one.
        base = next((base for base in data_type.__mro__ if self.has(base)), Enum)
        return None if issubclass(base, Enum) else self.find(Enum)

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
                mappings. The placeholder only depends on the type, so
                any two unsupported objects of the same type hash the
                same; a ``UserWarning`` is emitted each time a
                placeholder is used. Defaults to ``False``, which
                raises an error.

        Returns:
            A string representing the hash of the input data.

        Raises:
            KeyError: If no hasher is registered for ``data``'s type
                (or any of its parent types) and ``ignore_unhashable``
                is ``False``.
            ValueError: If ``data`` contains itself (a self-referencing
                container), or is nested too deeply for the recursion
                limit of the interpreter.

        Example:
            ```pycon
            >>> from coola.hashing import get_default_registry
            >>> registry = get_default_registry()
            >>> len(registry.hash({"scores": [95, 87, 92], "name": "test"}))
            64

            ```
        """
        try:
            hasher = self._find_enum_hasher(type(data)) if isinstance(data, Enum) else None
            if hasher is None:
                hasher = self.find_hasher(type(data))
        except KeyError as exc:
            if ignore_unhashable:
                warnings.warn(
                    f"No hasher registered for {type(data)!r}: using a type-only placeholder, "
                    "so all objects of this type hash the same.",
                    UserWarning,
                    stacklevel=2,
                )
                return hash_string(f"<unhashable:{type(data)!r}>", length=length)
            if isinstance(exc, TypeNotRegisteredError):
                msg = (
                    f"{exc}\nTo hash this type, register a hasher "
                    f"(`register_hashers({{{type(data).__name__}: ...}})`), "
                    "give the class a `hash()` method (see `SupportsHash`), "
                    "or pass `ignore_unhashable=True` to hash a placeholder that only "
                    "depends on the type."
                )
                raise TypeNotRegisteredError(msg) from exc
            raise
        active: set[int] | None = getattr(_state, "active", None)
        if active is None:
            active = _state.active = set()
        key = id(data)
        if key in active:
            msg = f"Cannot hash a self-referencing {type(data)!r}."
            raise ValueError(msg)
        active.add(key)
        try:
            return hasher.hash(
                data, registry=self, length=length, ignore_unhashable=ignore_unhashable
            )
        except RecursionError as exc:
            if len(active) > 1:  # only the outermost call reports the error
                raise
            msg = "Cannot hash data that is nested too deeply (maximum recursion depth exceeded)."
            raise ValueError(msg) from exc
        finally:
            active.discard(key)
