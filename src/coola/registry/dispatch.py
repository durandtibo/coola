r"""Define a generic base class for type-dispatch registries.

This module provides ``BaseTypeDispatchRegistry``, the shared
implementation behind the small family of registries that map Python
types to a single handler object (equality tester, hasher, transformer,
summarizer, child finder, iterator, ...) and dispatch on an object's
type using Method Resolution Order (MRO). Concrete registries subclass
it to get ``register``, ``register_many``, ``has``, and ``find`` for
free, and only need to add their domain-specific entry point (e.g.
``objects_are_equal``, ``hash``, ``transform``).
"""

from __future__ import annotations

__all__ = ["BaseTypeDispatchRegistry"]

from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from coola.display import MultilineDisplayMixin
from coola.registry.type import TypeRegistry

if TYPE_CHECKING:
    from collections.abc import ItemsView, Mapping

V = TypeVar("V")


class BaseTypeDispatchRegistry(MultilineDisplayMixin, Generic[V]):
    r"""Base class for registries that dispatch a single handler based on
    data type.

    This class maintains a mapping from Python types to handler
    instances (e.g. equality testers, hashers, transformers) and uses
    the Method Resolution Order (MRO) for type lookup, via an internal
    ``TypeRegistry``. When resolving a handler for some data, the most
    specific registered handler for the data's type is used, falling
    back to parent types (or the handler registered for ``object``, if
    any) if needed.

    Subclasses typically expose their own, more specifically named
    ``has_<x>``/``find_<x>`` wrappers around ``has``/``find`` plus a
    domain-specific entry point (e.g. ``hash``, ``transform``), but the
    registration and lookup machinery itself lives entirely here.

    Args:
        initial_state: Optional initial mapping of types to handlers.
            If provided, the state is copied to prevent external
            mutations.

    Attributes:
        _state: Internal ``TypeRegistry`` of registered types to
            handlers.
    """

    def __init__(self, initial_state: dict[type, V] | None = None) -> None:
        self._state: TypeRegistry[V] = TypeRegistry[V](initial_state)

    def _display_fields(self) -> dict[str, Any]:
        return {"state": self._state}

    def __contains__(self, data_type: type) -> bool:
        return self.has(data_type)

    def __len__(self) -> int:
        return len(self._state)

    def register(self, data_type: type, value: V, exist_ok: bool = False) -> None:
        r"""Register a handler for a given data type.

        The internal type-lookup cache is automatically cleared after
        registration to ensure consistency.

        Args:
            data_type: The Python type to register (e.g., ``list``,
                ``dict``, custom classes).
            value: The handler instance that handles this type.
            exist_ok: If ``False`` (default), raises an error if the
                type is already registered. If ``True``, overwrites
                the existing registration silently.

        Raises:
            AlreadyRegisteredError: If the type is already registered and
                ``exist_ok`` is ``False``.
        """
        self._state.register(data_type, value, exist_ok=exist_ok)

    def register_many(self, mapping: Mapping[type, V], exist_ok: bool = False) -> None:
        r"""Register multiple handlers at once.

        This is a convenience method for bulk registration that
        registers all the pairs at once: nothing is registered if one
        of the types is invalid or (with ``exist_ok=False``) already registered.

        Args:
            mapping: Dictionary mapping Python types to handler
                instances.
            exist_ok: If ``False`` (default), raises an error if any
                type is already registered. If ``True``, overwrites
                existing registrations silently.

        Raises:
            AlreadyRegisteredError: If any type is already registered and
                ``exist_ok`` is ``False``.
        """
        self._state.register_many(mapping, exist_ok=exist_ok)

    def has(self, data_type: type) -> bool:
        r"""Check if a handler is explicitly registered for the given
        type.

        Note that this only checks for direct registration. Even if
        this returns ``False``, ``find`` may still return a handler
        via MRO lookup or the handler registered for ``object``.

        Args:
            data_type: The type to check.

        Returns:
            ``True`` if a handler is explicitly registered for this
                type, ``False`` otherwise.
        """
        return data_type in self._state

    def find(self, data_type: type) -> V:
        r"""Find the appropriate handler for a given type.

        Uses the Method Resolution Order (MRO) to find the most
        specific registered handler. For example, if a handler is
        registered for ``Sequence`` but not for ``list``, lists will
        use the ``Sequence`` handler.

        Args:
            data_type: The Python type to find a handler for.

        Returns:
            The most specific registered handler for this type,
            resolved via MRO (the handler registered for ``object``
            matches every type).

        Raises:
            TypeNotRegisteredError: If no handler is registered for
                ``data_type`` or any of its parent types. It is a
                ``KeyError``.

        Note:
            Results are cached internally (see ``TypeRegistry.resolve``)
            so repeated lookups for the same type are fast. The cache
            is invalidated automatically by ``register``/``register_many``.
        """
        return self._state.resolve(data_type)

    def unregister(self, data_type: type) -> V:
        r"""Remove the handler registered for a given data type.

        Args:
            data_type: The type to unregister.

        Returns:
            The handler that was registered for this type.

        Raises:
            TypeNotRegisteredError: If no handler is directly registered
                for ``data_type``.
        """
        return self._state.unregister(data_type)

    def clear(self) -> None:
        r"""Remove all the registered handlers."""
        self._state.clear()

    def items(self) -> ItemsView[type, V]:
        r"""Return a view of the registered ``(type, handler)`` pairs.

        The view is a snapshot: it is not updated by later changes.
        """
        return self._state.items()

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        r"""Indicate if two registries are equal or not.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

        Returns:
            ``True`` if ``other`` has the same type and the same
                registered handlers, otherwise ``False``.
        """
        if type(other) is not type(self):
            return False
        return self._state.equal(cast("BaseTypeDispatchRegistry[V]", other)._state, equal_nan)
