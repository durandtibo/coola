r"""Define a thread-safe generic key-value registry."""

from __future__ import annotations

__all__ = ["Registry"]

import copy
import threading
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, cast

from coola.formatting.structure import repr_indent, repr_mapping, str_indent, str_mapping
from coola.registry.exceptions import AlreadyRegisteredError, KeyNotRegisteredError
from coola.utils.fork import track_fork

if TYPE_CHECKING:
    from collections.abc import (
        ItemsView,
        Iterator,
        KeysView,
        Mapping,
        ValuesView,
    )

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class Registry(Generic[K, V]):
    r"""A thread-safe generic key-value registry for storing and managing
    typed mappings.

    The Registry class provides a type-safe container for registering
    and retrieving values by key. It supports all standard dictionary
    operations through operator overloading and provides additional
    methods for safe registration and querying.

    The state is a dictionary that is replaced on every write and never
    mutated, so the reads take no lock, ``items()``, ``keys()`` and
    ``values()`` return snapshots, and a writer never shows a partial
    state to a reader. The writers are serialized by one reentrant lock.
    ``TypeRegistry`` extends this class.

    Args:
        initial_state: An optional dictionary to initialize the registry
            with. If provided, a copy is made to prevent external
            modifications. Defaults to None, which creates an empty
            registry.

    Attributes:
        _state: Internal dictionary storing the key-value pairs. It is
            replaced, never mutated.
        _lock: Threading lock for synchronizing the writers.

    Raises:
        TypeError: If a key is not valid for the registry.

    Example:
        Basic usage with registration and retrieval:

        ```pycon
        >>> from coola.registry import Registry
        >>> registry = Registry[str, int]()
        >>> registry.register("key1", 42)
        >>> registry.get("key1")
        42
        >>> registry
        Registry(
          (key1): 42
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

    def __init__(self, initial_state: dict[K, V] | None = None) -> None:
        for key in initial_state or ():
            self._validate_key(key)
        # Replaced on every write, never mutated: the readers need no lock.
        self._state: dict[K, V] = initial_state.copy() if initial_state else {}
        self._lock: threading.RLock = threading.RLock()  # RLock allows re-entrant locking
        track_fork(self)

    def __contains__(self, key: K) -> bool:
        r"""Indicate if a key is registered.

        Args:
            key: The key.

        Returns:
            ``True`` if the key is registered, otherwise ``False``.
        """
        return self.has(key)

    def __getitem__(self, key: K) -> V:
        r"""Get the value of a key.

        Args:
            key: The key.

        Returns:
            The value of the key.

        Raises:
            KeyNotRegisteredError: If the key is not registered.
        """
        self._load_lazy_for(key)
        with self._lock:
            if key not in self._state:
                raise self._not_registered_error(self._getitem_not_registered_msg(key))
            return self._state[key]

    def __setitem__(self, key: K, value: V) -> None:
        r"""Register a value, replacing the existing one.

        Args:
            key: The key.
            value: The value.
        """
        self.register(key, value, exist_ok=True)

    def __delitem__(self, key: K) -> None:
        r"""Unregister a key.

        Args:
            key: The key.

        Raises:
            KeyNotRegisteredError: If the key is not registered.
        """
        self.unregister(key)

    def __iter__(self) -> Iterator[K]:
        r"""Iterate over the registered keys.

        Returns:
            An iterator over the keys.
        """
        self._load_lazy_all()
        return iter(self._state)

    def __len__(self) -> int:
        r"""Return the number of registered keys.

        Returns:
            The number of entries.
        """
        self._load_lazy_all()
        return len(self._state)

    def __repr__(self) -> str:
        r"""Return the representation, with the ``repr`` of the entries.

        Returns:
            The representation of the registry.
        """
        # Displaying a registry must not import the optional libraries: the entries that
        # are not loaded yet are listed by group name instead.
        with self._lock:
            state = self._state
            pending = self._pending_msg()
        body = "\n  ".join(filter(None, [repr_indent(repr_mapping(state)), pending]))
        return f"{self.__class__.__qualname__}(\n  {body}\n)"

    def __str__(self) -> str:
        r"""Return the string, with the ``str`` of the entries.

        Returns:
            The string of the registry.
        """
        with self._lock:
            state = self._state
            pending = self._pending_msg()
        body = "\n  ".join(filter(None, [str_indent(str_mapping(state)), pending]))
        return f"{self.__class__.__qualname__}(\n  {body}\n)"

    def __getstate__(self) -> dict[str, Any]:
        r"""Return the state used by ``pickle`` and ``copy``.

        Returns:
            The state, without the lock.
        """
        # The lock cannot be pickled: the clone creates its own.
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        r"""Restore the state used by ``pickle`` and ``copy``.

        Args:
            state: The state returned by ``__getstate__``.
        """
        self.__dict__.update(state)
        self._lock = threading.RLock()
        track_fork(self)

    def _pending_msg(self) -> str:
        r"""Return the lines that complete the display of the entries
        (for example the lazy groups that are not loaded).

        This must be called while holding ``self._lock``. The default
        implementation returns nothing.

        Returns:
            The lines to add to the display, or an empty string.
        """
        return ""

    def _changed(self) -> None:
        r"""React to a change of the state.

        This is called while holding ``self._lock``, after the state is
        replaced. Subclasses can override it to invalidate any derived
        state (e.g. a resolution cache).
        """

    def _publish(self, state: dict[K, V]) -> None:
        r"""Replace the state.

        This must be called while holding ``self._lock``.

        Args:
            state: The new state. It must not be mutated afterwards.
        """
        self._state = state
        self._changed()

    def copy(self) -> Self:
        r"""Create a copy of the registry.

        The copy has the same entries and the same pending lazy groups
        as the registry, and is independent from it: registering or
        removing an entry in one registry does not change the other.
        The keys and the values are not copied.

        Returns:
            A new registry of the same class.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]({"key": 42})
            >>> other = registry.copy()
            >>> other.register("other", 1)
            >>> registry.has("other")
            False
            >>> other.get("key")
            42

            ```
        """
        # The attributes that are not replaced below (the configuration of a subclass) are shared.
        new = copy.copy(self)
        with self._lock:
            # The state is never mutated, so it is shared until the next write.
            new._state = self._state
        new._lock = threading.RLock()
        new._changed()
        track_fork(new)
        return new

    def _reset_after_fork(self) -> None:
        r"""Recreate the lock in a forked child process.

        The lock may have been held by a thread that does not exist in
        the child process, which would block the registry forever. The
        content of the registry is kept: the child has a copy of the
        memory of its parent.
        """
        self._lock = threading.RLock()

    _not_registered_error: type[KeyNotRegisteredError] = KeyNotRegisteredError
    """The exception raised when a key is not registered."""

    def _validate_key(self, key: K) -> None:
        r"""Validate a key before it is stored in the registry.

        The default implementation accepts every key. Subclasses override
        it to reject invalid keys.

        Args:
            key: The key to validate.

        Raises:
            TypeError: If the key is invalid.
        """

    def _not_registered_msg(self, key: K) -> str:
        r"""Return the error message used when ``key`` is not registered.

        Args:
            key: The key.

        Returns:
            The message.
        """
        return f"Key '{key}' is not registered"

    def _getitem_not_registered_msg(self, key: K) -> str:
        r"""Return the error message used by ``__getitem__`` when ``key``
        is not registered.

        Args:
            key: The key.

        Returns:
            The message.
        """
        return self._not_registered_msg(key)

    def _already_registered_msg(self, key: K) -> str:
        r"""Return the error message used when ``key`` is already
        registered.

        Args:
            key: The key.

        Returns:
            The message.
        """
        return (
            f"A value is already registered for '{key}'. "
            "Use a different key or set exist_ok=True to override."
        )

    def _many_already_registered_msg(self, duplicates: set[K]) -> str:
        r"""Return the error message used when several keys are already
        registered.

        Args:
            duplicates: The keys that are already registered.

        Returns:
            The message.
        """
        return (
            f"Keys already registered: {', '.join(sorted(map(str, duplicates)))}. "
            "Use different keys or set exist_ok=True to override."
        )

    def clear(self) -> None:
        r"""Remove all entries from the registry.

        This method empties the registry, leaving it in the same state as a
        newly created empty registry. This operation cannot be undone.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]({"key1": 42, "key2": 100})
            >>> len(registry)
            2
            >>> registry.clear()
            >>> len(registry)
            0
            >>> registry.has("key1")
            False

            ```
        """
        with self._lock:
            self._publish({})

    def equal(self, other: object) -> bool:
        r"""Indicate if two objects are equal or not.

        Two registries are equal if they have the same class and the
        same entries. The values are compared with ``==``, so a value
        without ``__eq__`` is equal to itself only. A registry is equal
        to its copy. The pending lazy groups are loaded before the
        comparison, except if the two registries have the same pending
        groups.

        Args:
            other: The object to compare with.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry1 = Registry[str, int]({"key1": 42, "key2": 100})
            >>> registry2 = Registry[str, int]({"key1": 42, "key2": 100})
            >>> registry3 = Registry[str, int]({"key1": 42})
            >>> registry1.equal(registry2)
            True
            >>> registry1.equal(registry3)
            False

            ```
        """
        if type(other) is not type(self):
            return False
        other = cast("Registry[K, V]", other)
        self._load_lazy_for_equal(other)
        return self._state == other._state

    def get(self, key: K, default: V | None = None) -> V | None:
        r"""Retrieve the value associated with a key.

        This method performs a lookup in the registry and returns the
        corresponding value.

        Args:
            key: The key whose value should be retrieved.
            default: Value to return if the key does not exist.

        Returns:
            The value associated with the specified key.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]({"key1": 42, "key2": 100})
            >>> registry.get("key1")
            42
            >>> registry.get("missing") is None
            True

            ```
        """
        self._load_lazy_for(key)
        return self._state.get(key, default)

    def has(self, key: K) -> bool:
        r"""Check whether a key is registered in the registry.

        This method provides a safe way to test for key existence without
        risking a KeyError exception.

        Args:
            key: The key to check for existence.

        Returns:
            True if the key exists in the registry, False otherwise.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]({"key1": 42, "key2": 100})
            >>> registry.has("key1")
            True
            >>> registry.has("missing")
            False

            ```
        """
        self._load_lazy_for(key)
        return key in self._state

    def register(self, key: K, value: V, *, exist_ok: bool = False) -> None:
        r"""Register a new key-value pair in the registry.

        By default, this method raises an error if you try to register a key
        that already exists. This prevents accidental overwriting of values.
        Set exist_ok=True to allow overwriting.

        Args:
            key: The key to register. Must be hashable.
            value: The value to associate with the key.
            exist_ok: Controls behavior when the key already exists.
                If False (default), raises AlreadyRegisteredError for duplicate keys.
                If True, silently overwrites the existing value.

        Raises:
            AlreadyRegisteredError: If the key is already registered and exist_ok is False.
            TypeError: If the key is rejected by the registry (for example a
                non-type key in a ``TypeRegistry``).
                The error message provides guidance on how to resolve the conflict.

        Example:
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
            >>> registry.register("key1", 100)
            Traceback (most recent call last):
                ...
            coola.registry.exceptions.AlreadyRegisteredError: A value is already registered for 'key1'. Use a different key or set exist_ok=True to override.

            ```

            Overwriting with exist_ok:

            ```pycon
            >>> registry.register("key1", 100, exist_ok=True)
            >>> registry.get("key1")
            100

            ```
        """
        self._validate_key(key)
        with self._lock:
            self._load_lazy_for(key)
            if key in self._state and not exist_ok:
                raise AlreadyRegisteredError(self._already_registered_msg(key))
            self._publish({**self._state, key: value})

    def register_many(self, mapping: Mapping[K, V], *, exist_ok: bool = False) -> None:
        r"""Register multiple key-value pairs in a single operation.

        This is a convenience method for bulk registration. It iterates through
        the provided mapping and registers each key-value pair. All registrations
        follow the same exist_ok policy. The operation is atomic *with respect to
        this registry* when exist_ok is False: either every key-value pair in
        ``mapping`` ends up registered, or (if any key already exists) none of
        them do - there is no partial registration. This says nothing about
        atomicity across multiple registries: if ``mapping`` is registered into
        several registries in sequence, each ``register_many`` call is atomic on
        its own, but the overall multi-registry operation is not - a failure on
        a later registry does not roll back an earlier one.

        Args:
            mapping: A dictionary or mapping containing the key-value pairs
                to register. The keys and values must match the registry's
                type parameters.
            exist_ok: Controls behavior when any key already exists.
                If False (default), raises error on the first duplicate key.
                If True, overwrites all existing values without error.

        Raises:
            AlreadyRegisteredError: If exist_ok is False and any key in the mapping
                is already registered. The error occurs on the first duplicate
                encountered, and no partial registration occurs.
            TypeError: If a key is rejected by the registry. No key is registered.

        Example:
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
        for key in mapping:
            self._validate_key(key)
        with self._lock:
            for key in mapping:
                self._load_lazy_for(key)
            # Check all keys first if exist_ok is False
            if not exist_ok and (duplicates := set(mapping) & set(self._state)):
                raise AlreadyRegisteredError(self._many_already_registered_msg(duplicates))
            self._publish({**self._state, **mapping})

    def unregister(self, key: K) -> V:
        r"""Remove a key-value pair from the registry and return the
        value.

        This method removes the specified key from the registry and returns
        the value that was associated with it. This allows you to retrieve
        the value one last time before it's removed.

        Args:
            key: The key to unregister and remove from the registry.

        Returns:
            The value that was associated with the key before removal.

        Raises:
            KeyNotRegisteredError: If the key is not registered. It
                subclasses ``KeyError``, and its message includes the
                key that was not found. Registries with type keys raise
                its subclass ``TypeNotRegisteredError``.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]({"key1": 42, "key2": 100})
            >>> registry.has("key1")
            True
            >>> value = registry.unregister("key1")
            >>> value
            42
            >>> registry.has("key1")
            False

            ```
        """
        with self._lock:
            self._load_lazy_for(key)
            if key not in self._state:
                raise self._not_registered_error(self._not_registered_msg(key))
            state = dict(self._state)
            value = state.pop(key)
            self._publish(state)
            return value

    def _load_lazy_for(self, key: K) -> None:
        r"""Load the entries that are not loaded yet and are related to a
        key.

        This is a hook for the subclasses that have lazy entries. The
        default implementation does nothing.

        Args:
            key: The key.
        """

    def _load_lazy_all(self) -> None:
        r"""Load all the entries that are not loaded yet.

        This is a hook for the subclasses that have lazy entries. The
        default implementation does nothing.
        """

    def _load_lazy_for_equal(self, other: Registry[K, V]) -> None:
        r"""Load the entries that must be loaded before comparing the
        registry with ``other``.

        Args:
            other: The registry to compare with.
        """
        self._load_lazy_all()
        other._load_lazy_all()

    def items(self) -> ItemsView[K, V]:
        r"""Return key-value pairs.

        Returns:
            The key-value pairs.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int](initial_state={"a": 1, "b": 2})
            >>> registry.items()
            dict_items([('a', 1), ('b', 2)])

            ```
        """
        self._load_lazy_all()
        return self._state.items()

    def keys(self) -> KeysView[K]:
        r"""Return registered keys.

        Returns:
            The registered keys.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int](initial_state={"a": 1, "b": 2})
            >>> registry.keys()
            dict_keys(['a', 'b'])

            ```
        """
        self._load_lazy_all()
        return self._state.keys()

    def values(self) -> ValuesView[V]:
        r"""Return registered values.

        Returns:
            The registered values.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int](initial_state={"a": 1, "b": 2})
            >>> registry.values()
            dict_values([1, 2])

            ```
        """
        self._load_lazy_all()
        return self._state.values()
