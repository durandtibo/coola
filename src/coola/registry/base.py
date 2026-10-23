r"""Define a thread-safe base registry shared by ``Registry`` and
``TypeRegistry``."""

from __future__ import annotations

__all__ = ["BaseRegistry"]

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from coola.registry.exceptions import AlreadyRegisteredError, KeyNotRegisteredError
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping
from coola.utils.imports import import_object
from coola.utils.introspection import is_same_type

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        ItemsView,
        Iterable,
        Iterator,
        KeysView,
        Mapping,
        Sequence,
        ValuesView,
    )

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


@dataclass
class _LazyEntry(Generic[K, V]):
    r"""Entries of a registry that are loaded when they are needed.

    Attributes:
        names: The names of the group. The entries are loaded the first
            time the registry needs one of these names.
        factory: A callable that returns the entries, or the path
            ``"package.module:name"`` of this callable. A path is not
            imported until the entries are loaded.
    """

    names: tuple[str, ...]
    factory: str | Callable[[], Mapping[K, V]]

    def load(self) -> Mapping[K, V]:
        r"""Build the entries."""
        factory = import_object(self.factory) if isinstance(self.factory, str) else self.factory
        return factory()


class BaseRegistry(Generic[K, V]):
    r"""A thread-safe generic key-value registry for storing and managing
    typed mappings.

    This class implements the common storage, locking, and dictionary-like
    behavior shared by ``Registry`` and ``TypeRegistry``. Subclasses may
    override ``_on_change`` to react to mutations, e.g. to invalidate an
    internal cache.

    Args:
        initial_state: An optional dictionary to initialize the registry with.
            If provided, a copy is made to prevent external modifications.
            Defaults to None, which creates an empty registry.

    Attributes:
        _state: Internal dictionary storing the key-value pairs.
        _lock: Threading lock for synchronizing access to the registry.
    """

    def __init__(self, initial_state: dict[K, V] | None = None) -> None:
        for key in initial_state or ():
            self._validate_key(key)
        self._state: dict[K, V] = initial_state.copy() if initial_state else {}
        self._lock: threading.RLock = threading.RLock()  # RLock allows re-entrant locking
        # Cached snapshot of ``_state`` used by ``items``/``keys``/``values`` and by
        # ``__iter__``/``__repr__``/``__str__``. It is
        # invalidated (set to None) by ``_invalidate`` on every mutation, so repeated
        # read-only calls between mutations reuse the same copy instead of paying for
        # a fresh O(n) ``dict.copy()`` each time.
        self._snapshot: dict[K, V] | None = None
        # Entries that are not loaded yet, by group name (see ``register_lazy``).
        self._lazy: dict[str, _LazyEntry[K, V]] = {}

    def __contains__(self, key: K) -> bool:
        return self.has(key)

    def __getitem__(self, key: K) -> V:
        self._load_lazy_for(key)
        with self._lock:
            if key not in self._state:
                raise self._not_registered_error(self._getitem_not_registered_msg(key))
            return self._state[key]

    def __setitem__(self, key: K, value: V) -> None:
        self.register(key, value, exist_ok=True)

    def __delitem__(self, key: K) -> None:
        self.unregister(key)

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            return iter(self._get_snapshot())

    def __len__(self) -> int:
        self._load_lazy_all()
        with self._lock:
            return len(self._state)

    def __repr__(self) -> str:
        with self._lock:
            snapshot = self._get_snapshot()
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(snapshot))}\n)"

    def __str__(self) -> str:
        with self._lock:
            snapshot = self._get_snapshot()
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(snapshot))}\n)"

    def _on_change(self) -> None:
        r"""React to a mutation of the registry state.

        This is called while holding ``self._lock``. Subclasses can
        override it to invalidate any derived state (e.g. a resolution
        cache).
        """

    def _invalidate(self) -> None:
        r"""Invalidate state derived from ``_state`` after a mutation.

        This is called while holding ``self._lock`` by every method that
        mutates ``_state``. It clears the ``items``/``keys``/``values``
        snapshot cache and calls ``_on_change`` so subclasses can
        invalidate their own derived state (e.g. a resolution cache).
        """
        self._snapshot = None
        self._on_change()

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
        r"""Return the error message used when ``key`` is not
        registered."""
        return f"Key '{key}' is not registered"

    def _getitem_not_registered_msg(self, key: K) -> str:
        r"""Return the error message used by ``__getitem__`` when
        ``key`` is not registered."""
        return self._not_registered_msg(key)

    def _already_registered_msg(self, key: K) -> str:
        r"""Return the error message used when ``key`` is already
        registered."""
        return (
            f"A value is already registered for '{key}'. "
            "Use a different key or set exist_ok=True to override."
        )

    def _many_already_registered_msg(self, duplicates: set[K]) -> str:
        r"""Return the error message used when several keys are already
        registered."""
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
            self._state.clear()
            self._lazy.clear()
            self._invalidate()

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            other: The object to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.

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
        # Local import to avoid a cyclic import: coola.equality.interface's import
        # chain transitively depends on this module (e.g. via EqualityTesterRegistry).
        from coola.equality.interface import objects_are_equal  # noqa: PLC0415

        if not is_same_type(other, self):
            return False
        # The snapshots are taken one at a time and the locks are released before the
        # comparison: it can call arbitrary ``equal``/``__eq__`` code (for example a value
        # that holds another registry), which must not run while a lock is held.
        with self._lock:
            snapshot = self._get_snapshot()
        with other._lock:
            other_snapshot = other._get_snapshot()
        return objects_are_equal(snapshot, other_snapshot, equal_nan=equal_nan)

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
        with self._lock:
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
        with self._lock:
            return key in self._state

    def register(self, key: K, value: V, exist_ok: bool = False) -> None:
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
            >>> registry.register("key1", 100)  # doctest: +SKIP
            AlreadyRegisteredError: A value is already registered for 'key1'...

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
            self._state[key] = value
            self._invalidate()

    def register_many(self, mapping: Mapping[K, V], exist_ok: bool = False) -> None:
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
            self._state.update(mapping)
            self._invalidate()

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
            value = self._state.pop(key)
            self._invalidate()
            return value

    def _get_snapshot(self) -> dict[K, V]:
        r"""Return a cached, read-only-by-convention copy of ``_state``.

        This must be called while holding ``self._lock``. The copy is
        made lazily and cached on ``self._snapshot`` until the next
        mutation (see ``_invalidate``), so that repeated calls to
        ``items``/``keys``/ ``values`` between mutations reuse the same
        ``dict.copy()`` instead of paying for a fresh O(n) allocation
        every time.
        """
        self._load_lazy_all()
        if self._snapshot is None:
            self._snapshot = self._state.copy()
        return self._snapshot

    def register_lazy(
        self, names: str | Sequence[str], factory: str | Callable[[], Mapping[K, V]]
    ) -> None:
        r"""Register entries that are built the first time they are needed.

        This is used to register the entries that depend on an optional
        library without importing the library when the registry is
        created. The entries are loaded the first time the registry is
        asked for a key that is related to one of the ``names`` (see
        ``_lazy_names``; by default, all the groups are related to every
        key), or for the whole content of the registry (``len``,
        iteration, ``items``, ``repr``, ``equal``...).

        The entries that are loaded never replace the entries that are
        already registered, so a key that was registered by hand keeps
        its value. ``clear`` removes the entries that are not loaded.

        Args:
            names: The name or the names of the group of entries. The
                group is loaded when any of these names is needed.
            factory: A callable that returns the entries, or its path
                ``"package.module:name"``. A path is imported when the
                entries are loaded.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]()
            >>> registry.register_lazy("group", lambda: {"key": 42})
            >>> len(registry)
            1
            >>> registry.get("key")
            42

            ```
        """
        entry = _LazyEntry((names,) if isinstance(names, str) else tuple(names), factory)
        with self._lock:
            for name in entry.names:
                self._lazy[name] = entry

    def _lazy_names(self, key: K) -> Iterable[str]:  # noqa: ARG002
        r"""Return the names of the lazy groups related to a key.

        By default, the keys do not tell which group they come from, so
        all the groups that are not loaded are related to every key.
        Subclasses can override it to load only the groups that matter
        for the key.
        """
        return list(self._lazy)

    def _load_lazy_for(self, key: K) -> None:
        r"""Load the lazy groups that are related to a key."""
        if self._lazy:
            self._load_lazy(self._lazy_names(key))

    def _load_lazy_all(self) -> None:
        r"""Load all the lazy groups."""
        if self._lazy:
            self._load_lazy(list(self._lazy))

    def _load_lazy(self, names: Iterable[str]) -> None:
        r"""Load the lazy groups with these names.

        The entries that are already registered are not replaced.
        """
        with self._lock:
            for name in names:
                entry = self._lazy.get(name)
                if entry is None:
                    continue  # The group is loaded or has no entries for this name
                for alias in entry.names:
                    del self._lazy[alias]
                try:
                    entries = entry.load()
                except BaseException:
                    # Keep the group so the next call tries again and fails the same way,
                    # instead of silently continuing without these entries.
                    for alias in entry.names:
                        self._lazy[alias] = entry
                    raise
                for key, value in entries.items():
                    self._validate_key(key)
                    self._state.setdefault(key, value)
                self._invalidate()

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
        with self._lock:
            return self._get_snapshot().items()

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
        with self._lock:
            return self._get_snapshot().keys()

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
        with self._lock:
            return self._get_snapshot().values()
