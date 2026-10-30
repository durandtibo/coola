r"""Define a thread-safe type-based registry with an internal cache."""

from __future__ import annotations

__all__ = ["TypeRegistry"]

import pkgutil
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, cast

from coola.registry.exceptions import (
    AlreadyRegisteredError,
    KeyNotRegisteredError,
    TypeNotRegisteredError,
)
from coola.registry.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Callable, ItemsView, Mapping, Sequence

T = TypeVar("T")


@dataclass(frozen=True)
class _LazyGroup(Generic[T]):
    r"""A group of entries that are loaded when they are needed.

    Attributes:
        packages: The top-level packages of the group. The group is
            loaded the first time a type from one of these packages is
            needed.
        factory: A callable that returns the entries, or its path
            ``"package.module:name"``. A path is not imported until the
            entries are loaded.
    """

    packages: frozenset[str]
    factory: str | Callable[[], Mapping[type, T]]

    def load(self) -> Mapping[type, T]:
        r"""Build the entries.

        Returns:
            The entries of the group.
        """
        factory = (
            pkgutil.resolve_name(self.factory) if isinstance(self.factory, str) else self.factory
        )
        return factory()


@dataclass(frozen=True)
class _NamedType:
    r"""The name of a built-in type that cannot be pickled by
    reference."""

    name: str


#: The built-in types that have no importable name (``pickle`` cannot find
#: ``builtins.mappingproxy`` or ``builtins.dict_values``), by a name that
#: ``_NamedType`` can carry.
_UNPICKLABLE_TYPES: dict[str, type] = {
    "mappingproxy": MappingProxyType,
    "dict_keys": type({}.keys()),
    "dict_values": type({}.values()),
    "dict_items": type({}.items()),
}


def _encode_type(key: type) -> type | _NamedType:
    r"""Replace a type that ``pickle`` cannot find by its name.

    Args:
        key: The type.

    Returns:
        A ``_NamedType`` if ``pickle`` cannot find the type, otherwise
            ``key``.
    """
    for name, dtype in _UNPICKLABLE_TYPES.items():
        if key is dtype:
            return _NamedType(name)
    return key


def _decode_type(key: type | _NamedType) -> type:
    r"""Reverse ``_encode_type``.

    Args:
        key: The type, or the name of a type that ``pickle`` cannot
            find.

    Returns:
        The type.
    """
    return _UNPICKLABLE_TYPES[key.name] if isinstance(key, _NamedType) else key


#: Maximum number of entries kept in the ``resolve()`` cache.
_MAX_CACHE_SIZE = 1024
_MISSING: Any = object()  # Returned by ``_resolve_uncached`` if no type matches


class TypeRegistry(Registry[type, T], Generic[T]):
    r"""A thread-safe type-based registry for storing and retrieving
    values.

    The TypeRegistry class provides a thread-safe container for mapping
    Python types to values. It supports standard dictionary operations
    through operator overloading and provides methods for safe
    registration and querying.

    The registry uses the Method Resolution Order (MRO) for type lookup
    through the `resolve()` method. When resolving a type, it
    automatically selects the most specific registered type, walking up
    the inheritance hierarchy if needed. This makes it ideal for
    type-based dispatching systems.

    The registry includes an internal cache for type resolution, bounded
    to ``_MAX_CACHE_SIZE`` (1024) entries. The cache is a plain
    dictionary that is filled under the lock and replaced by an empty
    one on any change of the registry, so cache hits in ``resolve()``
    are lock-free and do not contend under threads. When the cache is
    full, it is reset rather than evicting entries. It holds strong
    references to the types, so a dynamically created class stays alive
    until the cache is reset.

    A type that cannot be a key (every dataclass, the classes that
    define a method, the virtual subclasses of an abstract base class)
    is supported by registering a fallback predicate with
    ``register_fallback``. The order of precedence of ``resolve()`` is:
    the registered types (MRO order, ``object`` excluded), then the
    fallbacks (registration order), then ``object``.

    Note: ``resolve()`` walks ``dtype.__mro__``, which reflects real
    (static) inheritance only. Virtual subclasses registered via
    ``abc.ABCMeta.register()`` do not appear in ``__mro__``, so
    registering a value for an ABC does not make ``resolve()`` match
    that ABC's virtual subclasses even though ``isinstance`` would
    report them as instances of the ABC. See ``resolve()`` for details.

    A fallback on ``issubclass`` is deliberately not provided. Many
    built-in types are virtual subclasses of ``Sequence`` (``str``,
    ``bytes``, ``range``, ``memoryview``...), so such a fallback would
    send ``str`` to the handler registered for ``Sequence`` and, for
    instance, iterate over a string character by character. Register the
    virtual subclass explicitly instead.

    Args:
        initial_state: An optional dictionary to initialize the registry
            with. If provided, a copy is made to prevent external
            modifications. Defaults to None, which creates an empty
            registry.

    Attributes: _state: Internal dictionary storing the type-value
    pairs. _cache: Bounded dictionary of type resolution lookups, filled
    under the lock and replaced by an empty one on any change.
    _fallbacks: The ``(predicate, handler)`` pairs in registration
    order. The dictionary is replaced on write, never mutated. _lock:
    Threading lock for synchronizing access to both state and cache.

    Raises:
        TypeError: If a key is not a type.

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
        # Bounded cache for type lookups. Only writers holding the lock add an entry,
        # so readers can look up a type without locking.
        self._cache: dict[type, T] = {}
        self._fallbacks: dict[Callable[[type], bool], T] = {}
        # Replaced on write, never mutated (see ``register_lazy``).
        self._lazy: tuple[_LazyGroup[T], ...] = ()

    _not_registered_error = TypeNotRegisteredError

    def _changed(self) -> None:
        # Clear cache when registry changes to ensure new registrations are used
        self._cache = {}

    def _validate_key(self, key: type) -> None:
        if not isinstance(key, type):
            msg = f"Expected a type as key, got {key!r} of type {type(key).__qualname__}"
            raise TypeError(msg)

    def _not_registered_msg(self, key: type) -> str:
        msg = f"Type '{key}' is not registered"
        mro = getattr(key, "__mro__", None) or ()
        registered = list(self._state)
        # Two classes can have the same ``__qualname__`` (e.g. a class redefined in another
        # module): the module is added to the names that would otherwise be ambiguous.
        counts = Counter(t.__qualname__ for t in {*mro, *registered})

        def name(t: type) -> str:
            r"""Return the name of a type, qualified by its module if the
            name is ambiguous.

            Args:
                t: The type.

            Returns:
                The name of the type.
            """
            return (
                t.__qualname__
                if counts[t.__qualname__] == 1
                else f"{t.__module__}.{t.__qualname__}"
            )

        if mro:
            msg += f" (MRO: {', '.join(name(t) for t in mro)})"
        if registered:
            msg += f". Registered types: {', '.join(sorted(name(t) for t in registered))}"
        return msg

    def _already_registered_msg(self, key: type) -> str:
        return (
            f"A value is already registered for {key}. "
            "Use a different type or set exist_ok=True to override."
        )

    def _many_already_registered_msg(self, duplicates: set[type]) -> str:
        return (
            f"Types already registered: {', '.join(sorted(map(str, duplicates)))}. "
            "Use different types or set exist_ok=True to override."
        )

    def _pending_msg(self) -> str:
        lines = [
            f"(fallback) {predicate!r}: {handler}" for predicate, handler in self._fallbacks.items()
        ]
        if self._lazy:
            packages = sorted({package for group in self._lazy for package in group.packages})
            lines.append(f"<not loaded: {', '.join(packages)}>")
        return "\n  ".join(lines)

    def copy(self) -> Self:
        r"""Create a copy of the registry.

        The copy has the same entries, the same fallbacks and the same
        pending lazy groups as the registry, and is independent from it.
        The keys, the predicates and the values are not copied.

        Returns:
            A new registry of the same class.

        Example:
            ```pycon
            >>> from dataclasses import is_dataclass
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]({object: "object"})
            >>> registry.register_fallback(is_dataclass, "dataclass")
            >>> other = registry.copy()
            >>> other.unregister_fallback(is_dataclass)
            'dataclass'
            >>> len(registry.fallbacks())
            1

            ```
        """
        with self._lock:
            fallbacks, lazy = self._fallbacks, self._lazy
            new = super().copy()
        # Replaced on write, never mutated, so they can be shared
        new._fallbacks = fallbacks
        new._lazy = lazy
        return new

    def clear(self) -> None:
        r"""Remove all the entries, the fallbacks and the pending lazy
        groups from the registry.

        Example:
            ```pycon
            >>> from dataclasses import is_dataclass
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]({int: "int"})
            >>> registry.register_fallback(is_dataclass, "dataclass")
            >>> registry.clear()
            >>> len(registry), len(registry.fallbacks())
            (0, 0)

            ```
        """
        with self._lock:
            self._fallbacks = {}
            self._lazy = ()
            super().clear()

    def equal(self, other: object) -> bool:
        r"""Indicate if two objects are equal or not.

        Two registries are equal if they have the same class, the same
        entries and the same fallbacks (predicates and handlers).

        Args:
            other: The object to compare with.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from dataclasses import is_dataclass
            >>> from coola.registry import TypeRegistry
            >>> registry1 = TypeRegistry[str]({int: "int"})
            >>> registry2 = TypeRegistry[str]({int: "int"})
            >>> registry1.equal(registry2)
            True
            >>> registry2.register_fallback(is_dataclass, "dataclass")
            >>> registry1.equal(registry2)
            False

            ```
        """
        if not isinstance(other, TypeRegistry) or not super().equal(other):
            return False
        # The order of the fallbacks is their order of precedence.
        return list(self._fallbacks.items()) == list(other._fallbacks.items())

    def __getstate__(self) -> dict[str, Any]:
        # The cache is rebuilt by the lookups.
        state = super().__getstate__()
        del state["_cache"]
        state["_state"] = {_encode_type(key): value for key, value in state["_state"].items()}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state["_state"] = {_decode_type(key): value for key, value in state["_state"].items()}
        super().__setstate__(state)
        self._cache = {}

    def _load_lazy_for_equal(self, other: Registry[type, T]) -> None:
        # Two registries with the same pending groups (a registry and its copy) are compared
        # without loading: each registry would build its own values, which are not equal.
        if self._lazy != cast("TypeRegistry[T]", other)._lazy:
            super()._load_lazy_for_equal(other)

    def register_lazy(
        self, packages: str | Sequence[str], factory: str | Callable[[], Mapping[type, T]]
    ) -> None:
        r"""Register entries that are built the first time they are needed.

        This is used to register the handlers of an optional library
        without importing the library when the registry is created. The
        group is loaded the first time the registry is asked for a type
        that comes from one of the ``packages`` (the top-level package of
        the type or of one of its base classes), or for the whole content
        of the registry (``len``, iteration, ``items``, ``keys``,
        ``values``, ``equal``). ``repr``, ``copy`` and ``register_lazy``
        load nothing.

        The entries that are loaded never replace the entries that are
        already registered, so a type that was registered by hand keeps
        its value. A group whose factory raises stays pending, so the
        next access raises again. Two groups with a package in common
        both stay and are both loaded. ``clear`` removes the groups that
        are not loaded.

        Args:
            packages: The name or the names of the top-level packages of
                the group. The group is loaded when a type from any of
                these packages is needed.
            factory: A callable that returns the entries, or its path
                ``"package.module:name"``. A path is imported when the
                entries are loaded.

        Example:
            ```pycon
            >>> from decimal import Decimal
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]()
            >>> registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
            >>> registry.resolve(Decimal)
            'decimal'

            ```
        """
        group = _LazyGroup(
            frozenset((packages,) if isinstance(packages, str) else packages), factory
        )
        with self._lock:
            self._lazy = (*self._lazy, group)

    def _load_lazy_for(self, key: type) -> None:
        if not self._lazy:
            return
        # Only an object whose class (or a base class) comes from a library can need the
        # entries of this library, so the library is the first part of the module name.
        packages = {
            getattr(base, "__module__", "").partition(".")[0]
            for base in getattr(key, "__mro__", ())
        }
        self._load_groups([group for group in self._lazy if group.packages & packages])

    def _load_lazy_all(self) -> None:
        if self._lazy:
            self._load_groups(list(self._lazy))

    def _load_groups(self, groups: Sequence[_LazyGroup[T]]) -> None:
        r"""Load some lazy groups.

        The entries that are already registered are not replaced.

        Args:
            groups: The groups to load.

        Raises:
            Exception: If a group fails to load. The group is kept, so
                the next call tries again.
        """
        with self._lock:
            for group in groups:
                if not any(g is group for g in self._lazy):
                    continue  # Loaded by a factory that was called in the meantime
                self._lazy = tuple(g for g in self._lazy if g is not group)
                try:
                    entries = group.load()
                except BaseException:
                    # Keep the group so the next call tries again and fails the same way,
                    # instead of silently continuing without these entries.
                    self._lazy = (group, *self._lazy)
                    raise
                state = dict(self._state)
                for key, value in entries.items():
                    self._validate_key(key)
                    state.setdefault(key, value)
                self._publish(state)

    def register_fallback(
        self, predicate: Callable[[type], bool], value: T, *, exist_ok: bool = False
    ) -> None:
        r"""Register a value for the types accepted by a predicate.

        A fallback supports the types that cannot be a key of the
        registry, for example every dataclass, the classes that define a
        method, or the virtual subclasses of an abstract base class.
        ``resolve`` tries the fallbacks, in registration order, only if
        no registered type (``object`` excluded) matches, and before
        ``object``. The predicate must be pure and cheap: its result is
        cached per type until the next change of the registry. A
        predicate that raises propagates and is not cached.

        Args:
            predicate: A function that takes a type and returns
                ``True`` if ``value`` handles it.
            value: The value to return for the types accepted by
                ``predicate``.
            exist_ok: If ``False`` (default), raises an error if the
                predicate is already registered. If ``True``, replaces
                the value and keeps the position of the predicate.

        Raises:
            AlreadyRegisteredError: If the predicate is already
                registered and ``exist_ok`` is ``False``.

        Example:
            ```pycon
            >>> from collections.abc import Sequence
            >>> from collections import deque
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]({object: "object"})
            >>> registry.register_fallback(lambda t: issubclass(t, Sequence), "sequence")
            >>> registry.resolve(deque)
            'sequence'

            ```
        """
        with self._lock:
            if predicate in self._fallbacks and not exist_ok:
                msg = (
                    f"A fallback is already registered for {predicate!r}. "
                    "Set exist_ok=True to override."
                )
                raise AlreadyRegisteredError(msg)
            self._fallbacks = {**self._fallbacks, predicate: value}
            self._changed()

    def unregister_fallback(self, predicate: Callable[[type], bool]) -> T:
        r"""Remove a fallback and return its value.

        Args:
            predicate: The predicate given to ``register_fallback``.

        Returns:
            The value that was registered with the predicate.

        Raises:
            KeyNotRegisteredError: If the predicate is not registered.

        Example:
            ```pycon
            >>> from dataclasses import is_dataclass
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]()
            >>> registry.register_fallback(is_dataclass, "dataclass")
            >>> registry.unregister_fallback(is_dataclass)
            'dataclass'

            ```
        """
        with self._lock:
            if predicate not in self._fallbacks:
                msg = f"No fallback is registered for {predicate!r}"
                raise KeyNotRegisteredError(msg)
            fallbacks = dict(self._fallbacks)
            value = fallbacks.pop(predicate)
            self._fallbacks = fallbacks
            self._changed()
            return value

    def fallbacks(self) -> ItemsView[Callable[[type], bool], T]:
        r"""Return a snapshot of the ``(predicate, value)`` pairs in
        registration order.

        A fallback registered after the call does not change the view.

        Returns:
            A view of the ``(predicate, value)`` pairs.

        Example:
            ```pycon
            >>> from dataclasses import is_dataclass
            >>> from coola.registry import TypeRegistry
            >>> registry = TypeRegistry[str]()
            >>> registry.register_fallback(is_dataclass, "dataclass")
            >>> list(registry.fallbacks())
            [(<function is_dataclass at ...>, 'dataclass')]

            ```
        """
        return self._fallbacks.items()

    def resolve(self, dtype: type) -> T:
        r"""Resolve a type to its associated value using MRO lookup.

        This method finds the most appropriate value for a given type by
        walking the Method Resolution Order (MRO). It first checks for a
        direct match, then searches through parent types in MRO order to
        find the most specific registered type.

        Results are cached internally to optimize performance for repeated
        lookups of the same type.

        Note:
            Resolution walks ``dtype.__mro__``, i.e. real (static)
            inheritance only. Virtual subclasses registered with
            ``abc.ABCMeta.register()`` do not appear in ``__mro__``, so
            registering a value for an ABC does *not* make this method
            resolve any of that ABC's virtual subclasses, even though
            ``isinstance(instance_of_virtual_subclass, the_abc)`` is
            ``True``. To make a virtual subclass resolve to a value, either
            register the subclass itself or a real (non-virtual) ancestor of
            it.

        Args:
            dtype: The type to resolve.

        Returns:
            The value associated with the type or its nearest registered
            parent type in the MRO.

        Raises:
            TypeError: If ``dtype`` is not a type.
            TypeNotRegisteredError: If no matching type is found in the registry, including
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
        value = self._resolve(dtype)
        if value is _MISSING:
            raise TypeNotRegisteredError(self._not_registered_msg(dtype))
        return value

    def _resolve(self, dtype: type) -> T:
        r"""Find the value of a type, with the cache.

        The lookup is lock-free if the type is in the cache. A miss is
        not cached.

        Args:
            dtype: The type.

        Returns:
            The value, or the ``_MISSING`` marker if no value matches.

        Raises:
            TypeError: If ``dtype`` is not a type.
        """
        # Lock-free fast path: a hit is a single dict lookup.
        try:
            value = self._cache.get(dtype, _MISSING)
        except TypeError:  # Unhashable, rejected by _validate_key
            value = _MISSING
        if value is not _MISSING:
            return value
        with self._lock:
            self._validate_key(dtype)
            value = self._cache.get(dtype, _MISSING)  # Another thread may have filled it
            if value is not _MISSING:
                return value
            self._load_lazy_for(dtype)  # Loading entries clears the cache
            value = self._resolve_uncached(dtype)
            if value is _MISSING:
                return value  # A miss is not cached
            # The cache is only written under the lock, which a registration also
            # holds when it empties the cache, so a stale value is never stored.
            if len(self._cache) >= _MAX_CACHE_SIZE:
                self._cache = {}
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
            The appropriate value for the type or its nearest parent type, or
            ``_MISSING`` if no type matches.
        """
        # MRO lookup - finds the most specific parent type, ``object`` excluded: it is
        # the last resort, after the fallbacks.
        # ``issubclass`` is not used on purpose: it is slower and ambiguous when
        # several registered types match, and the MRO gives a deterministic "most
        # specific first" order. The consequence is that an abstract base class
        # (e.g. ``Sequence``) only matches the types that really inherit from it, not
        # the virtual subclasses (``list``, ``tuple``, ``deque``): these types must be
        # registered explicitly, or matched by a fallback.
        state = self._state
        for base_type in dtype.__mro__:
            if base_type is not object and base_type in state:
                return state[base_type]

        for predicate, value in self._fallbacks.items():
            if predicate(dtype):
                return value

        if object in state:
            return state[object]

        return _MISSING
