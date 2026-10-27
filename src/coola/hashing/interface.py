r"""Define the public interface to recursively apply a function to all
items in nested data."""

from __future__ import annotations

__all__ = ["get_default_hasher_registry", "get_default_registry", "hash_object", "register_hashers"]

import dataclasses
import inspect
import warnings
from collections import OrderedDict, UserString, deque
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import PurePath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, overload
from uuid import UUID
from zoneinfo import ZoneInfo

from coola.hashing.bytes import BytesHasher
from coola.hashing.canonical import DecimalHasher, FractionHasher, RangeHasher
from coola.hashing.dataclass import DataclassHasher
from coola.hashing.datetime import DatetimeHasher
from coola.hashing.enum import EnumHasher
from coola.hashing.hashable import HashableHasher
from coola.hashing.mapping import MappingHasher
from coola.hashing.path import PathHasher
from coola.hashing.registry import HasherRegistry
from coola.hashing.repr import ReprHasher
from coola.hashing.sequence import SequenceHasher
from coola.hashing.set import SetHasher
from coola.hashing.str import StrHasher
from coola.hashing.string import StringHasher, validate_hash_length
from coola.utils.imports import (
    package_available,
)
from coola.utils.singleton import LazySingleton, make_default_registry_singleton

if TYPE_CHECKING:
    from coola.hashing.base import Hasher


# (names of the library in the module of the types, module with the hashers)
_LAZY_HASHERS = (
    (("jax", "jaxlib"), "coola.hashing.jax"),
    (("numpy",), "coola.hashing.numpy"),
    (("pandas",), "coola.hashing.pandas"),
    (("polars",), "coola.hashing.polars"),
    (("pyarrow",), "coola.hashing.pyarrow"),
    (("pydantic",), "coola.hashing.pydantic"),
    (("torch",), "coola.hashing.torch"),
    (("xarray",), "coola.hashing.xarray"),
)

_LEGACY_POSITIONAL_NAMES = ("registry", "length", "ignore_unhashable")
_DEFAULTS: dict[str, Any] = {"registry": None, "length": 64, "ignore_unhashable": False}


@overload
def hash_object(
    data: object,
    *,
    registry: HasherRegistry | None = None,
    length: int = 64,
    ignore_unhashable: bool = False,
) -> str: ...


@overload
def hash_object(
    data: object,
    registry: HasherRegistry | None,
    length: int = 64,
    ignore_unhashable: bool = False,
) -> str: ...


def hash_object(
    data: object,
    *args: Any,
    registry: HasherRegistry | None = None,
    length: int = 64,
    ignore_unhashable: bool = False,
) -> str:
    """Compute a hash of a nested data structure.

    The hash is deterministic and depends on the type of every value, on
    the order of ordered containers, and not on the order of unordered
    ones. For example ``[1, 2]``, ``(1, 2)`` and ``{1, 2}`` all hash
    differently, while ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}``
    hash the same. An ``OrderedDict`` is the exception among mappings:
    the order of its keys matters, so two ``OrderedDict`` with the same
    items in a different order hash differently.

    Note:
        The hash of a dataclass, a pydantic model, an enum member or an
        object with a ``hash`` method includes the module and the
        qualified name of its class. The hash changes when the class is
        renamed or moved to another module, and a class defined in a
        script run as ``__main__`` hashes differently from the same
        class imported. Do not persist these hashes (or the IDs derived
        from them, such as ``generate_stable_uuid5``) across such a
        change.

    The options must be passed by keyword. Passing ``registry``,
    ``length`` or ``ignore_unhashable`` as positional arguments is
    deprecated and emits a ``DeprecationWarning``: it was easy to write
    ``hash_object(data, 32)`` and pass ``32`` as the registry.

    Args:
        data: The data to hash. Can be a nested structure such as a
            ``list``, ``dict``, or ``tuple``.
        *args: Deprecated positional form of ``registry``, ``length``
            and ``ignore_unhashable``, in this order.
        registry: The registry used to resolve hashers for each data
            type. If ``None``, the default registry is used.
        length: The desired length of the returned hex string. Must be
            an even number between 2 and 128 inclusive. Defaults to
            ``64``.
        ignore_unhashable: If ``True``, objects for which no hasher is
            registered (including nested objects found while hashing
            sequences or mappings) are replaced by a deterministic
            placeholder hash instead of raising an error. The
            placeholder only depends on the type, so it is lossy, and a
            ``UserWarning`` is emitted each time one is used. Defaults
            to ``False``, which raises a ``KeyError`` when an
            unhashable object is encountered.

    Returns:
        A string representing the hash of the input data.

    Raises:
        KeyError: If ``data`` (or a nested object within it) has a
            type for which no hasher is registered and
            ``ignore_unhashable`` is ``False``.
        ValueError: If ``length`` is not an even number between 2 and
            128, or if ``data`` contains itself (a self-referencing
            container).

    Example:
        ```pycon
        >>> from coola.hashing import hash_object
        >>> len(hash_object({"a": 1, "b": "abc"}))
        64
        >>> len(hash_object({"a": 1, "b": "abc"}, length=16))
        16

        ```
    """
    if args:
        registry, length, ignore_unhashable = _resolve_legacy_arguments(
            args, registry=registry, length=length, ignore_unhashable=ignore_unhashable
        )
    validate_hash_length(length)  # before the (possibly expensive) traversal
    if registry is None:
        registry = get_default_registry()
    return registry.hash(data, length=length, ignore_unhashable=ignore_unhashable)


def _resolve_legacy_arguments(
    args: tuple[Any, ...], **kwargs: Any
) -> tuple[HasherRegistry | None, int, bool]:
    r"""Map the deprecated positional arguments of ``hash_object`` to the
    keyword arguments.

    Raises:
        TypeError: if there are too many positional arguments, or an
            argument is given twice (by position and by keyword), or the
            second positional argument is not a registry (the common
            mistake ``hash_object(data, 32)``).
    """
    if len(args) > len(_LEGACY_POSITIONAL_NAMES):
        msg = (
            f"hash_object() takes at most {len(_LEGACY_POSITIONAL_NAMES) + 1} positional "
            f"arguments but {len(args) + 1} were given"
        )
        raise TypeError(msg)
    if args[0] is not None and not isinstance(args[0], HasherRegistry):
        msg = (
            f"The second positional argument of hash_object() is `registry`, which must be a "
            f"HasherRegistry or None, got {type(args[0]).__qualname__}. "
            "Pass `length` and `ignore_unhashable` by keyword, e.g. `length=32`."
        )
        raise TypeError(msg)
    warnings.warn(
        "Passing `registry`, `length` and `ignore_unhashable` to `hash_object` as positional "
        "arguments is deprecated, use keyword arguments",
        DeprecationWarning,
        stacklevel=3,
    )
    for name, value in zip(_LEGACY_POSITIONAL_NAMES, args, strict=False):
        if kwargs[name] != _DEFAULTS[name]:
            msg = f"hash_object() got multiple values for argument {name!r}"
            raise TypeError(msg)
        kwargs[name] = value
    return kwargs["registry"], kwargs["length"], kwargs["ignore_unhashable"]


def register_hashers(
    mapping: Mapping[type, Hasher[Any]],
    exist_ok: bool = False,
) -> None:
    """Register custom hashers into the default global registry.

    This function extends the singleton registry returned by
    ``get_default_registry`` with additional type-to-hasher mappings.

    Args:
        mapping: Mapping of Python types to hasher instances.
        exist_ok: If ``False`` (default), raises an error if any type
            is already registered. If ``True``, overwrites existing
            registrations silently.

    Raises:
        AlreadyRegisteredError: If any type is already registered and ``exist_ok``
            is ``False``.

    Example:
        ```pycon
        >>> from coola.hashing import register_hashers, StrHasher
        >>> class MyType:
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        >>> register_hashers({MyType: StrHasher()})

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> HasherRegistry:
    """Return the default global hasher registry.

    The registry is created on the first call and reused on all
    subsequent calls (singleton pattern). It is pre-configured with
    hashers for common Python built-in types.

    Returns:
        A singleton ``HasherRegistry`` configured for common Python
        built-in types.

    Notes:
        Because the registry is a singleton, modifications to it
        (e.g. via ``register_hashers``) affect all future calls to
        this function. If you need an isolated registry, create a new
        ``HasherRegistry`` instance directly.

    Example:
        ```pycon
        >>> from coola.hashing import get_default_registry
        >>> registry = get_default_registry()
        >>> len(registry.hash("meowwwwww"))
        64

        ```
    """
    return _default_registry.get()


get_default_hasher_registry = get_default_registry


def get_stable_registry() -> HasherRegistry:
    """Return the registry of the stable identifiers.

    This registry only has the hashers that are built in ``coola``. Unlike
    the default registry, it does not load the ``coola.hashing.hashers``
    entry points of the installed packages and it is not changed by
    ``register_hashers``, so the identifiers computed with it depend on the
    version of ``coola`` and not on the installed packages or on the code
    that runs in the process. It is the registry used by
    ``generate_stable_content_id`` and ``generate_stable_uuid5`` when no
    registry is given.

    Returns:
        A singleton ``HasherRegistry`` with the built-in hashers.

    Notes:
        The registry is shared by the process: do not register hashers in
        it. Pass your own registry to the functions that accept one to opt
        out of the stability.

    Example:
        ```pycon
        >>> from coola.hashing import get_stable_registry
        >>> registry = get_stable_registry()
        >>> len(registry.hash("meowwwwww"))
        64

        ```
    """
    return _stable_registry.get()


def _register_default_hashers(registry: HasherRegistry) -> None:
    """Populate a registry with hashers for common Python built-in
    types.

    Args:
        registry: The registry to populate.
    """
    string_hasher = StringHasher()
    bytes_hasher = BytesHasher()
    datetime_hasher = DatetimeHasher()
    repr_hasher = ReprHasher()

    registry.register_many(
        {
            # Strings should not be iterated character by character
            str: string_hasher,
            UserString: string_hasher,
            bytes: bytes_hasher,
            bytearray: bytes_hasher,
            memoryview: bytes_hasher,
            # Numeric types - no recursion needed
            int: repr_hasher,
            float: repr_hasher,
            complex: repr_hasher,
            bool: repr_hasher,
            type(None): ReprHasher("NoneType"),
            Enum: EnumHasher(),
            # Ordered containers - one tag per family, so a list and a
            # tuple with the same items hash differently
            list: SequenceHasher("list"),
            tuple: SequenceHasher("tuple"),
            deque: SequenceHasher("deque"),
            Sequence: SequenceHasher("sequence"),
            # Mappings - recursive transformation of keys and values
            dict: MappingHasher("dict"),
            OrderedDict: MappingHasher("OrderedDict", ordered=True),
            Mapping: MappingHasher("mapping"),
            MappingProxyType: MappingHasher("mappingproxy"),
            # Unordered containers
            set: SetHasher("set"),
            frozenset: SetHasher("frozenset"),
            # Date and time types
            date: datetime_hasher,
            datetime: datetime_hasher,
            time: datetime_hasher,
            timedelta: ReprHasher("timedelta"),
            timezone: ReprHasher("timezone"),
            # Path (hashed as written, without touching the filesystem)
            PurePath: PathHasher(),
        }
    )
    registry.register_many(_get_value_hashers())
    # The types that cannot be registered one by one: any dataclass, and any class
    # that defines a ``hash`` method. The type lookup comes first, so a hasher
    # registered for a type wins.
    registry.register_fallback(dataclasses.is_dataclass, DataclassHasher())
    registry.register_fallback(_has_hash_method, HashableHasher())
    # The hashers of the optional libraries are loaded the first time the registry needs a type
    # of the library, so the libraries are not imported with ``coola.hashing``.
    for names, module in _LAZY_HASHERS:
        if package_available(names[0]):
            registry.register_lazy(names, f"{module}:get_default_hashers")


def _has_hash_method(data_type: type) -> bool:
    r"""Indicate if the instances of a type have a ``hash`` method (see
    ``SupportsHash``)."""
    method = getattr(data_type, "hash", None)
    if not callable(method):
        return False
    # A ``hash`` method with another signature (``polars.Series.hash(seed=...)``)
    # is not the ``SupportsHash`` protocol.
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    kwargs = {"registry": None, "length": 64, "ignore_unhashable": False}
    for args in ((None,), ()):  # with and without ``self``
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return True
    return False


def _get_value_hashers() -> dict[type, Hasher[Any]]:
    r"""Get the hashers of the standard library values that are not
    containers."""
    return {
        # ``str`` is canonical for these types
        UUID: StrHasher("UUID"),
        ZoneInfo: StrHasher("ZoneInfo"),
        # The equal values must have the same hash
        Decimal: DecimalHasher(),
        Fraction: FractionHasher(),
        range: RangeHasher(),
    }


_default_registry: LazySingleton[HasherRegistry] = make_default_registry_singleton(
    HasherRegistry, _register_default_hashers, plugin_group="coola.hashing.hashers"
)


_stable_registry: LazySingleton[HasherRegistry] = make_default_registry_singleton(
    HasherRegistry, _register_default_hashers
)
