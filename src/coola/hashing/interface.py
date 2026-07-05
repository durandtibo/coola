r"""Define the public interface to recursively apply a function to all
items in nested data."""

from __future__ import annotations

__all__ = ["get_default_registry", "hash_object", "register_hashers"]

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coola.hashing.datetime import DatetimeHasher
from coola.hashing.mapping import MappingHasher
from coola.hashing.path import PathHasher
from coola.hashing.registry import HasherRegistry
from coola.hashing.repr import ReprHasher
from coola.hashing.sequence import SequenceHasher
from coola.hashing.string import StringHasher

if TYPE_CHECKING:
    from coola.hashing.base import BaseHasher


def hash_object(
    data: object,
    registry: HasherRegistry | None = None,
    length: int = 64,
) -> str:
    """Compute a hash of a nested data structure.

    Args:
        data: The data to hash. Can be a nested structure such as a
            ``list``, ``dict``, or ``tuple``.
        registry: The registry used to resolve hashers for each data
            type. If ``None``, the default registry is used.
        length: The desired length of the returned hex string. Must be
            an even number between 2 and 128 inclusive. Defaults to
            ``64``.

    Returns:
        A string representing the hash of the input data.

    Example:
        ```pycon
        >>> from coola.hashing import hash_object
        >>> hash_object({"a": 1, "b": "abc"})
        '8579f51cd67c8be8fd22301d4c085e2b676c7c7d49991645a85a3c77692a1056'

        ```
    """
    if registry is None:
        registry = get_default_registry()
    return registry.hash(data, length=length)


def register_hashers(
    mapping: Mapping[type, BaseHasher[Any]],
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
        RuntimeError: If any type is already registered and ``exist_ok``
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
        >>> registry.hash("meowwwwww")
        '36a34d8fd93344d3be9a68e8c797601210f7d4585e30c102f3e8fceea38192aa'

        ```
    """
    if not hasattr(get_default_registry, "_registry"):
        registry = HasherRegistry()
        _register_default_hashers(registry)
        get_default_registry._registry = registry
    return get_default_registry._registry


def _register_default_hashers(registry: HasherRegistry) -> None:
    """Populate a registry with hashers for common Python built-in
    types.

    Args:
        registry: The registry to populate.
    """
    sequence_hasher = SequenceHasher()
    string_hasher = StringHasher()
    mapping_hasher = MappingHasher()
    datetime_hasher = DatetimeHasher()
    repr_hasher = ReprHasher()

    registry.register_many(
        {
            # Strings should not be iterated character by character
            str: string_hasher,
            # Numeric types - no recursion needed
            int: repr_hasher,
            float: repr_hasher,
            complex: repr_hasher,
            bool: repr_hasher,
            # Sequences - recursive transformation preserving order
            list: sequence_hasher,
            tuple: sequence_hasher,
            Sequence: sequence_hasher,
            # Mappings - recursive transformation of keys and values
            dict: mapping_hasher,
            Mapping: mapping_hasher,
            # Date types
            date: datetime_hasher,
            datetime: datetime_hasher,
            # Path
            Path: PathHasher(),
            # None
            type(None): repr_hasher,
        }
    )
