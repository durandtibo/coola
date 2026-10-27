r"""Define the dataclass hasher."""

from __future__ import annotations

__all__ = ["DataclassHasher"]

import dataclasses
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class DataclassHasher(InlineDisplayMixin, BaseHasher[Any]):
    r"""Hasher for dataclass instances.

    The hash combines the qualified class name with the hash of a
    mapping from field name to field value, with every value hashed
    through the registry. Only the fields with ``compare=True`` are
    hashed, like in ``objects_are_equal``, so two equal dataclasses have
    the same hash. Because dataclasses have no common base
    class, it cannot be registered for a type: the default registry
    uses it as a fallback (see ``HasherRegistry.register_fallback``)
    for every dataclass that has no hasher of its own.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.hashing import DataclassHasher, get_default_registry
        >>> @dataclass
        ... class Point:
        ...     x: int
        ...     y: int
        ...
        >>> registry = get_default_registry()
        >>> hasher = DataclassHasher()
        >>> hasher
        DataclassHasher()
        >>> len(hasher.hash(Point(1, 2), registry=registry))
        64

        ```
    """

    def hash(
        self,
        data: Any,
        registry: HasherRegistry,
        *,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        r"""Compute a deterministic hash of a dataclass instance.

        Args:
            data: The dataclass instance to hash.
            registry: The hasher registry used to hash the field
                values.
            length: The desired length of the returned hex string. Must
                be an even number between 2 and 128 inclusive. Defaults
                to 64.
            ignore_unhashable: If ``True``, field values for which no
                hasher is registered are replaced by a deterministic
                placeholder hash instead of raising an error.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            TypeError: If ``data`` is not a dataclass instance.
            KeyError: If a field value has a type for which no hasher
                is registered in ``registry`` and ``ignore_unhashable``
                is ``False``.
            ValueError: If ``length`` is not an even number between 2
                and 128.
        """
        if not dataclasses.is_dataclass(data) or isinstance(data, type):
            msg = f"Expected a dataclass instance, got {type(data)!r}."
            raise TypeError(msg)
        fields = {
            field.name: getattr(data, field.name)
            for field in dataclasses.fields(data)
            if field.compare
        }
        digest = registry.hash(fields, length=length, ignore_unhashable=ignore_unhashable)
        cls = type(data)
        return hash_tagged(f"dataclass:{cls.__module__}.{cls.__qualname__}", digest, length=length)
