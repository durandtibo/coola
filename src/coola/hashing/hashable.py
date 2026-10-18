r"""Define a hasher for objects that know how to hash themselves."""

from __future__ import annotations

__all__ = ["HashableHasher", "SupportsHash"]

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from coola.display import InlineDisplayMixin, NoArgsDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


@runtime_checkable
class SupportsHash(Protocol):
    r"""Implement a protocol to represent objects with a ``hash``
    method."""

    def hash(
        self,
        registry: HasherRegistry | None = None,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        r"""Return a hash uniquely identifying this object.

        Args:
            registry: The registry used to recursively hash the
                object's nested data. If ``None``, the default
                registry is used.
            length: The length of the returned hash.
            ignore_unhashable: If ``True``, unhashable values are
                replaced by a placeholder instead of raising an
                error.

        Returns:
            The hash of this object.
        """
        ...


class HashableHasher(InlineDisplayMixin, NoArgsDisplayMixin, BaseHasher[SupportsHash]):
    r"""Hasher for objects that implement their own ``hash`` method.

    This hasher delegates to the object's own ``hash`` method, passing
    through the registry so the object can recursively hash its
    nested data via ``registry`` the same way any other hasher would.
    It works for any object matching the ``SupportsHash`` protocol,
    regardless of its concrete type.

    Example:
        ```pycon
        >>> from coola.hashing import HashableHasher, get_default_registry
        >>> class MyClass:
        ...     def __init__(self, value):
        ...         self._value = value
        ...     def hash(self, registry=None, length=64, ignore_unhashable=False):
        ...         if registry is None:
        ...             registry = get_default_registry()
        ...         return registry.hash(
        ...             self._value, length=length, ignore_unhashable=ignore_unhashable
        ...         )
        ...
        >>> registry = get_default_registry()
        >>> hasher = HashableHasher()
        >>> hasher
        HashableHasher()
        >>> len(hasher.hash(MyClass(42), registry=registry))
        64

        ```
    """

    def hash(
        self,
        data: SupportsHash,
        registry: HasherRegistry,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        r"""Compute a deterministic hash by delegating to the object's
        own ``hash`` method.

        Args:
            data: The object to hash. Must implement the
                ``SupportsHash`` protocol, i.e. expose a ``hash``
                method accepting ``registry``, ``length``, and
                ``ignore_unhashable``.
            registry: The hasher registry, forwarded to ``data.hash``
                so the object can recursively hash its nested data.
            length: The desired length of the returned hex string.
                Forwarded to ``data.hash``. Defaults to 64.
            ignore_unhashable: If ``True``, nested values for which no
                hasher is registered are replaced by a deterministic
                placeholder hash instead of raising an error.
                Forwarded to ``data.hash``.

        Returns:
            A string representing the hash of ``data``: the value
            returned by ``data.hash``, namespaced by the qualified
            class name of ``data``.

        Raises:
            AttributeError: If ``data`` does not implement a ``hash``
                method.
        """
        digest = data.hash(registry=registry, length=length, ignore_unhashable=ignore_unhashable)
        cls = type(data)
        return hash_tagged(f"custom:{cls.__module__}.{cls.__qualname__}", digest, length=length)
