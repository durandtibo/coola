r"""Define a thread-safe registry."""

from __future__ import annotations

__all__ = ["Registry"]

from typing import TypeVar

from coola.registry.base import BaseRegistry

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class Registry(BaseRegistry[K, V]):
    r"""A thread-safe generic key-value registry for storing and managing
    typed mappings.

    The Registry class provides a type-safe container for registering and
    retrieving values by key. It supports all standard dictionary operations
    through operator overloading and provides additional methods for safe
    registration and querying. All operations are protected by a lock to
    ensure thread safety in concurrent environments.

    Args:
        initial_state: An optional dictionary to initialize the registry with.
            If provided, a copy is made to prevent external modifications.
            Defaults to None, which creates an empty registry.

    Attributes:
        _state: Internal dictionary storing the key-value pairs.
        _lock: Threading lock for synchronizing access to the registry.

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
