r"""Define a simple registry."""

from __future__ import annotations

__all__ = ["Registry"]

from typing import TYPE_CHECKING, Generic, TypeVar

from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class Registry(Generic[K, V]):
    r"""Define a simple registry.

    Args:
        initial_state: Optional dictionary representing the initial state.

    Example:
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
    """

    def __init__(self, initial_state: dict[K, V] | None = None) -> None:
        self._registry: dict[K, V] = initial_state.copy() if initial_state else {}

    def __contains__(self, key: K) -> bool:
        return self.has(key)

    def __getitem__(self, key: K) -> V:
        return self.get(key)

    def __setitem__(self, key: K, value: V) -> None:
        self.register(key, value, exist_ok=True)

    def __delitem__(self, key: K) -> None:
        self.unregister(key)

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(self._registry))}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self._registry))}\n)"

    def clear(self) -> None:
        """Remove all entries from the registry.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]()
            >>> registry.register("key1", 42)
            >>> registry.has("key1")
            True
            >>> registry.clear()
            >>> registry.has("key1")
            False

            ```
        """
        self._registry.clear()

    def get(self, key: K) -> V:
        """Get the value for the given key.

        Args:
            key: The key to retrieve

        Returns:
            The value associated with the key

        Raises:
            KeyError: If the key is not registered

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]()
            >>> registry.register("key1", 42)
            >>> registry.get("key1")
            42

            ```
        """
        if key not in self._registry:
            msg = f"Key '{key}' is not registered"
            raise KeyError(msg)
        return self._registry[key]

    def has(self, key: K) -> bool:
        """Check if a value is registered for the given key.

        Args:
            key: The key to check

        Returns:
            ``True`` if the key exists in the registry, ``False`` otherwise.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]()
            >>> registry.register("key1", 42)
            >>> registry.has("key1")
            True

            ```
        """
        return key in self._registry

    def register(self, key: K, value: V, exist_ok: bool = False) -> None:
        """Register a new value for the given key.

        Args:
            key: The key to register
            value: The value to associate with the key
            exist_ok: If False (default), raises an error if the key is already
                registered. If True, overwrites the existing registration silently.

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]()
            >>> registry.register("key1", 42)
            >>> registry.get("key1")
            42

            ```
        """
        if key in self._registry and not exist_ok:
            msg = (
                f"A value is already registered for '{key}'. "
                "Use a different key or set exist_ok=True to override."
            )
            raise RuntimeError(msg)
        self._registry[key] = value

    def register_many(self, mapping: Mapping[K, V], exist_ok: bool = False) -> None:
        """Register multiple key-value pairs at once.

        Args:
            mapping: Dictionary of keys and values to register
            exist_ok: If False, raises error if any key already exists

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]()
            >>> registry.register_many({"key1": 42, "key2": 1})
            >>> registry.get("key1")
            42
            >>> registry.get("key2")
            1

            ```
        """
        for key, value in mapping.items():
            self.register(key, value, exist_ok=exist_ok)

    def unregister(self, key: K) -> V:
        """Unregister the value for the given key.

        Args:
            key: The key to unregister

        Returns:
            The value that was unregistered

        Raises:
            KeyError: If the key is not registered

        Example:
            ```pycon
            >>> from coola.registry import Registry
            >>> registry = Registry[str, int]()
            >>> registry.register("key1", 42)
            >>> registry.has("key1")
            True
            >>> value = registry.unregister("key1")
            >>> value
            42
            >>> registry.has("key1")
            False

            ```
        """
        if key not in self._registry:
            msg = f"Key '{key}' is not registered"
            raise KeyError(msg)
        return self._registry.pop(key)
