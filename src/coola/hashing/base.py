r"""Define the hasher base class."""

from __future__ import annotations

__all__ = ["BaseHasher"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry

T = TypeVar("T")


class BaseHasher(ABC, Generic[T]):
    r"""Base class for type-specific hashers.

    This abstract base class defines the interface for hashers that
    compute a hash value for a specific data type. Each concrete
    hasher implementation handles a specific data type and knows how
    to hash its nested elements recursively.

    Notes:
        Subclasses must implement the ``hash`` method to define how
        their specific type should be hashed.

    Example:
        ```pycon
        >>> from coola.hashing import DefaultHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = DefaultHasher()
        >>> hasher
        DefaultHasher()
        >>> hasher.hash([1, 2, 3], registry=registry)

        ```
    """

    @abstractmethod
    def hash(
        self,
        data: T,
        registry: HasherRegistry,
    ) -> str:
        r"""Hash the given data structure recursively.

        This method traverses the data structure and computes a hash
        value for it. The registry is used to resolve appropriate
        hashers for nested data types encountered during traversal.

        Args:
            data: The data structure to hash. Must be of the type ``T``
                that this hasher handles.
            registry: The hasher registry used to look up hashers
                for nested data structures of different types.

        Returns:
            A string representing the hash of the input data.

        Example:
            ```pycon
            >>> from coola.hashing import DefaultHasher, HasherRegistry
            >>> registry = HasherRegistry()
            >>> hasher = DefaultHasher()
            >>> hasher.hash("Meowwww", registry=registry)

            ```
        """
