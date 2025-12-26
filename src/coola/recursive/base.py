r"""Define the transformer base class."""

from __future__ import annotations

__all__ = ["BaseTransformer"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry

T = TypeVar("T")


class BaseTransformer(ABC, Generic[T]):
    r"""Base class for type-specific transformers.

    This abstract base class defines the interface for transformers that
    recursively apply functions to nested data structures. Each concrete
    transformer implementation handles a specific data type and knows how
    to reconstruct that type after transforming its nested elements.

    Notes:
        Subclasses must implement the `transform` method to define how
        their specific type should be traversed and reconstructed.

    Example:
        ```pycon
        >>> from coola.recursive import DefaultTransformer, TransformerRegistry
        >>> registry = TransformerRegistry()
        >>> transformer = DefaultTransformer()
        >>> transformer
        DefaultTransformer()
        >>> transformer.transform([1, 2, 3], func=str, registry=registry)
        '[1, 2, 3]'

        ```
    """

    @abstractmethod
    def transform(
        self,
        data: T,
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> Any:
        r"""Transform data by recursively applying a function.

        This method traverses the data structure, applies the given function
        to leaf values, and reconstructs the original structure with the
        transformed values. The registry is used to resolve appropriate
        transformers for nested data types encountered during traversal.

        Args:
            data: The data structure to transform. Must be of type T that
                this transformer handles.
            func: A function to apply to leaf values (non-container elements).
                Should accept a single argument and return the transformed value.
            registry: The transformer registry used to look up transformers
                for nested data structures of different types.

        Returns:
            The transformed data structure, maintaining the original type
            and structure but with leaf values transformed by func.

        Example:
            ```pycon
            >>> from coola.recursive import DefaultTransformer, SequenceTransformer, TransformerRegistry
            >>> registry = TransformerRegistry()
            >>> transformer = DefaultTransformer()
            >>> # Convert numeric values to strings
            >>> transformer.transform([1, 2, 3], func=str, registry=registry)
            '[1, 2, 3]'
            >>> # Apply a mathematical operation
            >>> transformer.transform([1, 2, 3], func=lambda x: x * 2, registry=registry)
            [1, 2, 3, 1, 2, 3]

            ```
        """
