r"""Define the default transformer."""

from __future__ import annotations

__all__ = ["DefaultTransformer"]

from typing import TYPE_CHECKING, Any

from coola.recursive.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


class DefaultTransformer(BaseTransformer[Any]):
    r"""Transformer for leaf nodes that directly applies the function.

    This is the default transformer used for values that don't require
    recursive traversal. It simply applies the given function directly
    to the data without any structural transformation or nested processing.
    This transformer is typically used as the terminal case in recursive
    transformations when a leaf value (non-container) is encountered.

    Notes:
        Unlike container-specific transformers (e.g., list, dict), this
        transformer does not traverse nested structures. It treats all
        input as atomic values and applies the function directly.

    Example:
        ```pycon
        >>> from coola.recursive import DefaultTransformer, TransformerRegistry
        >>> registry = TransformerRegistry()
        >>> transformer = DefaultTransformer()
        >>> transformer
        DefaultTransformer()
        >>> # Transform a simple value directly
        >>> transformer.transform(42, func=str, registry=registry)
        '42'
        >>> # Transform a string
        >>> transformer.transform("hello", func=str.upper, registry=registry)
        'HELLO'
        >>> # Apply a mathematical operation
        >>> transformer.transform(10, func=lambda x: x * 2, registry=registry)
        20
        >>> # Even container types are treated as atomic values
        >>> transformer.transform([1, 2, 3], func=str, registry=registry)
        '[1, 2, 3]'

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        data: Any,
        func: Callable[[Any], Any],
        registry: TransformerRegistry,  # noqa: ARG002
    ) -> Any:
        return func(data)
