r"""Define a conditional transformer to add filtering without changing
the core design."""

from __future__ import annotations

__all__ = ["ConditionalTransformer"]

from typing import TYPE_CHECKING, Any, TypeVar

from coola.recursive.base import BaseTransformer
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


T = TypeVar("T")


class ConditionalTransformer(BaseTransformer[T]):
    r"""Wrapper transformer that conditionally applies transformations.

    This transformer wraps another transformer and only applies it when
    a given condition evaluates to True. If the condition is False, the
    data is returned unchanged. This allows for selective transformation
    based on runtime checks without modifying the underlying transformer
    or core architecture.

    Args:
        transformer: The underlying transformer to apply when the condition
            is met. This can be any BaseTransformer implementation.
        condition: A predicate function that determines whether to apply
            the transformation. Should accept the data as input and return
            True to transform or False to pass through unchanged.

    Example:
        ```pycon
        >>> from coola.recursive import (
        ...     DefaultTransformer,
        ...     ConditionalTransformer,
        ...     TransformerRegistry,
        ... )
        >>> registry = TransformerRegistry()
        >>> # Create a transformer that only processes positive numbers
        >>> transformer = ConditionalTransformer(
        ...     transformer=DefaultTransformer(),
        ...     condition=lambda x: isinstance(x, (int, float)) and x > 0,
        ... )
        >>> transformer
        ConditionalTransformer(
          (transformer): DefaultTransformer()
          (condition): <function <lambda> at 0x...>
        )
        >>> # Positive number: condition passes, transformation applied
        >>> transformer.transform(5, func=lambda x: x * 2, registry=registry)
        10
        >>> # Negative number: condition fails, returned unchanged
        >>> transformer.transform(-5, func=lambda x: x * 2, registry=registry)
        -5
        >>> # Non-numeric: condition fails, returned unchanged
        >>> transformer.transform("text", func=lambda x: x * 2, registry=registry)
        'text'

        ```
    """

    def __init__(
        self,
        transformer: BaseTransformer[T],
        condition: Callable[[Any], bool],
    ) -> None:
        self._transformer = transformer
        self._condition = condition

    def __repr__(self) -> str:
        params = {"transformer": self._transformer, "condition": self._condition}
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(params))}\n)"

    def __str__(self) -> str:
        params = {"transformer": self._transformer, "condition": self._condition}
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(params))}\n)"

    def transform(
        self,
        data: T,
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> Any:
        if self._condition(data):
            return self._transformer.transform(data, func, registry)
        return data
