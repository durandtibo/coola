r"""Define a transformer for dataclass instances."""

from __future__ import annotations

__all__ = ["DataclassTransformer"]

import dataclasses
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.recursive.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


class DataclassTransformer(InlineDisplayMixin, BaseTransformer[Any]):
    r"""Transformer for dataclass instances that recursively transforms
    the fields.

    The fields are transformed through the registry and the instance is
    rebuilt with ``dataclasses.replace``, so a frozen dataclass is
    supported and the input is not modified.

    Notes:
        - The fields with ``init=False`` are not passed to the
          constructor: they get the value that ``__init__`` or
          ``__post_init__`` gives them.
        - A dataclass with an ``InitVar`` without a default value
          cannot be rebuilt with ``dataclasses.replace`` and raises a
          ``ValueError``.
        - It is not registered by default: see
          ``coola.recursive.enter_dataclasses``.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.recursive import (
        ...     DataclassTransformer,
        ...     DefaultTransformer,
        ...     SequenceTransformer,
        ...     TransformerRegistry,
        ... )
        >>> @dataclass
        ... class Config:
        ...     lr: float
        ...     layers: list
        ...
        >>> transformer = DataclassTransformer()
        >>> registry = TransformerRegistry({object: DefaultTransformer(), list: SequenceTransformer()})
        >>> transformer.transform(Config(0.1, [1, 2]), func=str, registry=registry)
        Config(lr='0.1', layers=['1', '2'])

        ```
    """

    def transform(
        self,
        data: Any,
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> Any:
        changes = {
            field.name: registry.transform(getattr(data, field.name), func)
            for field in dataclasses.fields(data)
            if field.init
        }
        return dataclasses.replace(data, **changes)
