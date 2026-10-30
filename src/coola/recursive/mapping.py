r"""Define the default transformer for mapping data (dict)."""

from __future__ import annotations

__all__ = ["MappingTransformer"]

from collections.abc import Mapping
from typing import Any

from coola.display import InlineDisplayMixin
from coola.recursive.base import BaseTransformer, Expansion
from coola.recursive.rebuild import rebuild_mapping


class MappingTransformer(InlineDisplayMixin, BaseTransformer[Mapping[Any, Any]]):
    r"""Transformer for mapping types that recursively transforms values.

    This transformer handles dict-like mapping structures by recursively
    transforming all values while preserving keys unchanged. After
    transformation, it reconstructs the mapping using its original type
    (dict, OrderedDict, defaultdict, etc.), maintaining the mapping's
    specific characteristics and behavior.

    Notes:
        - Keys are never transformed, only values are processed recursively
        - The original mapping type is preserved in the output
        - Nested mappings and other containers in values are handled recursively
        - Empty mappings are preserved as empty mappings of the same type

    Example:
        ```pycon
        >>> from coola.recursive import MappingTransformer, TransformerRegistry, DefaultTransformer
        >>> transformer = MappingTransformer()
        >>> transformer
        MappingTransformer()
        >>> children, rebuild = transformer.expand({"a": 1, "b": 2})
        >>> children
        [1, 2]
        >>> rebuild([10, 20])
        {'a': 10, 'b': 20}
        >>> registry = TransformerRegistry({object: DefaultTransformer(), dict: transformer})
        >>> # Transform simple dict values
        >>> registry.transform({"a": 1, "b": 2}, str)
        {'a': '1', 'b': '2'}
        >>> # Keys remain unchanged
        >>> registry.transform({1: "x", 2: "y"}, str.upper)
        {1: 'X', 2: 'Y'}
        >>> # Empty mappings are preserved
        >>> registry.transform({}, str)
        {}

        ```
    """

    def expand(self, data: Mapping[Any, Any]) -> Expansion | None:
        return list(data.values()), lambda items: rebuild_mapping(
            data, dict(zip(data, items, strict=True))
        )
