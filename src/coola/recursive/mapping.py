r"""Define the default transformer for mapping data (dict)."""

from __future__ import annotations

__all__ = ["MappingTransformer"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from coola.recursive.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


class MappingTransformer(BaseTransformer[Mapping[Any, Any]]):
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
        >>> from coola.recursive import MappingTransformer, TransformerRegistry
        >>> registry = TransformerRegistry()
        >>> transformer = MappingTransformer()
        >>> transformer
        MappingTransformer()
        >>> # Transform simple dict values
        >>> transformer.transform({"a": 1, "b": 2}, func=str, registry=registry)
        {'a': '1', 'b': '2'}
        >>> # Keys remain unchanged
        >>> transformer.transform({1: "x", 2: "y"}, func=str.upper, registry=registry)
        {1: 'X', 2: 'Y'}
        >>> # Nested structures in values are handled recursively
        >>> transformer.transform({"nums": [1, 2, 3], "text": "hello"}, func=str, registry=registry)
        {'nums': '[1, 2, 3]', 'text': 'hello'}
        >>> # Empty mappings are preserved
        >>> transformer.transform({}, func=str, registry=registry)
        {}

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        data: Mapping[Any, Any],
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> Mapping[Any, Any]:
        # Transform all values recursively using the registry
        transformed = {key: registry.transform(value, func) for key, value in data.items()}

        # Rebuild with original type to preserve mapping characteristics
        return type(data)(transformed)
