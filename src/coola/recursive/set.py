r"""Define the default transformer for set data (set, frozenset)."""

from __future__ import annotations

__all__ = ["SetTransformer"]

from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Any

from coola.recursive.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


class SetTransformer(BaseTransformer[AbstractSet[Any]]):
    r"""Transformer for set types that recursively transforms elements.

    This transformer handles set structures (set, frozenset) by recursively
    transforming all elements while preserving the original set type. After
    transformation, it reconstructs the set using its original type. Sets
    maintain their unordered, unique-element properties.

    Important:
        **HASHABILITY REQUIREMENT**: All transformed values MUST remain hashable
        (i.e., immutable and hashable) since sets can only contain hashable
        elements. Transforming to unhashable types (like lists or dicts) will
        raise a TypeError.

    Notes:
        - All elements are transformed recursively through the registry
        - The original set type is preserved (set remains set, frozenset remains frozenset)
        - Element order is not guaranteed (sets are unordered)
        - Duplicate transformed values will be automatically deduplicated
        - Empty sets are preserved as empty sets of the same type
        - If transformation produces unhashable values, a TypeError will be raised

    Example:
        ```pycon
        >>> from coola.recursive import SetTransformer, TransformerRegistry
        >>> registry = TransformerRegistry()
        >>> transformer = SetTransformer()
        >>> transformer
        SetTransformer()
        >>> # Transform set elements (order may vary in output)
        >>> transformer.transform({1}, func=str, registry=registry)
        {'1'}
        >>> # Frozenset type is preserved
        >>> transformer.transform(frozenset([1, 2, 3]), func=lambda x: x * 2, registry=registry)
        frozenset({2, 4, 6})
        >>> # Duplicate values after transformation are automatically deduplicated
        >>> transformer.transform({1, 2, 3}, func=lambda x: x // 4, registry=registry)
        {0}

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        data: AbstractSet[Any],
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> AbstractSet[Any]:
        # Transform all elements recursively using the registry
        # Note: This will raise TypeError if transformed values are not hashable
        transformed = {registry.transform(item, func) for item in data}

        # Rebuild with original type to preserve set characteristics
        return type(data)(transformed)
