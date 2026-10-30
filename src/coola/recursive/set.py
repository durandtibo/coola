r"""Define the default transformer for set data (set, frozenset)."""

from __future__ import annotations

__all__ = ["SetTransformer"]

from collections.abc import Set as AbstractSet
from typing import Any

from coola.display import InlineDisplayMixin
from coola.recursive.base import BaseTransformer, Expansion
from coola.recursive.rebuild import rebuild_set


class SetTransformer(InlineDisplayMixin, BaseTransformer[AbstractSet[Any]]):
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
        - All elements are transformed recursively by the engine
        - The original set type is preserved (set remains set, frozenset remains frozenset)
        - Element order is not guaranteed (sets are unordered)
        - Duplicate transformed values will be automatically deduplicated
        - Empty sets are preserved as empty sets of the same type
        - If transformation produces unhashable values, a TypeError will be raised

    Example:
        ```pycon
        >>> from coola.recursive import SetTransformer, TransformerRegistry, DefaultTransformer
        >>> transformer = SetTransformer()
        >>> transformer
        SetTransformer()
        >>> registry = TransformerRegistry({object: DefaultTransformer(), set: transformer})
        >>> # Transform set elements (order may vary in output)
        >>> registry.transform({1}, str)
        {'1'}
        >>> # Duplicate values after transformation are automatically deduplicated
        >>> registry.transform({1, 2, 3}, lambda x: x // 4)
        {0}

        ```
    """

    def expand(self, data: AbstractSet[Any]) -> Expansion | None:
        r"""Expand an object into its children.

        Args:
            data: The object.

        Returns:
            ``None`` if the object is a leaf. Otherwise the children and
                the function that rebuilds the set from the transformed
                children.

        Raises:
            TypeError: From the function that rebuilds the set, if a
                transformed value is not hashable.
        """
        # Note: the rebuild raises TypeError if the transformed values are not hashable
        return list(data), lambda items: rebuild_set(data, items)
