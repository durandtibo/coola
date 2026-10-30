r"""Define the default transformer."""

from __future__ import annotations

__all__ = ["DefaultTransformer"]

from typing import Any

from coola.display import InlineDisplayMixin
from coola.recursive.base import BaseTransformer, Expansion


class DefaultTransformer(InlineDisplayMixin, BaseTransformer[Any]):
    r"""Transformer for leaf nodes: the engine applies the function to
    the data.

    This is the default transformer used for values that don't require
    recursive traversal. It returns no children, so the data is a leaf.
    This transformer is typically used as the terminal case in recursive
    transformations when a leaf value (non-container) is encountered.

    Notes:
        Unlike container-specific transformers (e.g., list, dict), this
        transformer does not traverse nested structures. It treats all
        input as atomic values, even a container.

    Example:
        ```pycon
        >>> from coola.recursive import DefaultTransformer, TransformerRegistry
        >>> transformer = DefaultTransformer()
        >>> transformer
        DefaultTransformer()
        >>> # Nothing to expand: the data is a leaf
        >>> transformer.expand(42) is None
        True
        >>> # Even container types are treated as atomic values
        >>> registry = TransformerRegistry({list: transformer})
        >>> registry.transform([1, 2, 3], str)
        '[1, 2, 3]'

        ```
    """

    def expand(self, data: object) -> Expansion | None:
        r"""Expand an object into its children.

        Args:
            data: The object.

        Returns:
            ``None``: the object is a leaf.
        """
        del data  # a leaf: no children
        return None
