r"""Define the identity transformer."""

from __future__ import annotations

__all__ = ["IdentityTransformer"]

from typing import Any

from coola.recursive.base import BaseTransformer, Expansion


class IdentityTransformer(BaseTransformer[Any]):
    r"""Transformer for values that are kept as they are: the function
    is never applied.

    This transformer returns no children and rebuilds the value as is,
    so the engine does not call the function. It is useful to register
    for types that must be preserved as-is while traversing a nested
    data structure (e.g. to skip certain leaf types or a whole subtree
    instead of transforming them).

    Example:
        ```pycon
        >>> from coola.recursive import IdentityTransformer, TransformerRegistry
        >>> transformer = IdentityTransformer()
        >>> registry = TransformerRegistry({object: transformer})
        >>> registry.transform([1, 2], str)
        [1, 2]

        ```
    """

    def expand(self, data: Any) -> Expansion | None:
        r"""Describe ``data`` as a value without children that is
        rebuilt as is.

        Args:
            data: The data to keep unchanged.

        Returns:
            No children, and a function that returns ``data``.

        Example:
            ```pycon
            >>> from coola.recursive import IdentityTransformer
            >>> children, rebuild = IdentityTransformer().expand(42)
            >>> children
            ()
            >>> rebuild([])
            42

            ```
        """
        return (), lambda _: data
