r"""Define the identity transformer."""

from __future__ import annotations

__all__ = ["IdentityTransformer"]

from typing import TYPE_CHECKING, Any

from coola.recursive.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


class IdentityTransformer(BaseTransformer[Any]):
    r"""Transformer for leaf nodes that passes the value through
    unchanged, ignoring ``func``.

    This transformer never applies the transformation function. It is
    useful to register for types that must be preserved as-is while
    traversing a nested data structure (e.g. to skip certain leaf
    types instead of transforming them).

    Example:
        ```pycon
        >>> from coola.recursive import IdentityTransformer, TransformerRegistry
        >>> registry = TransformerRegistry()
        >>> transformer = IdentityTransformer()
        >>> transformer.transform(42, func=str, registry=registry)
        42

        ```
    """

    def transform(
        self,
        data: Any,
        func: Callable[[Any], bool],  # noqa: ARG002
        registry: TransformerRegistry,  # noqa: ARG002
    ) -> Any:
        r"""Return the data unchanged, ignoring ``func``.

        Args:
            data: The data to return unchanged.
            func: Unused. Accepted only to satisfy the
                ``BaseTransformer`` interface.
            registry: Unused. Accepted only to satisfy the
                ``BaseTransformer`` interface.

        Returns:
            The input ``data``, unchanged.
        """
        return data
