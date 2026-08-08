r"""Define the default transformer for set data (set, frozenset)."""

from __future__ import annotations

__all__ = ["IdentityTransformer"]

from typing import TYPE_CHECKING, Any

from coola.recursive.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


class IdentityTransformer(BaseTransformer[Any]):
    """Transformer for leaf nodes that passes the value through
    unchanged, ignoring ``func``."""

    def transform(
        self,
        data: Any,
        func: Callable[[Any], bool],  # noqa: ARG002
        registry: TransformerRegistry,  # noqa: ARG002
    ) -> Any:
        return data
