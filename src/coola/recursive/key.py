r"""Define the default transformer for set data (set, frozenset)."""

from __future__ import annotations

__all__ = ["KeyFilterTransformer"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from coola.recursive.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


class KeyFilterTransformer(BaseTransformer[Mapping[Any, Any]]):
    """Transformer for mappings that drops keys matching a predicate and
    recurses into the remaining values."""

    def transform(
        self,
        data: Mapping[Any, Any],
        func: Callable[[Any], bool],
        registry: TransformerRegistry,
    ) -> Mapping[Any, Any]:
        filtered = {
            key: registry.transform(value, func) for key, value in data.items() if not func(key)
        }
        return type(data)(filtered)
