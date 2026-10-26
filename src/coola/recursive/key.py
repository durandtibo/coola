r"""Define the mapping transformer that filters entries by key."""

from __future__ import annotations

__all__ = ["KeyFilterTransformer"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.recursive.base import BaseTransformer, Expansion
from coola.recursive.rebuild import rebuild_mapping

if TYPE_CHECKING:
    from collections.abc import Callable


class KeyFilterTransformer(InlineDisplayMixin, BaseTransformer[Mapping[Any, Any]]):
    r"""Transformer for mappings that drops entries whose key matches a
    predicate and recurses into the remaining values.

    The predicate is stored in the transformer and used for the keys of
    every mapping it transforms, however deep they are nested. The
    function given to ``recursive_apply`` is the function for the
    values: it is applied to the leaves.

    Args:
        key_predicate: A function applied to each key: the entries whose
            key satisfies it are dropped.

    Notes:
        - Entries whose key satisfies the predicate (i.e. the predicate
          returns a truthy value) are removed from the output.
        - Keys of the entries that are kept are preserved unchanged.

    Example:
        ```pycon
        >>> from coola.recursive import (
        ...     KeyFilterTransformer,
        ...     DefaultTransformer,
        ...     MappingTransformer,
        ...     TransformerRegistry,
        ... )
        >>> transformer = KeyFilterTransformer(key_predicate=lambda key: key == "secret")
        >>> registry = TransformerRegistry(
        ...     {object: DefaultTransformer(), dict: transformer}
        ... )
        >>> # The function of ``transform`` is applied to the values.
        >>> registry.transform({"keep": 1, "secret": 2, "nested": {"secret": 3, "x": 4}}, str)
        {'keep': '1', 'nested': {'x': '4'}}

        ```
    """

    def __init__(self, key_predicate: Callable[[Any], bool]) -> None:
        self._key_predicate = key_predicate

    def _display_fields(self) -> dict[str, Any]:
        return {"key_predicate": self._key_predicate}

    def expand(self, data: Mapping[Any, Any]) -> Expansion | None:
        predicate = self._key_predicate
        keys = [key for key in data if not predicate(key)]
        return [data[key] for key in keys], lambda items: rebuild_mapping(
            data, dict(zip(keys, items, strict=True))
        )
