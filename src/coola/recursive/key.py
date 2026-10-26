r"""Define the mapping transformer that filters entries by key."""

from __future__ import annotations

__all__ = ["KeyFilterTransformer"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.recursive.base import BaseTransformer
from coola.recursive.rebuild import rebuild_mapping

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


class KeyFilterTransformer(InlineDisplayMixin, BaseTransformer[Mapping[Any, Any]]):
    r"""Transformer for mappings that drops entries whose key matches a
    predicate and recurses into the remaining values.

    The predicate is stored in the transformer and used for the keys of
    every mapping it transforms, however deep they are nested. The
    ``func`` given to ``transform`` is the function for the values: it
    is forwarded unchanged to the registry and applied by the
    transformers of the leaf types.

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

    def transform(
        self,
        data: Mapping[Any, Any],
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> Mapping[Any, Any]:
        r"""Drop the entries whose key matches the predicate and
        transform the remaining values recursively.

        Args:
            data: The mapping to filter and transform.
            func: The function applied to the values by the
                transformers of the leaf types.
            registry: The transformer registry used to recursively
                transform the values of the kept entries.

        Returns:
            A new mapping of the same type as ``data``, containing
            only the entries whose key did not match the predicate, with
            their values transformed recursively.
        """
        predicate = self._key_predicate
        filtered = {
            key: registry.transform(value, func)
            for key, value in data.items()
            if not predicate(key)
        }
        return rebuild_mapping(data, filtered)
