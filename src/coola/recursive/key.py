r"""Define the mapping transformer that filters entries by key."""

from __future__ import annotations

__all__ = ["KeyFilterTransformer"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from coola.recursive.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


class KeyFilterTransformer(BaseTransformer[Mapping[Any, Any]]):
    r"""Transformer for mappings that drops entries whose key matches a
    predicate and recurses into the remaining values.

    ``func`` plays a dual role: it is called directly on each key to
    decide whether the entry should be dropped, and it is also passed
    down as the transformation function applied recursively (via the
    registry) to the values of the entries that are kept. After
    filtering and transforming, the mapping is reconstructed using its
    original type.

    Notes:
        - Entries whose key satisfies ``func`` (i.e. ``func(key)`` is
          truthy) are removed from the output.
        - Keys of the entries that are kept are preserved unchanged.
        - If the value of a kept entry is itself a nested mapping that
          is dispatched to a ``KeyFilterTransformer``, keys matching
          ``func`` are removed at every level of nesting.

    Example:
        ```pycon
        >>> from coola.recursive import KeyFilterTransformer, TransformerRegistry, DefaultTransformer
        >>> registry = TransformerRegistry({object: DefaultTransformer()})
        >>> transformer = KeyFilterTransformer()
        >>> transformer.transform(
        ...     {"keep": 1, "secret": 2}, func=lambda x: x == "secret", registry=registry
        ... )
        {'keep': False}

        ```
    """

    def transform(
        self,
        data: Mapping[Any, Any],
        func: Callable[[Any], bool],
        registry: TransformerRegistry,
    ) -> Mapping[Any, Any]:
        r"""Drop entries whose key matches ``func`` and transform the
        remaining values recursively.

        Args:
            data: The mapping to filter and transform.
            func: A predicate applied to each key: entries whose key
                satisfies it are dropped. It is also passed as the
                transformation function used recursively (via
                ``registry``) on the values of the entries that are
                kept.
            registry: The transformer registry used to recursively
                transform the values of the kept entries.

        Returns:
            A new mapping of the same type as ``data``, containing
            only the entries whose key did not match ``func``, with
            their values transformed recursively.
        """
        filtered = {
            key: registry.transform(value, func) for key, value in data.items() if not func(key)
        }
        return type(data)(filtered)
