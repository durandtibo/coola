r"""Define the mapping transformer that filters entries by key."""

from __future__ import annotations

__all__ = ["KeyFilterTransformer"]

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Any

from coola.recursive.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


class KeyFilterTransformer(BaseTransformer[Mapping[Any, Any]]):
    r"""Transformer for mappings that drops entries whose key matches a
    predicate and recurses into the remaining values.

    ``func`` is the predicate applied to each key to decide whether the
    entry should be dropped. It is unrelated to how the values of the
    kept entries are transformed: that is controlled by the separate
    ``transform_func`` keyword argument, which defaults to ``func``
    itself only for backward compatibility with registries (such as the
    one built by ``coola.nested.remove_keys_if``) where every non-
    mapping/sequence/set type is mapped to an identity-like transformer
    that ignores the function it receives. Pass an explicit
    ``transform_func`` (e.g. the identity function) to decouple the two
    roles when the registry contains transformers that actually apply
    the function to leaf values.

    Notes:
        - Entries whose key satisfies ``func`` (i.e. ``func(key)`` is
          truthy) are removed from the output.
        - Keys of the entries that are kept are preserved unchanged.
        - If the value of a kept entry is itself a nested mapping,
          sequence, or set, ``func`` (not ``transform_func``) is
          forwarded to it, so nested mappings keep being filtered with
          the same predicate however deep they are nested. This only
          holds when ``transform_func`` is left at its default: because
          ``TransformerRegistry.transform`` threads a single function
          through the recursion, an explicit ``transform_func`` only
          decouples value transformation from the predicate for the
          entries of ``data`` itself, not for mappings nested inside
          them.

    Example:
        ```pycon
        >>> from coola.recursive import (
        ...     KeyFilterTransformer,
        ...     TransformerRegistry,
        ...     DefaultTransformer,
        ... )
        >>> registry = TransformerRegistry({object: DefaultTransformer()})
        >>> transformer = KeyFilterTransformer()
        >>> # Without an explicit transform_func, values are left unchanged
        >>> # by default only for identity-like leaf transformers; here
        >>> # DefaultTransformer actually applies the function it is given,
        >>> # so pass transform_func explicitly to keep values untouched.
        >>> transformer.transform(
        ...     {"keep": 1, "secret": 2},
        ...     func=lambda x: x == "secret",
        ...     registry=registry,
        ...     transform_func=lambda x: x,
        ... )
        {'keep': 1}

        ```
    """

    def transform(
        self,
        data: Mapping[Any, Any],
        func: Callable[[Any], bool],
        registry: TransformerRegistry,
        *,
        transform_func: Callable[[Any], Any] | None = None,
    ) -> Mapping[Any, Any]:
        r"""Drop entries whose key matches ``func`` and transform the
        remaining values recursively.

        Args:
            data: The mapping to filter and transform.
            func: A predicate applied to each key: entries whose key
                satisfies it are dropped. It is also forwarded
                recursively (via ``registry``) so nested mappings keep
                being filtered with the same predicate.
            registry: The transformer registry used to recursively
                transform the values of the kept entries.
            transform_func: The function applied recursively (via
                ``registry``) to the values of the kept entries.
                Defaults to ``func`` for backward compatibility; pass
                an explicit function (e.g. the identity function) to
                decouple value transformation from the key predicate.

        Returns:
            A new mapping of the same type as ``data``, containing
            only the entries whose key did not match ``func``, with
            their values transformed recursively.
        """
        transform_func = func if transform_func is None else transform_func
        filtered = {
            key: registry.transform(value, self._value_func(value, func, transform_func))
            for key, value in data.items()
            if not func(key)
        }
        return type(data)(filtered)

    @staticmethod
    def _value_func(
        value: Any, func: Callable[[Any], bool], transform_func: Callable[[Any], Any]
    ) -> Callable[[Any], Any]:
        r"""Select the function to propagate to ``value``.

        ``func`` (the key predicate) is forwarded for containers
        (mapping, sequence, set, excluding ``str``) so nested mappings
        keep being filtered with the same predicate however deep they
        are nested. ``transform_func`` is forwarded for every other
        value, so it is the one applied when the value is eventually
        handled by a leaf transformer.
        """
        if isinstance(value, str):
            return transform_func
        if isinstance(value, (Mapping, Sequence, AbstractSet)):
            return func
        return transform_func
