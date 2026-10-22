r"""Define the mapping transformer that filters entries by key."""

from __future__ import annotations

__all__ = ["KeyFilterTransformer"]

from collections.abc import Mapping, Sequence, Set as AbstractSet
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

    A ``TransformerRegistry`` threads a single function through the
    recursion, so a transformer that needs two functions (a predicate
    for the keys and a function for the values) must keep one of them.
    This transformer supports two modes:

    - **With** ``key_predicate`` (recommended): the predicate is stored in
      the transformer and used for the keys of every mapping it
      transforms, however deep they are nested. The ``func`` given to
      ``transform`` is the function for the values: it is forwarded
      unchanged to the registry and applied by the transformers of the
      leaf types.
    - **Without** ``key_predicate`` (legacy): ``func`` is the key predicate,
      and it is forwarded to the nested containers so they are
      filtered with the same predicate. The values that are not
      containers are transformed with ``transform_func`` (the
      identity by default), because ``func`` is a predicate and
      applying it to the values would replace them with booleans.
      ``transform_func`` only applies to the entries of the mapping
      itself, not to the mappings nested in it, since a single function
      is threaded through the recursion: use ``key_predicate`` if the
      leaf transformers of the registry apply the function.

    Args:
        key_predicate: A function applied to each key: the entries whose
            key satisfies it are dropped. If ``None`` (default), the
            legacy mode is used.
        transform_func: Only for the legacy mode. The function applied
            (via the registry) to the values of the kept entries that
            are not containers. If ``None`` (default), the values are
            left unchanged.

    Raises:
        ValueError: If ``key_predicate`` and ``transform_func`` are both given.

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

    def __init__(
        self,
        key_predicate: Callable[[Any], bool] | None = None,
        transform_func: Callable[[Any], Any] | None = None,
    ) -> None:
        if key_predicate is not None and transform_func is not None:
            msg = (
                "`transform_func` is only used without `key_predicate`: with a "
                "`key_predicate`, the function of `transform` is applied to the values"
            )
            raise ValueError(msg)
        self._key_predicate = key_predicate
        self._transform_func = transform_func

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"key_predicate": self._key_predicate, "transform_func": self._transform_func}

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
            func: With a ``key_predicate``, the function applied to the
                values by the transformers of the leaf types. Without
                a ``key_predicate``, the key predicate: entries whose
                key satisfies it are dropped, and it is forwarded
                recursively (via ``registry``) so the nested containers
                are filtered with the same predicate.
            registry: The transformer registry used to recursively
                transform the values of the kept entries.

        Returns:
            A new mapping of the same type as ``data``, containing
            only the entries whose key did not match the predicate, with
            their values transformed recursively.
        """
        if self._key_predicate is not None:
            predicate = self._key_predicate
            filtered = {
                key: registry.transform(value, func)
                for key, value in data.items()
                if not predicate(key)
            }
            return rebuild_mapping(data, filtered)
        transform_func = _identity if self._transform_func is None else self._transform_func
        filtered = {
            key: registry.transform(value, self._value_func(value, func, transform_func))
            for key, value in data.items()
            if not func(key)
        }
        return rebuild_mapping(data, filtered)

    @staticmethod
    def _value_func(
        value: Any, func: Callable[[Any], bool], transform_func: Callable[[Any], Any]
    ) -> Callable[[Any], Any]:
        r"""Select the function to propagate to ``value`` (legacy mode).

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


def _identity(value: Any) -> Any:
    return value
