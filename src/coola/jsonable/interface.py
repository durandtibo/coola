r"""Define the public interface to convert nested data to JSON data."""

from __future__ import annotations

__all__ = ["get_default_registry", "register_transformers", "to_jsonable"]

import dataclasses
import warnings
from collections import deque
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from coola.jsonable import stdlib
from coola.jsonable.transformers import (
    DataclassTransformer,
    MappingTransformer,
    SequenceTransformer,
    SetTransformer,
)
from coola.recursive import DefaultTransformer, TransformerRegistry
from coola.registry.default import make_default_registry_singleton
from coola.validation import validate_in

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.base import Transformer
    from coola.utils.singleton import LazySingleton

_SCALARS = (str, int, float, bool, type(None))
_MODES = ("raise", "warn", "keep")


def to_jsonable(
    data: object,
    *,
    registry: TransformerRegistry | None = None,
    on_unsupported: str = "raise",
) -> Any:
    r"""Convert nested data to JSON data.

    The output contains only ``dict`` with ``str`` keys, ``list``,
    ``str``, ``int``, ``float``, ``bool`` and ``None``: the values that
    ``json.loads`` returns. Tuples, deques, sets and ranges become
    lists (a set becomes a sorted list), mappings and dataclasses become
    dictionaries. The keys are converted to strings like ``json.dumps``
    writes them. The conversion is one way and is not a round trip to
    the Python types. ``NaN`` and the infinite floats are kept.

    The options must be passed by keyword.

    Args:
        data: The data to convert.
        registry: The registry that finds the transformer of each type.
            If ``None``, the default registry is used.
        on_unsupported: What happens to a value whose type has no
            transformer. ``"raise"`` raises a ``TypeError``, ``"warn"``
            issues a ``UserWarning`` and keeps the value, ``"keep"``
            keeps the value silently. With ``"warn"`` and ``"keep"``
            the result may contain objects that ``json`` rejects.

    Returns:
        The JSON data.

    Raises:
        ValueError: If ``on_unsupported`` is not ``"raise"``,
            ``"warn"`` or ``"keep"``, if a container contains itself,
            or if two keys give the same string.
        TypeError: If a value has no transformer (with
            ``on_unsupported="raise"``) or if a key cannot be
            converted to a string.

    Example:
        ```pycon
        >>> from pathlib import PurePosixPath
        >>> from coola.jsonable import to_jsonable
        >>> to_jsonable({"path": PurePosixPath("a/b"), 1: (1, 2), "s": {3, 1, 2}})
        {'path': 'a/b', '1': [1, 2], 's': [1, 2, 3]}

        ```
    """
    validate_in(on_unsupported, _MODES, name="on_unsupported")
    if registry is None:
        registry = get_default_registry()
    return registry.transform(data, _LEAF_FUNCTIONS[on_unsupported])


def register_transformers(
    mapping: Mapping[type, Transformer[Any]],
    *,
    exist_ok: bool = False,
) -> None:
    """Register custom transformers to the default global registry.

    This function extends the singleton registry returned by
    ``get_default_registry``.

    Args:
        mapping: Mapping of Python types to transformer instances.
        exist_ok: If ``False``, raise an error when a type is already
            registered.

    Raises:
        AlreadyRegisteredError: If any type is already registered and
            ``exist_ok`` is ``False``.

    Example:
        ```pycon
        >>> from coola.jsonable import ConvertTransformer, register_transformers, to_jsonable
        >>> class Point:
        ...     def __init__(self, x, y):
        ...         self.x, self.y = x, y
        ...
        >>> register_transformers({Point: ConvertTransformer(lambda p: {"x": p.x, "y": p.y})})
        >>> to_jsonable([Point(1, 2)])
        [{'x': 1, 'y': 2}]

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> TransformerRegistry:
    """Get or create the default global registry of the JSON
    transformers.

    Returns a singleton registry that is pre-configured with the
    transformers of the built-in containers, of the standard library
    types and of the plugins of the group
    ``coola.jsonable.transformers``.

    Returns:
        A singleton ``TransformerRegistry``.

    Notes:
        The singleton pattern means modifications to the returned
        registry affect all future calls to this function. Use
        ``registry.copy()`` for an isolated registry.

    Example:
        ```pycon
        >>> from coola.jsonable import get_default_registry
        >>> get_default_registry().transform({"a": (1, 2)}, lambda x: x)
        {'a': [1, 2]}

        ```
    """
    return _default_registry.get()


def _keep(value: Any) -> Any:
    r"""Keep a value that reaches the leaf function.

    Args:
        value: The leaf.

    Returns:
        The value.
    """
    return value


def _raise(value: Any) -> Any:
    r"""Keep a JSON scalar and reject the other leaves.

    Args:
        value: The leaf.

    Returns:
        The value, if it is a JSON scalar.

    Raises:
        TypeError: If the value is not a JSON scalar.
    """
    if isinstance(value, _SCALARS):
        return value
    msg = (
        f"Cannot convert an object of type {type(value).__qualname__} to JSON: register a "
        f"ConvertTransformer for it with coola.jsonable.register_transformers, or pass "
        f"on_unsupported='keep'"
    )
    raise TypeError(msg)


def _warn(value: Any) -> Any:
    r"""Keep a leaf and warn if it is not a JSON scalar.

    Args:
        value: The leaf.

    Returns:
        The value.
    """
    if not isinstance(value, _SCALARS):
        warnings.warn(
            f"Cannot convert an object of type {type(value).__qualname__} to JSON; kept as is",
            UserWarning,
            stacklevel=2,
        )
    return value


_LEAF_FUNCTIONS: dict[str, Callable[[Any], Any]] = {"raise": _raise, "warn": _warn, "keep": _keep}


def _register_default_transformers(registry: TransformerRegistry) -> None:
    """Register the default transformers.

    Args:
        registry: The registry to populate.
    """
    leaf = DefaultTransformer()
    sequence = SequenceTransformer()
    set_ = SetTransformer()
    mapping = MappingTransformer()
    registry.register_many(
        {
            # The leaf function of ``on_unsupported`` tells a scalar from an unsupported value
            object: leaf,
            str: leaf,
            int: leaf,
            float: leaf,
            bool: leaf,
            type(None): leaf,
            list: sequence,
            tuple: sequence,
            deque: sequence,
            Sequence: sequence,
            range: sequence,
            set: set_,
            frozenset: set_,
            dict: mapping,
            Mapping: mapping,
            MappingProxyType: mapping,
        }
    )
    # ``UserString`` is registered by ``stdlib``: it is a ``Sequence`` of ``UserString``
    registry.register_many(stdlib.get_default_transformers())
    registry.register_fallback(dataclasses.is_dataclass, DataclassTransformer())


_default_registry: LazySingleton[TransformerRegistry] = make_default_registry_singleton(
    TransformerRegistry, _register_default_transformers, plugin_group="coola.jsonable.transformers"
)
