r"""Define the public interface to recursively apply a function to all
items in nested data."""

from __future__ import annotations

__all__ = [
    "get_default_registry",
    "get_default_transformer_registry",
    "recursive_apply",
    "recursive_zip",
    "recursive_zip_apply",
    "register_transformers",
]

import dataclasses
from collections import UserString, deque
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from coola.recursive.dataclass import DataclassTransformer
from coola.recursive.default import (
    DefaultTransformer,
)
from coola.recursive.engine import transform_zip
from coola.recursive.mapping import MappingTransformer
from coola.recursive.registry import TransformerRegistry
from coola.recursive.sequence import SequenceTransformer
from coola.recursive.set import SetTransformer
from coola.registry.default import make_default_registry_singleton
from coola.utils.imports import package_available

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.base import Transformer
    from coola.utils.singleton import LazySingleton


def recursive_apply(
    data: object, func: Callable[[Any], Any], *, registry: TransformerRegistry | None = None
) -> Any:
    """Recursively transform nested data with a type-aware registry.

    Args:
        data: Input data, potentially containing nested containers.
        func: Function applied to leaf values.
        registry: Registry to resolve transformers for nested data.
            If ``None``, the default registry is used.

    Returns:
        Transformed data with the same structure as the input.

    Notes:
        Mapping keys are preserved. By default, values of ``str`` are treated
        as leaves and are not iterated character by character.

        The function is recursive: each nesting level uses a few Python frames, so a very
        deep structure (about a thousand levels) reaches the interpreter recursion limit and
        raises a ``RecursionError`` that explains how to fix it. A container that contains
        itself raises a ``ValueError``.

    Example:
        ```pycon
        >>> from coola.recursive import recursive_apply
        >>> recursive_apply({"a": 1, "b": "abc"}, str)
        {'a': '1', 'b': 'abc'}
        >>> # Guard the operation so non-numeric leaves are preserved.
        >>> recursive_apply([1, [2, 3], {"x": 4}], lambda x: x * 2 if isinstance(x, int) else x)
        [2, [4, 6], {'x': 8}]

        ```
    """
    if registry is None:
        registry = get_default_registry()
    return registry.transform(data, func)


def recursive_zip_apply(
    items: Sequence[object],
    func: Callable[[list[Any]], Any],
    *,
    registry: TransformerRegistry | None = None,
) -> Any:
    """Recursively apply a function to the leaves found at the same
    position in several nested structures.

    The structures are traversed in parallel: the function receives
    the list of the values found at the same position in every structure.
    It is the "N to 1" counterpart of ``recursive_apply``.

    Args:
        items: The structures to combine. They must have the same layout.
        func: Function applied to the list of leaves found at the same
            position in every item.
        registry: Registry to resolve transformers. If ``None``, the
            default registry is used.

    Returns:
        A structure with the layout (and the container types) of the
        first item, where each leaf is the output of ``func``.

    Raises:
        ValueError: If ``items`` is empty, if the items do not have the
            same keys, lengths or container kinds, or if a container
            contains itself.
        TypeError: If the items hold sets at the same position (their
            elements have no position to align).

    Notes:
        Container types and keys follow the first item. A ``list`` and a
        ``tuple`` at the same position are accepted. Mappings must have
        the same keys, but the order of the keys of the other items is
        ignored. ``str`` is a leaf. A custom transformer registered with
        ``register_transformers`` is used through its ``expand`` method:
        the containers of the same position must have the same number of
        children.

    Example:
        ```pycon
        >>> from coola.recursive import recursive_zip_apply
        >>> recursive_zip_apply([{"a": 1, "b": [2, 3]}, {"a": 10, "b": [20, 30]}], sum)
        {'a': 11, 'b': [22, 33]}

        ```
    """
    if registry is None:
        registry = get_default_registry()
    return transform_zip(items, func, registry)


def recursive_zip(items: Sequence[object], *, registry: TransformerRegistry | None = None) -> Any:
    """Recursively zip several nested structures with the same layout.

    The result has the layout of the first item, and each leaf is the
    tuple of the leaves found at the same position in every item. It is
    ``recursive_zip_apply`` with ``tuple`` as the function.

    Args:
        items: The structures to zip. They must have the same layout.
        registry: Registry to resolve transformers. If ``None``, the
            default registry is used.

    Returns:
        A structure with the layout (and the container types) of the
        first item, where each leaf is a ``tuple`` of the leaves of the
        items.

    Raises:
        ValueError: If ``items`` is empty, if the items do not have the
            same keys, lengths or container kinds, or if a container
            contains itself.
        TypeError: If the items hold sets at the same position.

    Example:
        ```pycon
        >>> from coola.recursive import recursive_zip
        >>> recursive_zip([{"a": 1, "b": [2, 3]}, {"a": 10, "b": [20, 30]}])
        {'a': (1, 10), 'b': [(2, 20), (3, 30)]}

        ```
    """
    return recursive_zip_apply(items, tuple, registry=registry)


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
        exist_ok: If ``False`` (default), raise an error when a type is
            already registered. If ``True``, overwrites existing
            registrations silently.

    Raises:
        AlreadyRegisteredError: If any type is already registered and ``exist_ok``
            is ``False``.

    Example:
        ```pycon
        >>> from coola.recursive import register_transformers, BaseTransformer
        >>> class MyType:
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        >>> class MyTransformer(BaseTransformer):
        ...     def expand(self, data):
        ...         return [data.value], lambda items: MyType(items[0])
        ...
        >>> register_transformers({MyType: MyTransformer()})

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> TransformerRegistry:
    """Get or create the default global registry with common Python
    types.

    Returns a singleton registry instance that is pre-configured with transformers
    for Python's built-in types including sequences (list, tuple), mappings (dict),
    sets, dataclasses, and scalar types (int, float, str, bool).

    This function uses a singleton pattern to ensure the same registry instance
    is returned on subsequent calls, which is efficient and maintains consistency
    across an application.

    Returns:
        A singleton ``TransformerRegistry`` configured for common built-in
        scalar and container types.

    Notes:
        The singleton pattern means modifications to the returned registry
        affect all future calls to this function. If you need an isolated
        registry, create a new TransformerRegistry instance directly.

    Example:
        ```pycon
        >>> from coola.recursive import get_default_registry
        >>> registry = get_default_registry()
        >>> # Registry is ready to use with common Python types
        >>> registry.transform([1, 2, 3], str)
        ['1', '2', '3']
        >>> registry.transform({"a": 1, "b": 2}, lambda x: x * 10)
        {'a': 10, 'b': 20}

        ```
    """
    return _default_registry.get()


get_default_transformer_registry = get_default_registry


def _register_default_transformers(registry: TransformerRegistry) -> None:
    """Register default transformers for common Python types.

    This internal function sets up the standard type-to-transformer mappings
    used by the default registry. It registers transformers in a specific order
    to ensure proper inheritance handling via MRO.

    The registration strategy:
        - Scalar types use DefaultTransformer (no recursion)
        - Strings use DefaultTransformer to prevent character-by-character iteration
        - Sequences use SequenceTransformer for recursive list/tuple processing
        - Sets use SetTransformer for recursive set processing
        - Mappings use MappingTransformer for recursive dict processing
        - Dataclasses use DataclassTransformer (the fields are transformed)
        - Pydantic models use PydanticModelTransformer (the fields are transformed)

    Args:
        registry: The registry to populate with default transformers

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    default_transformer = DefaultTransformer()
    sequence_transformer = SequenceTransformer()
    set_transformer = SetTransformer()
    mapping_transformer = MappingTransformer()

    registry.register_many(
        {
            # Object is the catch-all base for unregistered types
            object: default_transformer,
            # Strings should not be iterated character by character
            str: default_transformer,
            # UserString is a Sequence of UserString: expanding it never ends
            UserString: default_transformer,
            # Numeric types - no recursion needed
            int: default_transformer,
            float: default_transformer,
            complex: default_transformer,
            bool: default_transformer,
            # Sequences - recursive transformation preserving order
            list: sequence_transformer,
            tuple: sequence_transformer,
            deque: sequence_transformer,
            Sequence: sequence_transformer,
            # Sets - recursive transformation without order
            set: set_transformer,
            frozenset: set_transformer,
            # Mappings - recursive transformation of the values
            dict: mapping_transformer,
            Mapping: mapping_transformer,
            MappingProxyType: mapping_transformer,
        }
    )
    # Dataclasses without their own transformer - the fields are transformed
    registry.register_fallback(dataclasses.is_dataclass, DataclassTransformer())
    # The pydantic models are entered like the dataclasses. The transformer is loaded the first
    # time the registry needs a type of pydantic, so pydantic is not imported with ``coola.recursive``.
    if package_available("pydantic"):
        registry.register_lazy("pydantic", "coola.recursive.pydantic:get_default_transformers")


_default_registry: LazySingleton[TransformerRegistry] = make_default_registry_singleton(
    TransformerRegistry, _register_default_transformers, plugin_group="coola.recursive.transformers"
)
