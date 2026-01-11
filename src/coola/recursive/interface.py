r"""Define the public interface to recursively apply a function to all
items in nested data."""

from __future__ import annotations

__all__ = ["get_default_registry", "recursive_apply", "register_transformers"]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from coola.recursive.default import (
    DefaultTransformer,
)
from coola.recursive.mapping import MappingTransformer
from coola.recursive.registry import TransformerRegistry
from coola.recursive.sequence import SequenceTransformer
from coola.recursive.set import SetTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.base import BaseTransformer


def recursive_apply(
    data: Any, func: Callable[[Any], Any], registry: TransformerRegistry | None = None
) -> Any:
    """Recursively apply a function to all items in nested data.

    This is the main public interface that maintains compatibility
    with the original implementation.

    Args:
        data: Input data (can be nested)
        func: Function to apply to each leaf value
        registry: Registry to resolve transformers for nested data.

    Returns:
        Transformed data with same structure as input

    Example:
        ```pycon
        >>> from coola.recursive import recursive_apply
        >>> recursive_apply({"a": 1, "b": "abc"}, str)
        {'a': '1', 'b': 'abc'}
        >>> recursive_apply([1, [2, 3], {"x": 4}], lambda x: x * 2)
        [2, [4, 6], {'x': 8}]

        ```
    """
    if registry is None:
        registry = get_default_registry()
    return registry.transform(data, func)


def register_transformers(
    mapping: Mapping[type, BaseTransformer[Any]],
    exist_ok: bool = False,
) -> None:
    """Register custom transformers to the default global registry.

    This allows users to add support for custom types without modifying
    global state directly.

    Args:
        mapping: Dictionary mapping types to transformer instances
        exist_ok: If False, raises error if any type already registered

    Example:
        ```pycon
        >>> from coola.recursive import register_transformers, BaseTransformer
        >>> class MyType:
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        >>> class MyTransformer(BaseTransformer):
        ...     def transform(self, data, func, registry):
        ...         return MyType(func(data.value))
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
    sets, and scalar types (int, float, str, bool).

    This function uses a singleton pattern to ensure the same registry instance
    is returned on subsequent calls, which is efficient and maintains consistency
    across an application.

    Returns:
        A TransformerRegistry instance with transformers registered for:
            - Scalar types (int, float, complex, bool, str)
            - Sequences (list, tuple, Sequence ABC)
            - Sets (set, frozenset)
            - Mappings (dict, Mapping ABC)

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
    if not hasattr(get_default_registry, "_registry"):
        registry = TransformerRegistry()
        _register_default_transformers(registry)
        get_default_registry._registry = registry
    return get_default_registry._registry


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
            # Numeric types - no recursion needed
            int: default_transformer,
            float: default_transformer,
            complex: default_transformer,
            bool: default_transformer,
            # Sequences - recursive transformation preserving order
            list: sequence_transformer,
            tuple: sequence_transformer,
            Sequence: sequence_transformer,
            # Sets - recursive transformation without order
            set: set_transformer,
            frozenset: set_transformer,
            # Mappings - recursive transformation of keys and values
            dict: mapping_transformer,
            Mapping: mapping_transformer,
        }
    )
