r"""Define the public interface to recursively apply a function to all
items in nested data."""

from __future__ import annotations

__all__ = ["get_default_registry", "register_equality_testers"]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from coola.equality.tester import MappingEqualityTester, SequenceEqualityTester
from coola.equality.tester.default import DefaultEqualityTester
from coola.equality.tester.registry import EqualityTesterRegistry
from coola.equality.tester.scalar import ScalarEqualityTester

if TYPE_CHECKING:
    from coola.equality.tester.base import BaseEqualityTester


def register_equality_testers(
    mapping: Mapping[type, BaseEqualityTester[Any]],
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
        >>> from coola.equality.tester import register_equality_testers, BaseEqualityTester
        >>> class MyType:
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        >>> class MyEqualityTester(BaseEqualityTester):
        ...     def objects_are_equal(
        ...         self, actual: object, expected: object, config: EqualityConfig2
        ...     ) -> bool:
        ...         if type(other) is not type(self):
        ...             return False
        ...         return actual.value == expected.value
        ...
        >>> register_equality_testers({MyType: MyEqualityTester()})

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> EqualityTesterRegistry:
    """Get or create the default global registry with common Python
    types.

    Returns a singleton registry instance that is pre-configured with transformers
    for Python's built-in types including sequences (list, tuple), mappings (dict),
    sets, and scalar types (int, float, str, bool).

    This function uses a singleton pattern to ensure the same registry instance
    is returned on subsequent calls, which is efficient and maintains consistency
    across an application.

    Returns:
        A EqualityTesterRegistry instance with transformers registered for:
            - Scalar types (int, float, complex, bool, str)
            - Sequences (list, tuple, Sequence ABC)
            - Sets (set, frozenset)
            - Mappings (dict, Mapping ABC)

    Notes:
        The singleton pattern means modifications to the returned registry
        affect all future calls to this function. If you need an isolated
        registry, create a new EqualityTesterRegistry instance directly.

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
        registry = EqualityTesterRegistry()
        _register_default_equality_testers(registry)
        get_default_registry._registry = registry
    return get_default_registry._registry


def _register_default_equality_testers(registry: EqualityTesterRegistry) -> None:
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
    # TODO(tibo): after the other equality testers are implemented
    equality_testers = _get_native_equality_testers()
    registry.register_many(equality_testers)


def _get_native_equality_testers() -> dict[type, BaseEqualityTester]:
    r"""Get native equality testers and their associated type.

    Returns:
        A dict of native equality testers and their associated type.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    default = DefaultEqualityTester()
    scalar = ScalarEqualityTester()
    mapping = MappingEqualityTester()
    sequence = SequenceEqualityTester()
    return {
        # Object is the catch-all base for unregistered types
        object: default,
        # Strings should not be iterated character by character
        str: default,
        # Numeric types - no recursion needed
        int: scalar,
        float: scalar,
        complex: default,
        bool: scalar,
        # Sequences - recursive transformation preserving order
        list: sequence,
        tuple: sequence,
        Sequence: sequence,
        # Sets - recursive transformation without order
        set: default,
        frozenset: default,
        # Mappings - recursive transformation of keys and values
        dict: mapping,
        Mapping: mapping,
    }
