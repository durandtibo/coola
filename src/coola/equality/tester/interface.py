r"""Define the public interface for equality testers and registry
management."""

from __future__ import annotations

__all__ = [
    "get_default_equality_tester_registry",
    "get_default_registry",
    "register_equality_testers",
]

from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from typing import Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.tester.base import BaseEqualityTester
from coola.equality.tester.collection import (
    MappingEqualityTester,
    OrderedMappingEqualityTester,
    SequenceEqualityTester,
)
from coola.equality.tester.dataclass import DataclassEqualityTester
from coola.equality.tester.default import DefaultEqualityTester
from coola.equality.tester.equal import EqualEqualityTester, EqualNanEqualityTester
from coola.equality.tester.registry import DataclassMarker, EqualityTesterRegistry
from coola.equality.tester.scalar import ScalarEqualityTester
from coola.utils.imports import package_available
from coola.utils.singleton import LazySingleton, make_default_registry_singleton

# (names of the library in the module of the types, module with the testers)
_LAZY_TESTERS = (
    (("jax", "jaxlib"), "coola.equality.tester.jax"),
    (("numpy",), "coola.equality.tester.numpy"),
    (("pandas",), "coola.equality.tester.pandas"),
    (("polars",), "coola.equality.tester.polars"),
    (("pyarrow",), "coola.equality.tester.pyarrow"),
    (("torch",), "coola.equality.tester.torch"),
    (("xarray",), "coola.equality.tester.xarray"),
)


def register_equality_testers(
    mapping: Mapping[type, BaseEqualityTester[Any]],
    exist_ok: bool = False,
) -> None:
    """Register custom equality testers to the default global registry.

    This allows users to add support for custom types without modifying
    global state directly.

    Args:
        mapping: Dictionary mapping types to equality tester instances
        exist_ok: If False, raises error if any type already registered

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import register_equality_testers, BaseEqualityTester
        >>> class MyType:
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        >>> class MyEqualityTester(BaseEqualityTester[MyType]):
        ...     def equal(self, other: object) -> bool:
        ...         return type(other) is type(self)
        ...     def objects_are_equal(
        ...         self, actual: MyType, expected: object, config: EqualityConfig
        ...     ) -> bool:
        ...         if type(expected) is not type(actual):
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

    Returns a singleton registry instance that is pre-configured with equality testers
    for Python's built-in types including sequences (list, tuple), mappings (dict),
    and scalar types (int, float).

    This function uses a singleton pattern to ensure the same registry instance
    is returned on subsequent calls, which is efficient and maintains consistency
    across an application.

    Returns:
        A EqualityTesterRegistry instance with equality testers registered for:
            - Scalar types (int, float)
            - Sequences (list, tuple, deque, Sequence ABC)
            - Mappings (dict, OrderedDict, Mapping ABC)
            - Third-party library types (numpy, torch, pandas, polars, jax, xarray, pyarrow)

    Notes:
        The singleton pattern means modifications to the returned registry
        affect all future calls to this function. If you need an isolated
        registry, create a new EqualityTesterRegistry instance directly.

    Example:
        ```pycon
        >>> from coola.equality.tester import get_default_registry
        >>> from coola.equality.config import EqualityConfig
        >>> registry = get_default_registry()
        >>> # Registry is ready to use with common Python types
        >>> config = EqualityConfig()
        >>> registry.objects_are_equal([1, 2, 3], [1, 2, 3], config)
        True
        >>> registry.objects_are_equal([1, 2, 3], [1, 1], config)
        False

        ```
    """
    return _default_registry.get()


get_default_equality_tester_registry = get_default_registry


def _register_default_equality_testers(registry: EqualityTesterRegistry) -> None:
    """Register default equality testers for common Python types.

    This internal function sets up the standard type-to-equality tester mappings
    used by the default registry. It registers equality testers in a specific order
    to ensure proper inheritance handling via MRO.

    The registration strategy:
        - Scalar types use ScalarEqualityTester (supports NaN equality)
        - Sequences use SequenceEqualityTester for recursive list/tuple processing
        - Mappings use MappingEqualityTester for recursive dict processing, and
          OrderedDict uses OrderedMappingEqualityTester, which also checks the key order
        - Object type uses DefaultEqualityTester as catch-all for unregistered types

    Args:
        registry: The registry to populate with default equality testers

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    registry.register_many(_get_native_equality_testers() | _get_io_equality_testers())
    # The testers of the optional libraries are loaded the first time the registry needs
    # a type of the library, so the libraries are not imported with ``coola.equality``.
    for names, module in _LAZY_TESTERS:
        if package_available(names[0]):
            registry.register_lazy(names, f"{module}:get_default_testers")


def _get_native_equality_testers() -> dict[type, BaseEqualityTester[Any]]:
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
    ordered_mapping = OrderedMappingEqualityTester()
    sequence = SequenceEqualityTester()
    equal = EqualEqualityTester()
    return {
        # Object is the catch-all base for unregistered types
        object: default,
        # Numeric types - no recursion needed
        int: scalar,
        float: scalar,
        complex: scalar,
        # Dataclasses without their own tester
        DataclassMarker: DataclassEqualityTester(),
        # Sequences - recursive transformation preserving order
        deque: sequence,
        list: sequence,
        tuple: sequence,
        Sequence: sequence,
        # Mappings - recursive transformation of keys and values
        dict: mapping,
        OrderedDict: ordered_mapping,
        Mapping: mapping,
        # coola types
        BaseEqualityHandler: equal,
        BaseEqualityTester: equal,
    }


def _get_io_equality_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the equality testers for the loaders and savers of
    ``coola.io``.

    The loaders and the savers have an ``equal`` method, so they are
    compared with it.

    Returns:
        The equality testers for the base classes of the loaders and
            the savers.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    # Local import: ``coola.io`` depends on ``coola.equality``, so it cannot be imported
    # when this module is imported. It is imported when the registry is created.
    from coola.io.base import BaseLoader, BaseSaver  # noqa: PLC0415

    tester = EqualNanEqualityTester()
    return {BaseLoader: tester, BaseSaver: tester}


_default_registry: LazySingleton[EqualityTesterRegistry] = make_default_registry_singleton(
    EqualityTesterRegistry,
    _register_default_equality_testers,
    plugin_group="coola.equality.testers",
)
