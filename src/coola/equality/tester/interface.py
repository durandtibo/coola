r"""Define the public interface to recursively apply a function to all
items in nested data."""

from __future__ import annotations

__all__ = ["get_default_registry", "register_equality_testers"]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from coola.equality.tester.pyarrow import PyarrowEqualityTester
from coola.equality.tester.collection import (
    MappingEqualityTester,
    SequenceEqualityTester,
)
from coola.equality.tester.default import DefaultEqualityTester
from coola.equality.tester.jax import JaxArrayEqualityTester, get_array_impl_class
from coola.equality.tester.numpy import (
    NumpyArrayEqualityTester,
    NumpyMaskedArrayEqualityTester,
)
from coola.equality.tester.pandas import (
    PandasDataFrameEqualityTester,
    PandasSeriesEqualityTester,
)
from coola.equality.tester.polars import (
    PolarsDataFrameEqualityTester,
    PolarsLazyFrameEqualityTester,
    PolarsSeriesEqualityTester,
)
from coola.equality.tester.registry import EqualityTesterRegistry
from coola.equality.tester.scalar import ScalarEqualityTester
from coola.equality.tester.torch import (
    TorchPackedSequenceEqualityTester,
    TorchTensorEqualityTester,
)
from coola.equality.tester.xarray import (
    XarrayDataArrayEqualityTester,
    XarrayDatasetEqualityTester,
    XarrayVariableEqualityTester,
)
from coola.utils.imports import (
    is_jax_available,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_torch_available,
    is_xarray_available,
)

if TYPE_CHECKING:
    from coola.equality.tester.base import BaseEqualityTester

if is_jax_available():
    import jax.numpy as jnp
if is_numpy_available():
    import numpy as np
if is_pandas_available():
    import pandas as pd
if is_polars_available():
    import polars as pl
if is_pyarrow_available():
    import pyarrow as pa
if is_torch_available():
    import torch
if is_xarray_available():
    import xarray as xr


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
        >>> from coola.equality.tester import get_default_registry
        >>> from coola.equality.config import EqualityConfig2
        >>> registry = get_default_registry()
        >>> # Registry is ready to use with common Python types
        >>> config = EqualityConfig2()
        >>> registry.objects_are_equal([1, 2, 3], [1, 2, 3], config)
        True
        >>> registry.objects_are_equal([1, 2, 3], [1, 1], config)
        False

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
    testers = (
        _get_native_equality_testers()
        | _get_jax_equality_testers()
        | _get_numpy_equality_testers()
        | _get_pandas_equality_testers()
        | _get_polars_equality_testers()
        | _get_pyarrow_equality_testers()
        | _get_torch_equality_testers()
        | _get_xarray_equality_testers()
    )
    registry.register_many(testers)


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


def _get_jax_equality_testers() -> dict[type, BaseEqualityTester]:
    r"""Get the equality testers for jax objects.

    Returns:
        A dict of equality testers for jax objects.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_jax_available():  # pragma: no cover
        return {}

    tester = JaxArrayEqualityTester()
    return {jnp.ndarray: tester, get_array_impl_class(): tester}


def _get_numpy_equality_testers() -> dict[type, BaseEqualityTester]:
    r"""Get the equality testers for NumPy objects.

    Returns:
        A dict of equality testers for NumPy objects.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_numpy_available():  # pragma: no cover
        return {}

    return {
        np.ndarray: NumpyArrayEqualityTester(),
        np.ma.MaskedArray: NumpyMaskedArrayEqualityTester(),
    }


def _get_pandas_equality_testers() -> dict[type, BaseEqualityTester]:
    r"""Get the equality testers for pandas objects.

    Returns:
        A dict of equality testers for pandas objects.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_pandas_available():  # pragma: no cover
        return {}

    return {
        pd.DataFrame: PandasDataFrameEqualityTester(),
        pd.Series: PandasSeriesEqualityTester(),
    }


def _get_polars_equality_testers() -> dict[type, BaseEqualityTester]:
    r"""Get the equality testers for polars objects.

    Returns:
        A dict of equality testers for polars objects.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_polars_available():  # pragma: no cover
        return {}

    return {
        pl.DataFrame: PolarsDataFrameEqualityTester(),
        pl.LazyFrame: PolarsLazyFrameEqualityTester(),
        pl.Series: PolarsSeriesEqualityTester(),
    }


def _get_pyarrow_equality_testers() -> dict[type, BaseEqualityTester]:
    r"""Get the equality testers for pyarrow objects.

    Returns:
        A dict of equality testers for pyarrow objects.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_pyarrow_available():  # pragma: no cover
        return {}

    tester = PyarrowEqualityTester()
    return {pa.Array: tester, pa.Table: tester}


def _get_torch_equality_testers() -> dict[type, BaseEqualityTester]:
    r"""Get the equality testers for PyTorch objects.

    Returns:
        A dict of equality testers for PyTorch objects.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_torch_available():  # pragma: no cover
        return {}

    return {
        torch.nn.utils.rnn.PackedSequence: TorchPackedSequenceEqualityTester(),
        torch.Tensor: TorchTensorEqualityTester(),
    }


def _get_xarray_equality_testers() -> dict[type, BaseEqualityTester]:
    r"""Get the equality testers for PyTorch objects.

    Returns:
        A dict of equality testers for PyTorch objects.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_xarray_available():  # pragma: no cover
        return {}

    return {
        xr.DataArray: XarrayDataArrayEqualityTester(),
        xr.Dataset: XarrayDatasetEqualityTester(),
        xr.Variable: XarrayVariableEqualityTester(),
    }
