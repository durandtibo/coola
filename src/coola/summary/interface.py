r"""Define the public interface to summarize nested data."""

from __future__ import annotations

__all__ = [
    "get_default_registry",
    "get_default_summarizer_registry",
    "register_summarizers",
    "summarize",
]

from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from coola.summary.default import DefaultSummarizer
from coola.summary.jax import JaxArraySummarizer
from coola.summary.mapping import MappingSummarizer
from coola.summary.numpy import NDArraySummarizer
from coola.summary.pandas import PandasDataFrameSummarizer, PandasSeriesSummarizer
from coola.summary.polars import (
    PolarsDataFrameSummarizer,
    PolarsLazyFrameSummarizer,
    PolarsSeriesSummarizer,
)
from coola.summary.pyarrow import PyarrowArraySummarizer, PyarrowTableSummarizer
from coola.summary.registry import SummarizerRegistry
from coola.summary.sequence import SequenceSummarizer
from coola.summary.set import SetSummarizer
from coola.summary.torch import TensorSummarizer
from coola.summary.xarray import (
    XarrayDataArraySummarizer,
    XarrayDatasetSummarizer,
    XarrayVariableSummarizer,
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
from coola.utils.singleton import LazySingleton, make_default_registry_singleton
from coola.validation import validate_non_negative

if TYPE_CHECKING:
    from coola.summary.base import Summarizer

if is_jax_available():  # pragma: no cover
    import jax.numpy as jnp
else:  # pragma: no cover
    from coola.utils.fallback.jax import jnp

if is_pandas_available():  # pragma: no cover
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd

if is_polars_available():  # pragma: no cover
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl

if is_pyarrow_available():  # pragma: no cover
    import pyarrow as pa
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa

if is_xarray_available():  # pragma: no cover
    import xarray as xr
else:  # pragma: no cover
    from coola.utils.fallback.xarray import xarray as xr

if is_torch_available():  # pragma: no cover
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch

if is_numpy_available():  # pragma: no cover
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np


def summarize(data: object, max_depth: int = 1, registry: SummarizerRegistry | None = None) -> str:
    r"""Create a summary string representation of nested data.

    Args:
        data: Input data to summarize.
        max_depth: The maximum nesting level to expand when summarizing.
            Structures deeper than this level are shown in compact form.
            Must be non-negative. Default is 1, which expands only the
            top level of nested structures.
        registry: Registry to resolve summarizers for nested data.
            If ``None``, the default registry is used.

    Returns:
        A formatted string summary of ``data``.

    Raises:
        ValueError: If ``max_depth`` is negative.

    Example:
        ```pycon
        >>> from coola.summary import summarize
        >>> print(summarize({"a": 1, "b": "abc"}))
        <class 'dict'> (length=2)
          (a): 1
          (b): abc

        ```
    """
    validate_non_negative(max_depth, name="max_depth")
    if registry is None:
        registry = get_default_registry()
    return registry.summarize(data, max_depth=max_depth)


def register_summarizers(
    mapping: Mapping[type, Summarizer[Any]],
    exist_ok: bool = False,
) -> None:
    """Register custom summarizers to the default global registry.

    This function extends the singleton registry returned by
    ``get_default_registry``.

    Args:
        mapping: Mapping of Python types to summarizer instances.
        exist_ok: If ``False``, raise an error when a type is already
            registered.

    Example:
        ```pycon
        >>> from coola.summary import register_summarizers, BaseSummarizer, SummarizerRegistry
        >>> class MyType:
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        >>> class MySummarizer(BaseSummarizer[MyType]):
        ...     def equal(self, other: object) -> bool:
        ...         return type(other) is type(self)
        ...     def summarize(
        ...         self,
        ...         data: MyType,
        ...         registry: SummarizerRegistry,
        ...         depth: int = 0,
        ...         max_depth: int = 1,
        ...     ) -> str:
        ...         return f"<MyType> value={data.value}"
        ...
        >>> register_summarizers({MyType: MySummarizer()})

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> SummarizerRegistry:
    """Get or create the default global registry with common Python
    types.

    Returns a singleton registry instance that is pre-configured with summarizers
    for Python's built-in types including sequences (list, tuple), mappings (dict),
    sets, and scalar types (int, float, str, bool).

    This function uses a singleton pattern to ensure the same registry instance
    is returned on subsequent calls, which is efficient and maintains consistency
    across an application.

    Returns:
        A singleton ``SummarizerRegistry`` configured for common built-in
        scalar and container types.

    Notes:
        The singleton pattern means modifications to the returned registry
        affect all future calls to this function. If you need an isolated
        registry, create a new SummarizerRegistry instance directly.

    Example:
        ```pycon
        >>> from coola.summary import get_default_registry
        >>> registry = get_default_registry()
        >>> # Registry is ready to use with common Python types
        >>> print(registry.summarize([1, 2, 3]))
        <class 'list'> (length=3)
          (0): 1
          (1): 2
          (2): 3
        >>> print(registry.summarize({"a": 1, "b": 2}))
        <class 'dict'> (length=2)
          (a): 1
          (b): 2

        ```
    """
    return _default_registry.get()


get_default_summarizer_registry = get_default_registry


def _register_default_summarizers(registry: SummarizerRegistry) -> None:
    """Register default summarizers for common Python types.

    This internal function sets up the standard type-to-summarizer mappings
    used by the default registry. It registers summarizers in a specific order
    to ensure proper inheritance handling via MRO.

    The registration strategy:
        - Scalar types use DefaultSummarizer (no recursion)
        - Strings use DefaultSummarizer to prevent character-by-character iteration
        - Sequences use SequenceSummarizer for summary list/tuple processing
        - Sets use SetSummarizer for summary set processing
        - Mappings use MappingSummarizer for summary dict processing

    Args:
        registry: The registry to populate with default summarizers

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    summarizers: dict[type, Summarizer[Any]] = _get_native_summarizers()
    summarizers.update(_get_jax_summarizers())
    summarizers.update(_get_numpy_summarizers())
    summarizers.update(_get_pandas_summarizers())
    summarizers.update(_get_polars_summarizers())
    summarizers.update(_get_pyarrow_summarizers())
    summarizers.update(_get_torch_summarizers())
    summarizers.update(_get_xarray_summarizers())
    registry.register_many(summarizers)


def _get_native_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the native summarizers for common Python types.

    Returns:
        The native summarizers for common Python types.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    default_summarizer = DefaultSummarizer()
    sequence_summarizer = SequenceSummarizer()
    set_summarizer = SetSummarizer()
    mapping_summarizer = MappingSummarizer()
    return {
        # Object is the catch-all base for unregistered types
        object: default_summarizer,
        # Strings should not be iterated character by character
        str: default_summarizer,
        # Numeric types - no recursion needed
        int: default_summarizer,
        float: default_summarizer,
        complex: default_summarizer,
        bool: default_summarizer,
        # Sequences - summary transformation preserving order
        list: sequence_summarizer,
        tuple: sequence_summarizer,
        deque: sequence_summarizer,
        Sequence: sequence_summarizer,
        # Sets - summary transformation without order
        set: set_summarizer,
        frozenset: set_summarizer,
        # Mappings - summary transformation of keys and values
        dict: mapping_summarizer,
        Mapping: mapping_summarizer,
    }


def _get_numpy_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the native summarizers for NumPy types.

    Returns:
        The summarizers for NumPy array types.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_numpy_available():  # pragma: no cover
        return {}
    return {np.ndarray: NDArraySummarizer()}


def _get_torch_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the native summarizers for PyTorch types.

    Returns:
        The summarizers for PyTorch tensor types.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_torch_available():  # pragma: no cover
        return {}
    return {torch.Tensor: TensorSummarizer()}


def _get_jax_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the summarizers for JAX types.

    Returns:
        The summarizers for JAX types, or an empty dictionary if the
            package is not installed.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_jax_available():  # pragma: no cover
        return {}
    summarizer = JaxArraySummarizer()
    return {jnp.ndarray: summarizer, type(jnp.ones(1)): summarizer}


def _get_pandas_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the summarizers for pandas types.

    Returns:
        The summarizers for pandas types, or an empty dictionary if the
            package is not installed.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_pandas_available():  # pragma: no cover
        return {}
    return {pd.DataFrame: PandasDataFrameSummarizer(), pd.Series: PandasSeriesSummarizer()}


def _get_polars_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the summarizers for polars types.

    Returns:
        The summarizers for polars types, or an empty dictionary if the
            package is not installed.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_polars_available():  # pragma: no cover
        return {}
    return {
        pl.DataFrame: PolarsDataFrameSummarizer(),
        pl.LazyFrame: PolarsLazyFrameSummarizer(),
        pl.Series: PolarsSeriesSummarizer(),
    }


def _get_pyarrow_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the summarizers for pyarrow types.

    Returns:
        The summarizers for pyarrow types, or an empty dictionary if the
            package is not installed.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_pyarrow_available():  # pragma: no cover
        return {}
    array_summarizer = PyarrowArraySummarizer()
    return {
        pa.Array: array_summarizer,
        pa.ChunkedArray: array_summarizer,
        pa.Table: PyarrowTableSummarizer(),
    }


def _get_xarray_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the summarizers for xarray types.

    Returns:
        The summarizers for xarray types, or an empty dictionary if the
            package is not installed.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    if not is_xarray_available():  # pragma: no cover
        return {}
    return {
        xr.DataArray: XarrayDataArraySummarizer(),
        xr.Dataset: XarrayDatasetSummarizer(),
        xr.Variable: XarrayVariableSummarizer(),
    }


_default_registry: LazySingleton[SummarizerRegistry] = make_default_registry_singleton(
    SummarizerRegistry, _register_default_summarizers, plugin_group="coola.summary.summarizers"
)
