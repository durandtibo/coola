r"""Define some pytest fixtures for testing.

`pytest` is required to use these fixtures.

Each optional dependency has a pair of marks: ``<name>_available`` skips
the test unless the dependency is available, and
``<name>_not_available`` skips it if the dependency is available. They
are created by ``requires``, so supporting a new dependency takes one
line, plus the two names in ``__all__``.

Most marks are named after one package. The ``torch_cuda_*``,
``torch_mps_*`` and ``torch_numpy_*`` marks are named after a feature of
PyTorch or a pair of packages: they check that PyTorch is installed
*and* that the CUDA device, the MPS device or NumPy (in a compatible
version) is usable.
"""

from __future__ import annotations

__all__ = [
    "colorlog_available",
    "colorlog_not_available",
    "jax_available",
    "jax_not_available",
    "numpy_available",
    "numpy_not_available",
    "packaging_available",
    "packaging_not_available",
    "pandas_available",
    "pandas_not_available",
    "polars_available",
    "polars_not_available",
    "pyarrow_available",
    "pyarrow_not_available",
    "pydantic_available",
    "pydantic_not_available",
    "requires",
    "rich_available",
    "rich_not_available",
    "torch_available",
    "torch_cuda_available",
    "torch_cuda_not_available",
    "torch_mps_available",
    "torch_mps_not_available",
    "torch_not_available",
    "torch_numpy_available",
    "torch_numpy_not_available",
    "xarray_available",
    "xarray_not_available",
]

import pytest

from coola.utils.imports import (
    is_colorlog_available,
    is_jax_available,
    is_numpy_available,
    is_packaging_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_pydantic_available,
    is_rich_available,
    is_torch_available,
    is_torch_numpy_available,
    is_xarray_available,
)
from coola.utils.tensor import is_cuda_available, is_mps_available


def requires(
    is_available: bool, name: str, *, plural: bool = False
) -> tuple[pytest.MarkDecorator, pytest.MarkDecorator]:
    r"""Create the pair of marks for a dependency.

    Use it to support a new optional dependency: the marks are meant to
    be assigned to module-level names, so they can be imported and used
    as test decorators.

    Args:
        is_available: Whether the dependency is available.
        name: The name of the dependency in the skip reasons, e.g.
            ``"PyTorch and CUDA"``.
        plural: If ``True``, the reason of the second mark uses "are"
            instead of "is" (for a name that lists several
            dependencies).

    Returns:
        The ``(available, not_available)`` marks. The first one skips
            the test if the dependency is not available, the second one
            skips it if the dependency is available.

    Example:
        ```pycon
        >>> from coola.testing.fixtures import requires
        >>> numpy_available, numpy_not_available = requires(True, "NumPy")
        >>> numpy_available.kwargs["reason"]
        'Requires NumPy'
        >>> numpy_not_available.kwargs["reason"]
        'Skip if NumPy is available'

        ```
    """
    verb = "are" if plural else "is"
    return (
        pytest.mark.skipif(not is_available, reason=f"Requires {name}"),
        pytest.mark.skipif(is_available, reason=f"Skip if {name} {verb} available"),
    )


colorlog_available: pytest.MarkDecorator
colorlog_not_available: pytest.MarkDecorator
jax_available: pytest.MarkDecorator
jax_not_available: pytest.MarkDecorator
numpy_available: pytest.MarkDecorator
numpy_not_available: pytest.MarkDecorator
packaging_available: pytest.MarkDecorator
packaging_not_available: pytest.MarkDecorator
pandas_available: pytest.MarkDecorator
pandas_not_available: pytest.MarkDecorator
polars_available: pytest.MarkDecorator
polars_not_available: pytest.MarkDecorator
pyarrow_available: pytest.MarkDecorator
pyarrow_not_available: pytest.MarkDecorator
pydantic_available: pytest.MarkDecorator
pydantic_not_available: pytest.MarkDecorator
rich_available: pytest.MarkDecorator
rich_not_available: pytest.MarkDecorator
torch_available: pytest.MarkDecorator
torch_cuda_available: pytest.MarkDecorator
torch_cuda_not_available: pytest.MarkDecorator
torch_mps_available: pytest.MarkDecorator
torch_mps_not_available: pytest.MarkDecorator
torch_not_available: pytest.MarkDecorator
torch_numpy_available: pytest.MarkDecorator
torch_numpy_not_available: pytest.MarkDecorator
xarray_available: pytest.MarkDecorator
xarray_not_available: pytest.MarkDecorator

jax_available, jax_not_available = requires(is_jax_available(), "JAX")
numpy_available, numpy_not_available = requires(is_numpy_available(), "NumPy")
packaging_available, packaging_not_available = requires(is_packaging_available(), "packaging")
pandas_available, pandas_not_available = requires(is_pandas_available(), "pandas")
polars_available, polars_not_available = requires(is_polars_available(), "polars")
pyarrow_available, pyarrow_not_available = requires(is_pyarrow_available(), "pyarrow")
pydantic_available, pydantic_not_available = requires(is_pydantic_available(), "pydantic")
torch_available, torch_not_available = requires(is_torch_available(), "PyTorch")
torch_cuda_available, torch_cuda_not_available = requires(
    is_cuda_available(), "PyTorch and CUDA", plural=True
)
torch_numpy_available, torch_numpy_not_available = requires(
    is_torch_numpy_available(), "PyTorch and NumPy", plural=True
)
torch_mps_available, torch_mps_not_available = requires(
    is_mps_available(), "PyTorch and MPS", plural=True
)
xarray_available, xarray_not_available = requires(is_xarray_available(), "xarray")
colorlog_available, colorlog_not_available = requires(is_colorlog_available(), "colorlog")
rich_available, rich_not_available = requires(is_rich_available(), "rich")
