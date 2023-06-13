__all__ = [
    "AllCloseTester",
    "BaseAllCloseOperator",
    "BaseAllCloseTester",
    "BaseEqualityOperator",
    "BaseEqualityTester",
    "EqualityTester",
    "objects_are_allclose",
    "objects_are_equal",
]

from coola.allclose import (
    AllCloseTester,
    BaseAllCloseOperator,
    BaseAllCloseTester,
    objects_are_allclose,
)
from coola.equality import (
    BaseEqualityOperator,
    BaseEqualityTester,
    EqualityTester,
    objects_are_equal,
)
from coola.utils.imports import (
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
    is_xarray_available,
)

# Register NumPy comparators
if is_numpy_available():  # pragma: no cover
    from coola import _numpy  # noqa: F401

# Register pandas comparators
if is_pandas_available():  # pragma: no cover
    from coola import _pandas  # noqa: F401

# Register polars comparators
if is_polars_available():  # pragma: no cover
    from coola import _polars  # noqa: F401

# Register PyTorch comparators
if is_torch_available():  # pragma: no cover
    from coola import _torch  # noqa: F401

# Register xarray comparators
if is_xarray_available():  # pragma: no cover
    from coola import _xarray  # noqa: F401
