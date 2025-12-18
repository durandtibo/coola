from __future__ import annotations

import pytest

from coola.testing import (
    jax_available,
    jax_not_available,
    numpy_available,
    numpy_not_available,
    packaging_available,
    packaging_not_available,
    pandas_available,
    pandas_not_available,
    polars_available,
    polars_not_available,
    pyarrow_available,
    pyarrow_not_available,
    torch_available,
    torch_not_available,
    xarray_available,
    xarray_not_available,
)
from coola.utils.imports import (
    check_jax,
    check_numpy,
    check_packaging,
    check_pandas,
    check_polars,
    check_pyarrow,
    check_torch,
    check_xarray,
    is_jax_available,
    is_numpy_available,
    is_packaging_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_torch_available,
    is_xarray_available,
)

###############
#     jax     #
###############


@jax_available
def test_check_jax_with_package() -> None:
    check_jax()


@jax_not_available
def test_check_jax_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'jax' package is required but not installed."):
        check_jax()


@jax_available
def test_is_jax_available_true() -> None:
    assert is_jax_available()


@jax_not_available
def test_is_jax_available_false() -> None:
    assert not is_jax_available()


#################
#     numpy     #
#################


@numpy_available
def test_check_numpy_with_package() -> None:
    check_numpy()


@numpy_not_available
def test_check_numpy_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'numpy' package is required but not installed."):
        check_numpy()


@numpy_available
def test_is_numpy_available_true() -> None:
    assert is_numpy_available()


@numpy_not_available
def test_is_numpy_available_false() -> None:
    assert not is_numpy_available()


#####################
#     packaging     #
#####################


@packaging_available
def test_check_packaging_with_package() -> None:
    check_packaging()


@packaging_not_available
def test_check_packaging_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'packaging' package is required but not installed."):
        check_packaging()


@packaging_available
def test_is_packaging_available_true() -> None:
    assert is_packaging_available()


@packaging_not_available
def test_is_packaging_available_false() -> None:
    assert not is_packaging_available()


##################
#     pandas     #
##################


@pandas_available
def test_check_pandas_with_package() -> None:
    check_pandas()


@pandas_not_available
def test_check_pandas_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'pandas' package is required but not installed."):
        check_pandas()


@pandas_available
def test_is_pandas_available_true() -> None:
    assert is_pandas_available()


@pandas_not_available
def test_is_pandas_available_false() -> None:
    assert not is_pandas_available()


##################
#     polars     #
##################


@polars_available
def test_check_polars_with_package() -> None:
    check_polars()


@polars_not_available
def test_check_polars_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."):
        check_polars()


@polars_available
def test_is_polars_available_true() -> None:
    assert is_polars_available()


@polars_not_available
def test_is_polars_available_false() -> None:
    assert not is_polars_available()


###################
#     pyarrow     #
###################


@pyarrow_available
def test_check_pyarrow_with_package() -> None:
    check_pyarrow()


@pyarrow_not_available
def test_check_pyarrow_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'pyarrow' package is required but not installed."):
        check_pyarrow()


@pyarrow_available
def test_is_pyarrow_available_true() -> None:
    assert is_pyarrow_available()


@pyarrow_not_available
def test_is_pyarrow_available_false() -> None:
    assert not is_pyarrow_available()


#################
#     torch     #
#################


@torch_available
def test_check_torch_with_package() -> None:
    check_torch()


@torch_not_available
def test_check_torch_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."):
        check_torch()


@torch_available
def test_is_torch_available_true() -> None:
    assert is_torch_available()


@torch_not_available
def test_is_torch_available_false() -> None:
    assert not is_torch_available()


##################
#     xarray     #
##################


@xarray_available
def test_check_xarray_with_package() -> None:
    check_xarray()


@xarray_not_available
def test_check_xarray_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."):
        check_xarray()


@xarray_available
def test_is_xarray_available_true() -> None:
    assert is_xarray_available()


@xarray_not_available
def test_is_xarray_available_false() -> None:
    assert not is_xarray_available()
