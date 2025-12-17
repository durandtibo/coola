from __future__ import annotations

import pytest

from coola.testing import (
    jax_available,
    jax_not_available,
    numpy_available,
    numpy_not_available,
    pandas_available,
    pandas_not_available,
    polars_available,
    polars_not_available,
)
from coola.utils import is_jax_available
from coola.utils.imports import (
    check_jax,
    check_numpy,
    check_pandas,
    check_polars,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
)

#################
#     jax     #
#################


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


####################
#     pandas     #
####################


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
