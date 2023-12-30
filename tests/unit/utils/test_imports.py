from __future__ import annotations

import logging
from functools import partial
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_jax,
    check_numpy,
    check_pandas,
    check_polars,
    check_torch,
    check_xarray,
    decorator_package_available,
    is_jax_available,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
    is_xarray_available,
    jax_available,
    numpy_available,
    pandas_available,
    polars_available,
    torch_available,
    xarray_available,
)

logger = logging.getLogger(__name__)


def my_function(n: int = 0) -> int:
    return 42 + n


#################################################
#     Tests for decorator_package_available     #
#################################################


def test_decorator_package_available_condition_true() -> None:
    fn = decorator_package_available(my_function, condition=lambda *args: True)
    assert fn() == 42


def test_decorator_package_available_condition_true_args() -> None:
    fn = decorator_package_available(my_function, condition=lambda *args: True)
    assert fn(2) == 44


def test_decorator_package_available_condition_false() -> None:
    fn = decorator_package_available(my_function, condition=lambda *args: False)
    assert fn(2) is None


def test_decorator_package_available_decorator_condition_true() -> None:
    decorator = partial(decorator_package_available, condition=lambda *args: True)

    @decorator
    def fn(n: int = 0) -> int:
        return 42 + n

    assert fn() == 42


def test_decorator_package_available_decorator_condition_true_args() -> None:
    decorator = partial(decorator_package_available, condition=lambda *args: True)

    @decorator
    def fn(n: int = 0) -> int:
        return 42 + n

    assert fn(2) == 44


def test_decorator_package_available_decorator_condition_false() -> None:
    decorator = partial(decorator_package_available, condition=lambda *args: False)

    @decorator
    def fn(n: int = 0) -> int:
        return 42 + n

    assert fn(2) is None


###############
#     jax     #
###############


def test_check_jax_with_package() -> None:
    with patch("coola.utils.imports.is_jax_available", lambda *args: True):
        check_jax()


def test_check_jax_without_package() -> None:
    with patch("coola.utils.imports.is_jax_available", lambda *args: False), pytest.raises(
        RuntimeError, match="`jax` package is required but not installed."
    ):
        check_jax()


def test_is_jax_available() -> None:
    assert isinstance(is_jax_available(), bool)


def test_jax_available_with_package() -> None:
    with patch("coola.utils.imports.is_jax_available", lambda *args: True):
        fn = jax_available(my_function)
        assert fn(2) == 44


def test_jax_available_without_package() -> None:
    with patch("coola.utils.imports.is_jax_available", lambda *args: False):
        fn = jax_available(my_function)
        assert fn(2) is None


def test_jax_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_jax_available", lambda *args: True):

        @jax_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_jax_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_jax_available", lambda *args: False):

        @jax_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


#################
#     numpy     #
#################


def test_check_numpy_with_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args: True):
        check_numpy()


def test_check_numpy_without_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args: False), pytest.raises(
        RuntimeError, match="`numpy` package is required but not installed."
    ):
        check_numpy()


def test_is_numpy_available() -> None:
    assert isinstance(is_numpy_available(), bool)


def test_numpy_available_with_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args: True):
        fn = numpy_available(my_function)
        assert fn(2) == 44


def test_numpy_available_without_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args: False):
        fn = numpy_available(my_function)
        assert fn(2) is None


def test_numpy_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args: True):

        @numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_numpy_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args: False):

        @numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


##################
#     pandas     #
##################


def test_check_pandas_with_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda *args: True):
        check_pandas()


def test_check_pandas_without_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda *args: False), pytest.raises(
        RuntimeError, match="`pandas` package is required but not installed."
    ):
        check_pandas()


def test_is_pandas_available() -> None:
    assert isinstance(is_pandas_available(), bool)


def test_pandas_available_with_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda *args: True):
        fn = pandas_available(my_function)
        assert fn(2) == 44


def test_pandas_available_without_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda *args: False):
        fn = pandas_available(my_function)
        assert fn(2) is None


def test_pandas_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda *args: True):

        @pandas_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_pandas_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda *args: False):

        @pandas_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


##################
#     polars     #
##################


def test_check_polars_with_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args: True):
        check_polars()


def test_check_polars_without_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args: False), pytest.raises(
        RuntimeError, match="`polars` package is required but not installed."
    ):
        check_polars()


def test_is_polars_available() -> None:
    assert isinstance(is_polars_available(), bool)


def test_polars_available_with_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args: True):
        fn = polars_available(my_function)
        assert fn(2) == 44


def test_polars_available_without_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args: False):
        fn = polars_available(my_function)
        assert fn(2) is None


def test_polars_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args: True):

        @polars_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_polars_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args: False):

        @polars_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


#################
#     torch     #
#################


def test_check_torch_with_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args: True):
        check_torch()


def test_check_torch_without_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args: False), pytest.raises(
        RuntimeError, match="`torch` package is required but not installed."
    ):
        check_torch()


def test_is_torch_available() -> None:
    assert isinstance(is_torch_available(), bool)


def test_torch_available_with_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args: True):
        fn = torch_available(my_function)
        assert fn(2) == 44


def test_torch_available_without_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args: False):
        fn = torch_available(my_function)
        assert fn(2) is None


def test_torch_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args: True):

        @torch_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_torch_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args: False):

        @torch_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


##################
#     xarray     #
##################


def test_check_xarray_with_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args: True):
        check_xarray()


def test_check_xarray_without_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args: False), pytest.raises(
        RuntimeError, match="`xarray` package is required but not installed."
    ):
        check_xarray()


def test_is_xarray_available() -> None:
    assert isinstance(is_xarray_available(), bool)


def test_xarray_available_with_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args: True):
        fn = xarray_available(my_function)
        assert fn(2) == 44


def test_xarray_available_without_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args: False):
        fn = xarray_available(my_function)
        assert fn(2) is None


def test_xarray_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args: True):

        @xarray_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_xarray_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args: False):

        @xarray_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None
