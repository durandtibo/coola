from __future__ import annotations

import logging
from functools import partial
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    LazyModule,
    check_jax,
    check_numpy,
    check_package,
    check_packaging,
    check_pandas,
    check_polars,
    check_pyarrow,
    check_torch,
    check_xarray,
    decorator_package_available,
    is_jax_available,
    is_numpy_available,
    is_packaging_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_torch_available,
    is_xarray_available,
    jax_available,
    lazy_import,
    module_available,
    numpy_available,
    package_available,
    packaging_available,
    pandas_available,
    polars_available,
    pyarrow_available,
    torch_available,
    xarray_available,
)

logger = logging.getLogger(__name__)


def my_function(n: int = 0) -> int:
    return 42 + n


#######################################
#     Tests for package_available     #
#######################################


def test_package_available_true() -> None:
    assert package_available("os")


def test_package_available_false() -> None:
    assert not package_available("missing_package")


def test_package_available_false_subpackage() -> None:
    assert not package_available("missing.package")


######################################
#     Tests for module_available     #
######################################


def test_module_available_true() -> None:
    assert module_available("os")


def test_module_available_false() -> None:
    assert not module_available("os.missing")


def test_module_available_false_submodule() -> None:
    assert not module_available("missing.module")


###################################
#     Tests for check_package     #
###################################


def test_check_package_exist() -> None:
    with patch("coola.utils.imports.package_available", lambda name: name != "missing"):
        check_package("exist")


def test_check_package_missing() -> None:
    with (
        patch("coola.utils.imports.package_available", lambda name: name != "missing"),
        pytest.raises(RuntimeError, match="'missing' package is required but not installed."),
    ):
        check_package("missing")


def test_check_package_missing_with_command() -> None:
    msg = (
        "'missing' package is required but not installed. "
        "You can install 'missing' package with the command:\n\npip install missing"
    )
    with (
        patch("coola.utils.imports.package_available", lambda name: name != "missing"),
        pytest.raises(RuntimeError, match=msg),
    ):
        check_package("missing", command="pip install missing")


#################################################
#     Tests for decorator_package_available     #
#################################################


def test_decorator_package_available_condition_true() -> None:
    fn = decorator_package_available(my_function, condition=lambda: True)
    assert fn() == 42


def test_decorator_package_available_condition_true_args() -> None:
    fn = decorator_package_available(my_function, condition=lambda: True)
    assert fn(2) == 44


def test_decorator_package_available_condition_false() -> None:
    fn = decorator_package_available(my_function, condition=lambda: False)
    assert fn(2) is None


def test_decorator_package_available_decorator_condition_true() -> None:
    decorator = partial(decorator_package_available, condition=lambda: True)

    @decorator
    def fn(n: int = 0) -> int:
        return 42 + n

    assert fn() == 42


def test_decorator_package_available_decorator_condition_true_args() -> None:
    decorator = partial(decorator_package_available, condition=lambda: True)

    @decorator
    def fn(n: int = 0) -> int:
        return 42 + n

    assert fn(2) == 44


def test_decorator_package_available_decorator_condition_false() -> None:
    decorator = partial(decorator_package_available, condition=lambda: False)

    @decorator
    def fn(n: int = 0) -> int:
        return 42 + n

    assert fn(2) is None


###############
#     jax     #
###############


def test_check_jax_with_package() -> None:
    with patch("coola.utils.imports.is_jax_available", lambda: True):
        check_jax()


def test_check_jax_without_package() -> None:
    with (
        patch("coola.utils.imports.is_jax_available", lambda: False),
        pytest.raises(RuntimeError, match="'jax' package is required but not installed."),
    ):
        check_jax()


def test_is_jax_available() -> None:
    assert isinstance(is_jax_available(), bool)


def test_jax_available_with_package() -> None:
    with patch("coola.utils.imports.is_jax_available", lambda: True):
        fn = jax_available(my_function)
        assert fn(2) == 44


def test_jax_available_without_package() -> None:
    with patch("coola.utils.imports.is_jax_available", lambda: False):
        fn = jax_available(my_function)
        assert fn(2) is None


def test_jax_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_jax_available", lambda: True):

        @jax_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_jax_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_jax_available", lambda: False):

        @jax_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


#################
#     numpy     #
#################


def test_check_numpy_with_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda: True):
        check_numpy()


def test_check_numpy_without_package() -> None:
    with (
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match="'numpy' package is required but not installed."),
    ):
        check_numpy()


def test_is_numpy_available() -> None:
    assert isinstance(is_numpy_available(), bool)


def test_numpy_available_with_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda: True):
        fn = numpy_available(my_function)
        assert fn(2) == 44


def test_numpy_available_without_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda: False):
        fn = numpy_available(my_function)
        assert fn(2) is None


def test_numpy_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda: True):

        @numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_numpy_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda: False):

        @numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


#####################
#     packaging     #
#####################


def test_check_packaging_with_package() -> None:
    with patch("coola.utils.imports.is_packaging_available", lambda: True):
        check_packaging()


def test_check_packaging_without_package() -> None:
    with (
        patch("coola.utils.imports.is_packaging_available", lambda: False),
        pytest.raises(RuntimeError, match="'packaging' package is required but not installed."),
    ):
        check_packaging()


def test_is_packaging_available() -> None:
    assert isinstance(is_packaging_available(), bool)


def test_packaging_available_with_package() -> None:
    with patch("coola.utils.imports.is_packaging_available", lambda: True):
        fn = packaging_available(my_function)
        assert fn(2) == 44


def test_packaging_available_without_package() -> None:
    with patch("coola.utils.imports.is_packaging_available", lambda: False):
        fn = packaging_available(my_function)
        assert fn(2) is None


def test_packaging_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_packaging_available", lambda: True):

        @packaging_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_packaging_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_packaging_available", lambda: False):

        @packaging_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


##################
#     pandas     #
##################


def test_check_pandas_with_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda: True):
        check_pandas()


def test_check_pandas_without_package() -> None:
    with (
        patch("coola.utils.imports.is_pandas_available", lambda: False),
        pytest.raises(RuntimeError, match="'pandas' package is required but not installed."),
    ):
        check_pandas()


def test_is_pandas_available() -> None:
    assert isinstance(is_pandas_available(), bool)


def test_pandas_available_with_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda: True):
        fn = pandas_available(my_function)
        assert fn(2) == 44


def test_pandas_available_without_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda: False):
        fn = pandas_available(my_function)
        assert fn(2) is None


def test_pandas_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda: True):

        @pandas_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_pandas_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda: False):

        @pandas_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


##################
#     polars     #
##################


def test_check_polars_with_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda: True):
        check_polars()


def test_check_polars_without_package() -> None:
    with (
        patch("coola.utils.imports.is_polars_available", lambda: False),
        pytest.raises(RuntimeError, match="'polars' package is required but not installed."),
    ):
        check_polars()


def test_is_polars_available() -> None:
    assert isinstance(is_polars_available(), bool)


def test_polars_available_with_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda: True):
        fn = polars_available(my_function)
        assert fn(2) == 44


def test_polars_available_without_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda: False):
        fn = polars_available(my_function)
        assert fn(2) is None


def test_polars_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda: True):

        @polars_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_polars_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda: False):

        @polars_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


###################
#     pyarrow     #
###################


def test_check_pyarrow_with_package() -> None:
    with patch("coola.utils.imports.is_pyarrow_available", lambda: True):
        check_pyarrow()


def test_check_pyarrow_without_package() -> None:
    with (
        patch("coola.utils.imports.is_pyarrow_available", lambda: False),
        pytest.raises(RuntimeError, match="'pyarrow' package is required but not installed."),
    ):
        check_pyarrow()


def test_is_pyarrow_available() -> None:
    assert isinstance(is_pyarrow_available(), bool)


def test_pyarrow_available_with_package() -> None:
    with patch("coola.utils.imports.is_pyarrow_available", lambda: True):
        fn = pyarrow_available(my_function)
        assert fn(2) == 44


def test_pyarrow_available_without_package() -> None:
    with patch("coola.utils.imports.is_pyarrow_available", lambda: False):
        fn = pyarrow_available(my_function)
        assert fn(2) is None


def test_pyarrow_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_pyarrow_available", lambda: True):

        @pyarrow_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_pyarrow_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_pyarrow_available", lambda: False):

        @pyarrow_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


#################
#     torch     #
#################


def test_check_torch_with_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda: True):
        check_torch()


def test_check_torch_without_package() -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match="'torch' package is required but not installed."),
    ):
        check_torch()


def test_is_torch_available() -> None:
    assert isinstance(is_torch_available(), bool)


def test_torch_available_with_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda: True):
        fn = torch_available(my_function)
        assert fn(2) == 44


def test_torch_available_without_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda: False):
        fn = torch_available(my_function)
        assert fn(2) is None


def test_torch_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda: True):

        @torch_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_torch_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda: False):

        @torch_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


##################
#     xarray     #
##################


def test_check_xarray_with_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda: True):
        check_xarray()


def test_check_xarray_without_package() -> None:
    with (
        patch("coola.utils.imports.is_xarray_available", lambda: False),
        pytest.raises(RuntimeError, match="'xarray' package is required but not installed."),
    ):
        check_xarray()


def test_is_xarray_available() -> None:
    assert isinstance(is_xarray_available(), bool)


def test_xarray_available_with_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda: True):
        fn = xarray_available(my_function)
        assert fn(2) == 44


def test_xarray_available_without_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda: False):
        fn = xarray_available(my_function)
        assert fn(2) is None


def test_xarray_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda: True):

        @xarray_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_xarray_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda: False):

        @xarray_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


######################
#     LazyModule     #
######################


def test_lazy_module() -> None:
    os = LazyModule("os")
    assert isinstance(os.getcwd(), str)
    assert isinstance(os.cpu_count(), int)
    with pytest.raises(AttributeError, match="module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_module_getattr_first() -> None:
    pathlib = LazyModule("pathlib")
    assert isinstance(pathlib.Path.cwd().as_posix(), str)
    assert "Path" in dir(pathlib)


def test_lazy_module_dir_first() -> None:
    pathlib = LazyModule("pathlib")
    assert "Path" in dir(pathlib)
    assert isinstance(pathlib.Path.cwd().as_posix(), str)


def test_lazy_module_missing_attribute_first() -> None:
    os = LazyModule("os")
    with pytest.raises(AttributeError, match="module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_module_missing() -> None:
    with pytest.raises(ModuleNotFoundError, match="No module named 'missing'"):
        str(LazyModule("missing"))


#######################
#     lazy_import     #
#######################


def test_lazy_import() -> None:
    os = lazy_import("os")
    assert isinstance(os.getcwd(), str)
    assert isinstance(os.cpu_count(), int)
    with pytest.raises(AttributeError, match="module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_import_getattr_first() -> None:
    pathlib = lazy_import("pathlib")
    assert isinstance(pathlib.Path.cwd().as_posix(), str)
    assert "Path" in dir(pathlib)


def test_lazy_import_dir_first() -> None:
    pathlib = lazy_import("pathlib")
    assert "Path" in dir(pathlib)
    assert isinstance(pathlib.Path.cwd().as_posix(), str)


def test_lazy_import_missing_attribute_first() -> None:
    os = lazy_import("os")
    with pytest.raises(AttributeError, match="module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_import_missing() -> None:
    with pytest.raises(ModuleNotFoundError, match="No module named 'missing'"):
        str(lazy_import("missing"))
