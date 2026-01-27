from __future__ import annotations

import logging
from functools import partial
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_package,
    decorator_package_available,
    module_available,
    package_available,
    raise_package_missing_error,
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
    with patch("coola.utils.imports.universal.package_available", lambda name: name != "missing"):
        check_package("exist")


def test_check_package_missing() -> None:
    with (
        patch("coola.utils.imports.universal.package_available", lambda name: name != "missing"),
        pytest.raises(RuntimeError, match=r"'missing' package is required but not installed."),
    ):
        check_package("missing")


def test_check_package_missing_with_command() -> None:
    msg = (
        "'missing' package is required but not installed. "
        "You can install 'missing' package with the command:\n\npip install missing"
    )
    with (
        patch("coola.utils.imports.universal.package_available", lambda name: name != "missing"),
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


##############################################
#     Tests for raise_package_missing_error  #
##############################################


def test_raise_package_missing_error_basic() -> None:
    with pytest.raises(RuntimeError, match=r"'mypackage' package is required but not installed."):
        raise_package_missing_error("mypackage", "mypackage")


def test_raise_package_missing_error_different_install_cmd() -> None:
    with pytest.raises(RuntimeError, match=r"'git' package is required but not installed."):
        raise_package_missing_error("git", "gitpython")


def test_raise_package_missing_error_message_format() -> None:
    msg = (
        "'testpkg' package is required but not installed. "
        "You can install 'testpkg' package with the command:\n\n"
        "pip install test-package\n\nor\n\n"
        "uv pip install test-package\n"
    )
    with pytest.raises(RuntimeError, match=msg):
        raise_package_missing_error("testpkg", "test-package")
