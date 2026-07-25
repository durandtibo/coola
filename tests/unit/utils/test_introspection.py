from __future__ import annotations

from typing import Any

import pytest

from coola.utils.introspection import (
    check_not_lambda_function,
    get_fully_qualified_name,
    is_lambda_function,
)


class Fake:
    """Fake class."""

    def __init__(self, arg1: int, arg2: str = "abc") -> None:
        self.arg1 = arg1
        self.arg2 = arg2

    def method(self) -> None:
        """Do nothing."""

    @staticmethod
    def static_method() -> Fake:
        return Fake(1, "qwerty")

    @classmethod
    def class_method(cls) -> Fake:
        return cls(arg1=35, arg2="bac")


class Outer:
    class Inner:
        pass


def fake_func(arg1: int, arg2: str = "abc") -> Fake:
    """Fake function."""
    return Fake(arg1=arg1, arg2=arg2)


###############################################
#     Tests for check_not_lambda_function     #
###############################################


def test_check_not_lambda_function_lambda_raises_error() -> None:
    with pytest.raises(TypeError, match=r"lambda function"):
        check_not_lambda_function(lambda x: x)


def test_check_not_lambda_function_regular_function_does_not_raise() -> None:
    check_not_lambda_function(fake_func)


@pytest.mark.parametrize("obj", [-1, "abc", Fake])
def test_check_not_lambda_function_non_function_does_not_raise(obj: Any) -> None:
    check_not_lambda_function(obj)


####################################
#     get_fully_qualified_name     #
####################################


def test_get_fully_qualified_name_builtin() -> None:
    assert get_fully_qualified_name(int) == "builtins.int"


def test_get_fully_qualified_name_class() -> None:
    assert get_fully_qualified_name(Fake) == "tests.unit.utils.test_introspection.Fake"


def test_get_fully_qualified_name_method() -> None:
    assert (
        get_fully_qualified_name(Fake(1).method)
        == "tests.unit.utils.test_introspection.Fake.method"
    )


def test_get_fully_qualified_name_class_method() -> None:
    assert (
        get_fully_qualified_name(Fake.class_method)
        == "tests.unit.utils.test_introspection.Fake.class_method"
    )


def test_get_fully_qualified_name_static_method() -> None:
    assert (
        get_fully_qualified_name(Fake.static_method)
        == "tests.unit.utils.test_introspection.Fake.static_method"
    )


def test_get_fully_qualified_name_outer_class() -> None:
    assert get_fully_qualified_name(Outer) == "tests.unit.utils.test_introspection.Outer"


def test_get_fully_qualified_name_outer_object() -> None:
    assert get_fully_qualified_name(Outer()) == "tests.unit.utils.test_introspection.Outer"


def test_get_fully_qualified_name_inner() -> None:
    assert (
        get_fully_qualified_name(Outer.Inner) == "tests.unit.utils.test_introspection.Outer.Inner"
    )


def test_get_fully_qualified_name_local_class() -> None:
    class Fake: ...

    assert get_fully_qualified_name(Fake) == (
        "tests.unit.utils.test_introspection.test_get_fully_qualified_name_local_class"
        ".<locals>.Fake"
    )


def test_get_fully_qualified_name_function() -> None:
    assert get_fully_qualified_name(fake_func) == "tests.unit.utils.test_introspection.fake_func"


def test_get_fully_qualified_name_main_module_fallback() -> None:
    class Fake:
        pass

    Fake.__module__ = "__main__"

    assert (
        get_fully_qualified_name(Fake)
        == "test_get_fully_qualified_name_main_module_fallback.<locals>.Fake"
    )


def test_get_fully_qualified_name_object_in_main_module_returns_qualname_only() -> None:
    class MyClass:
        pass

    MyClass.__module__ = "__main__"  # override for testing
    assert get_fully_qualified_name(MyClass) == MyClass.__qualname__


def test_get_fully_qualified_name_builtin_function() -> None:
    assert get_fully_qualified_name(map) == "builtins.map"


########################################
#     Tests for is_lambda_function     #
########################################


def test_is_lambda_function_lambda() -> None:
    assert is_lambda_function(lambda x: x)


def test_is_lambda_function_regular_function() -> None:
    assert not is_lambda_function(fake_func)


@pytest.mark.parametrize("obj", [-1, "abc", Fake])
def test_is_lambda_function_non_function(obj: Any) -> None:
    assert not is_lambda_function(obj)
