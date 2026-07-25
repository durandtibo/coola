from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pytest

from coola.utils.introspection import (
    check_not_lambda_function,
    get_all_child_classes,
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


###########################################
#     Tests for get_all_child_classes     #
###########################################


def test_get_all_child_classes() -> None:
    """Test get_all_child_classes with a simple hierarchy."""

    class Foo: ...

    assert get_all_child_classes(Foo) == set()

    class Bar(Foo): ...

    assert get_all_child_classes(Foo) == {Bar}

    class Baz(Foo): ...

    assert get_all_child_classes(Foo) == {Bar, Baz}

    class Bing(Bar): ...

    assert get_all_child_classes(Foo) == {Bar, Baz, Bing}


def test_get_all_child_classes_empty_hierarchy() -> None:
    """Test get_all_child_classes with a class that has no children."""

    class Standalone: ...

    assert get_all_child_classes(Standalone) == set()


def test_get_all_child_classes_single_child() -> None:
    """Test get_all_child_classes with a single child class."""

    class Parent: ...

    class Child(Parent): ...

    assert get_all_child_classes(Parent) == {Child}


def test_get_all_child_classes_multiple_levels() -> None:
    """Test get_all_child_classes with multiple inheritance levels."""

    class Level0: ...

    class Level1A(Level0): ...

    class Level1B(Level0): ...

    class Level2A(Level1A): ...

    class Level2B(Level1A): ...

    class Level2C(Level1B): ...

    assert get_all_child_classes(Level0) == {Level1A, Level1B, Level2A, Level2B, Level2C}


def test_get_all_child_classes_multiple_inheritance() -> None:
    """Test get_all_child_classes with multiple inheritance (diamond
    pattern)."""

    class Base: ...

    class Left(Base): ...

    class Right(Base): ...

    class Diamond(Left, Right): ...

    assert get_all_child_classes(Base) == {Left, Right, Diamond}


def test_get_all_child_classes_with_abstract_base_class() -> None:
    """Test get_all_child_classes with abstract base classes."""

    class AbstractBase(ABC):
        @abstractmethod
        def method(self) -> None:
            pass

    class ConcreteChild(AbstractBase):
        def method(self) -> None:
            pass

    assert get_all_child_classes(AbstractBase) == {ConcreteChild}


def test_get_all_child_classes_deep_hierarchy() -> None:
    """Test get_all_child_classes with a deep inheritance hierarchy."""

    class Level0: ...

    class Level1(Level0): ...

    class Level2(Level1): ...

    class Level3(Level2): ...

    class Level4(Level3): ...

    assert get_all_child_classes(Level0) == {Level1, Level2, Level3, Level4}


def test_get_all_child_classes_siblings() -> None:
    """Test get_all_child_classes with sibling classes."""

    class Parent: ...

    class Sibling1(Parent): ...

    class Sibling2(Parent): ...

    class Sibling3(Parent): ...

    assert get_all_child_classes(Parent) == {Sibling1, Sibling2, Sibling3}


def test_get_all_child_classes_mixed_inheritance() -> None:
    """Test get_all_child_classes with mixed single and multiple
    inheritance."""

    class Base: ...

    class Mixin: ...

    class Child1(Base): ...

    class Child2(Base, Mixin): ...

    # Child2 should be a child of Base even though it has multiple parents
    assert get_all_child_classes(Base) == {Child1, Child2}


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
