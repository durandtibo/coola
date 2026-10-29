from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from coola.utils.introspection import (
    check_not_lambda_function,
    get_all_child_classes,
    get_qualified_class_name,
    has_generated_dataclass_eq,
    is_lambda_function,
    is_same_type,
    supports_methods,
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
    """Test get_all_child_classes with a class without child."""

    class Foo: ...

    assert get_all_child_classes(Foo) == set()


def test_get_all_child_classes_one_child() -> None:
    """Test get_all_child_classes with one child."""

    class Foo: ...

    class Bar(Foo): ...

    assert get_all_child_classes(Foo) == {Bar}


def test_get_all_child_classes_two_children() -> None:
    """Test get_all_child_classes with two children."""

    class Foo: ...

    class Bar(Foo): ...

    class Baz(Foo): ...

    assert get_all_child_classes(Foo) == {Bar, Baz}


def test_get_all_child_classes_grandchild() -> None:
    """Test get_all_child_classes with a grandchild."""

    class Foo: ...

    class Bar(Foo): ...

    class Baz(Foo): ...

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


######################################
#     Tests for supports_methods     #
######################################


class _WithMethods:
    def allclose(self) -> bool:
        return True

    def equal(self) -> bool:
        return True

    not_callable = 42


def test_supports_methods_no_method_names() -> None:
    assert supports_methods(_WithMethods())


def test_supports_methods_single_method_true() -> None:
    assert supports_methods(_WithMethods(), "allclose")


def test_supports_methods_multiple_methods_true() -> None:
    assert supports_methods(_WithMethods(), "allclose", "equal")


def test_supports_methods_missing_method_false() -> None:
    assert not supports_methods(_WithMethods(), "does_not_exist")


def test_supports_methods_one_missing_among_several_false() -> None:
    assert not supports_methods(_WithMethods(), "allclose", "does_not_exist")


def test_supports_methods_attribute_not_callable_false() -> None:
    assert not supports_methods(_WithMethods(), "not_callable")


def test_supports_methods_on_plain_object_false() -> None:
    assert not supports_methods(object(), "allclose")


def test_supports_methods_on_int_true() -> None:
    assert supports_methods(1, "bit_length")


##################################
#     Tests for is_same_type     #
##################################


class Parent:
    """Parent class."""


class Child(Parent):
    """Child class."""


def test_is_same_type_true() -> None:
    assert is_same_type(Parent(), Parent())


def test_is_same_type_false_different_classes() -> None:
    assert not is_same_type(Parent(), Fake(1))


def test_is_same_type_false_subclass() -> None:
    assert not is_same_type(Child(), Parent())


def test_is_same_type_false_parent() -> None:
    assert not is_same_type(Parent(), Child())


def test_is_same_type_bool_is_not_int() -> None:
    assert not is_same_type(True, 1)


def test_is_same_type_int() -> None:
    assert is_same_type(1, 2)


def test_is_same_type_str() -> None:
    assert is_same_type("a", "b")


def test_is_same_type_int_and_float() -> None:
    assert not is_same_type(1, 1.0)


class Point:
    def __init__(self, x: int) -> None:
        self.x = x

    def equal(self, other: object) -> bool:
        if not is_same_type(other, self):
            return False
        return self.x == other.x


def test_is_same_type_narrows_for_attribute_access() -> None:
    assert Point(1).equal(Point(1))


def test_is_same_type_narrows_for_attribute_access_different_value() -> None:
    assert not Point(1).equal(Point(2))


def test_is_same_type_narrows_for_attribute_access_different_type() -> None:
    assert not Point(1).equal(1)


##############################################
#     Tests for has_generated_dataclass_eq     #
##############################################


@dataclass
class GeneratedEq:
    x: int = 1


@dataclass(eq=False)
class NoEq:
    x: int = 1


@dataclass
class CustomEq:
    x: int = 1

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CustomEq)

    def __hash__(self) -> int:
        return 0


class NotDataclass:
    pass


def test_has_generated_dataclass_eq_generated() -> None:
    assert has_generated_dataclass_eq(GeneratedEq)


def test_has_generated_dataclass_eq_eq_false() -> None:
    assert not has_generated_dataclass_eq(NoEq)


def test_has_generated_dataclass_eq_custom_eq() -> None:
    assert not has_generated_dataclass_eq(CustomEq)


def test_has_generated_dataclass_eq_not_dataclass() -> None:
    assert not has_generated_dataclass_eq(NotDataclass)


def test_has_generated_dataclass_eq_builtin() -> None:
    assert not has_generated_dataclass_eq(int)


##############################################
#     Tests for get_qualified_class_name     #
##############################################


def test_get_qualified_class_name_builtin() -> None:
    assert get_qualified_class_name(int) == "int"


def test_get_qualified_class_name_standard_library() -> None:
    assert get_qualified_class_name(Path) == "pathlib.Path"


def test_get_qualified_class_name_nested_class() -> None:
    assert get_qualified_class_name(Outer.Inner) == f"{Outer.__module__}.Outer.Inner"


def test_get_qualified_class_name_main_module_is_kept() -> None:
    cls = type("Fake", (), {"__module__": "__main__"})
    assert get_qualified_class_name(cls) == "__main__.Fake"
