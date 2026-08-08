from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict

import pytest

from coola.factory import OBJECT_TARGET, factory, resolve_object


class BaseFakeClass(ABC):
    """Base abstract class for testing."""

    @abstractmethod
    def method(self) -> None:
        """Abstract method."""


class FakeClass:
    """Fake class used for testing factory functionality."""

    def __init__(self, arg1: int, arg2: str = "default") -> None:
        self.arg1 = arg1
        self.arg2 = arg2

    @classmethod
    def create_with_defaults(cls) -> FakeClass:
        """Create an instance with default values."""
        return cls(arg1=42, arg2="factory_method")

    @classmethod
    def create_with_custom_arg2(cls, arg2: str) -> FakeClass:
        """Create an instance with a custom arg2 value."""
        return cls(arg1=100, arg2=arg2)

    @staticmethod
    def static_create() -> FakeClass:
        """Create an instance using a static method."""
        return FakeClass(arg1=999, arg2="static")


def fake_function(arg1: int, arg2: str = "function_default") -> FakeClass:
    """Fake function that returns a FakeClass instance."""
    return FakeClass(arg1=arg1, arg2=arg2)


class Animal:
    """Base class for testing."""


class Dog(Animal):
    """Concrete subclass for testing."""

    def __init__(self, name: str = "Rex") -> None:
        self.name = name


#############################
#     Tests for factory     #
#############################


def test_factory_valid_object() -> None:
    """Test factory with a valid built-in class."""
    counter = factory("collections.Counter", [1, 2, 1, 3])
    assert isinstance(counter, Counter)
    assert counter.most_common(5) == [(1, 2), (2, 1), (3, 1)]


def test_factory_ordered_dict() -> None:
    """Test factory with OrderedDict class."""
    obj = factory("collections.OrderedDict", [("a", 1), ("b", 2)])
    assert isinstance(obj, OrderedDict)
    assert list(obj.keys()) == ["a", "b"]


def test_factory_with_kwargs() -> None:
    """Test factory with keyword arguments."""
    counter = factory("collections.Counter", a=4, b=2)
    assert counter["a"] == 4
    assert counter["b"] == 2


def test_factory_with_args_and_kwargs() -> None:
    """Test factory with both positional and keyword arguments."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", 10, arg2="mixed")
    assert isinstance(obj, FakeClass)
    assert obj.arg1 == 10
    assert obj.arg2 == "mixed"


def test_factory_with_only_kwargs() -> None:
    """Test factory with only keyword arguments."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", arg1=20, arg2="kwargs_only")
    assert isinstance(obj, FakeClass)
    assert obj.arg1 == 20
    assert obj.arg2 == "kwargs_only"


def test_factory_with_default_parameter() -> None:
    """Test factory where default parameter is used."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", 30)
    assert isinstance(obj, FakeClass)
    assert obj.arg1 == 30
    assert obj.arg2 == "default"


def test_factory_with_classmethod_init() -> None:
    """Test factory with custom class method as initializer."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", _init_="create_with_defaults")
    assert isinstance(obj, FakeClass)
    assert obj.arg1 == 42
    assert obj.arg2 == "factory_method"


def test_factory_with_classmethod_init_and_args() -> None:
    """Test factory with custom class method and arguments."""
    obj = factory(
        "tests.unit.factory.test_resolve.FakeClass",
        "custom_value",
        _init_="create_with_custom_arg2",
    )
    assert isinstance(obj, FakeClass)
    assert obj.arg1 == 100
    assert obj.arg2 == "custom_value"


def test_factory_with_classmethod_init_and_kwargs() -> None:
    """Test factory with custom class method and keyword arguments."""
    obj = factory(
        "tests.unit.factory.test_resolve.FakeClass",
        _init_="create_with_custom_arg2",
        arg2="kwarg_value",
    )
    assert isinstance(obj, FakeClass)
    assert obj.arg1 == 100
    assert obj.arg2 == "kwarg_value"


def test_factory_with_static_method_init() -> None:
    """Test factory with static method as initializer."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", _init_="static_create")
    assert isinstance(obj, FakeClass)
    assert obj.arg1 == 999
    assert obj.arg2 == "static"


def test_factory_with_new_init() -> None:
    """Test factory with __new__ as initializer."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", 50, "new_test", _init_="__new__")
    assert isinstance(obj, FakeClass)


def test_factory_function_target() -> None:
    """Test factory with a function as target."""
    obj = factory("tests.unit.factory.test_resolve.fake_function", 60)
    assert isinstance(obj, FakeClass)
    assert obj.arg1 == 60
    assert obj.arg2 == "function_default"


def test_factory_function_with_kwargs() -> None:
    """Test factory with a function and keyword arguments."""
    obj = factory("tests.unit.factory.test_resolve.fake_function", arg1=70, arg2="function_kwargs")
    assert obj.arg1 == 70
    assert obj.arg2 == "function_kwargs"


def test_factory_builtin_type() -> None:
    """Test factory with built-in types."""
    obj = factory("builtins.int", "42")
    assert obj == 42
    assert isinstance(obj, int)


def test_factory_builtin_list() -> None:
    """Test factory with built-in list type."""
    obj = factory("builtins.list", [1, 2, 3])
    assert obj == [1, 2, 3]
    assert isinstance(obj, list)


@pytest.mark.parametrize(("arg1", "arg2"), [(1, "a"), (10, "z"), (-5, "test")])
def test_factory_parametrized(arg1: int, arg2: str) -> None:
    """Test factory with parametrized inputs."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", arg1=arg1, arg2=arg2)
    assert obj.arg1 == arg1
    assert obj.arg2 == arg2


def test_factory_abstract_object() -> None:
    """Test that factory raises error when trying to instantiate
    abstract class."""
    with pytest.raises(
        TypeError,
        match=r"Cannot instantiate the class .* because it is an abstract class.",
    ):
        factory("tests.unit.factory.test_resolve.BaseFakeClass")


def test_factory_non_existing_object() -> None:
    """Test that factory raises error for non-existing target."""
    with pytest.raises(ImportError, match=r"The target object does not exist:"):
        factory("collections.NotACounter")


def test_factory_non_existing_module() -> None:
    """Test that factory raises error for non-existing module."""
    with pytest.raises(ImportError, match=r"The target object does not exist:"):
        factory("non_existing_module.SomeClass")


def test_factory_invalid_init_method() -> None:
    """Test that factory raises error for invalid _init_ method."""
    with pytest.raises(TypeError, match=r"does not have `invalid_method` attribute"):
        factory("tests.unit.factory.test_resolve.FakeClass", _init_="invalid_method")


def test_factory_empty_target() -> None:
    """Test that factory raises error for empty target string."""
    with pytest.raises((ImportError, ValueError)):
        factory("")


def test_factory_malformed_target() -> None:
    """Test that factory raises error for malformed target."""
    with pytest.raises(ImportError):
        factory("collections.")


def test_factory_with_custom_class() -> None:
    """Test factory with custom test class."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", arg1=123, arg2="test")
    assert obj.arg1 == 123
    assert obj.arg2 == "test"


def test_factory_preserves_type() -> None:
    """Test that factory preserves the correct type."""
    counter = factory("collections.Counter")
    assert type(counter).__name__ == "Counter"
    assert isinstance(counter, Counter)


####################################
#     Tests for resolve_object     #
####################################


# --- Pass-through ---


def test_resolve_object_returns_same_instance_when_no_cls() -> None:
    obj = Dog()
    assert resolve_object(obj) is obj


def test_resolve_object_returns_same_instance_with_matching_cls() -> None:
    obj = Dog()
    assert resolve_object(obj, cls=Dog) is obj


def test_resolve_object_returns_same_instance_with_parent_cls() -> None:
    obj = Dog()
    assert resolve_object(obj, cls=Animal) is obj


# --- Default cls=object ---


def test_resolve_object_default_cls_accepts_any_instance() -> None:
    assert resolve_object(42) == 42


def test_resolve_object_default_cls_accepts_string() -> None:
    assert resolve_object("hello") == "hello"


# --- From dict ---


def test_resolve_object_from_dict_returns_instance() -> None:
    result = resolve_object({OBJECT_TARGET: "tests.unit.factory.test_resolve.Dog"}, cls=Dog)
    assert isinstance(result, Dog)


def test_resolve_object_from_dict_no_cls_still_resolves() -> None:
    result = resolve_object({OBJECT_TARGET: "tests.unit.factory.test_resolve.Dog"})
    assert isinstance(result, Dog)


# --- Invalid input ---


def test_resolve_object_invalid_type_raises_incorrect_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a Animal instance"):
        resolve_object("not-an-animal", cls=Animal)


def test_resolve_object_wrong_subclass_raises_incorrect_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a Dog instance"):
        resolve_object(Animal(), cls=Dog)


def test_resolve_object_from_dict_wrong_cls_raises_incorrect_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a str instance"):
        resolve_object({OBJECT_TARGET: "tests.unit.factory.test_resolve.Dog"}, cls=str)


def test_resolve_object_from_dict_missing_target_raises_incorrect_type_error() -> None:
    with pytest.raises(TypeError, match=r"missing the `_target_` key"):
        resolve_object({"name": "Rex"}, cls=Dog)


# --- Dict subclasses are always treated as configuration ---


def test_resolve_object_dict_subclass_instance_is_treated_as_config() -> None:
    # Even though `od` is already a valid `OrderedDict` instance, it is a `dict`
    # subclass, so it is always treated as a factory configuration.
    od = OrderedDict(a=1)
    with pytest.raises(TypeError, match=r"missing the `_target_` key"):
        resolve_object(od, cls=OrderedDict)
