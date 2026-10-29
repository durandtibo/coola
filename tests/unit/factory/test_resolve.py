from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from collections.abc import Mapping
from unittest.mock import patch

import pytest

from coola.factory import OBJECT_INIT, OBJECT_PREFIXES, OBJECT_TARGET, factory, resolve_object


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


def test_factory_valid_object_type() -> None:
    """Test factory with a valid built-in class."""
    counter = factory("collections.Counter", [1, 2, 1, 3])
    assert isinstance(counter, Counter)


def test_factory_valid_object_most_common() -> None:
    counter = factory("collections.Counter", [1, 2, 1, 3])
    assert counter.most_common(5) == [(1, 2), (2, 1), (3, 1)]


def test_factory_ordered_dict_type() -> None:
    """Test factory with OrderedDict class."""
    obj = factory("collections.OrderedDict", [("a", 1), ("b", 2)])
    assert isinstance(obj, OrderedDict)


def test_factory_ordered_dict_keys() -> None:
    obj = factory("collections.OrderedDict", [("a", 1), ("b", 2)])
    assert list(obj.keys()) == ["a", "b"]


def test_factory_with_kwargs_a() -> None:
    """Test factory with keyword arguments."""
    counter = factory("collections.Counter", a=4, b=2)
    assert counter["a"] == 4


def test_factory_with_kwargs_b() -> None:
    counter = factory("collections.Counter", a=4, b=2)
    assert counter["b"] == 2


def test_factory_with_args_and_kwargs_type() -> None:
    """Test factory with both positional and keyword arguments."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", 10, arg2="mixed")
    assert isinstance(obj, FakeClass)


def test_factory_with_args_and_kwargs_arg1() -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", 10, arg2="mixed")
    assert obj.arg1 == 10


def test_factory_with_args_and_kwargs_arg2() -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", 10, arg2="mixed")
    assert obj.arg2 == "mixed"


def test_factory_with_only_kwargs_type() -> None:
    """Test factory with only keyword arguments."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", arg1=20, arg2="kwargs_only")
    assert isinstance(obj, FakeClass)


def test_factory_with_only_kwargs_arg1() -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", arg1=20, arg2="kwargs_only")
    assert obj.arg1 == 20


def test_factory_with_only_kwargs_arg2() -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", arg1=20, arg2="kwargs_only")
    assert obj.arg2 == "kwargs_only"


def test_factory_with_default_parameter_type() -> None:
    """Test factory where default parameter is used."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", 30)
    assert isinstance(obj, FakeClass)


def test_factory_with_default_parameter_arg1() -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", 30)
    assert obj.arg1 == 30


def test_factory_with_default_parameter_arg2() -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", 30)
    assert obj.arg2 == "default"


def test_factory_with_classmethod_init_type() -> None:
    """Test factory with custom class method as initializer."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", _init_="create_with_defaults")
    assert isinstance(obj, FakeClass)


def test_factory_with_classmethod_init_arg1() -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", _init_="create_with_defaults")
    assert obj.arg1 == 42


def test_factory_with_classmethod_init_arg2() -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", _init_="create_with_defaults")
    assert obj.arg2 == "factory_method"


def test_factory_with_classmethod_init_and_args_type() -> None:
    """Test factory with custom class method and arguments."""
    obj = factory(
        "tests.unit.factory.test_resolve.FakeClass",
        "custom_value",
        _init_="create_with_custom_arg2",
    )
    assert isinstance(obj, FakeClass)


def test_factory_with_classmethod_init_and_args_arg1() -> None:
    obj = factory(
        "tests.unit.factory.test_resolve.FakeClass",
        "custom_value",
        _init_="create_with_custom_arg2",
    )
    assert obj.arg1 == 100


def test_factory_with_classmethod_init_and_args_arg2() -> None:
    obj = factory(
        "tests.unit.factory.test_resolve.FakeClass",
        "custom_value",
        _init_="create_with_custom_arg2",
    )
    assert obj.arg2 == "custom_value"


def test_factory_with_classmethod_init_and_kwargs_type() -> None:
    """Test factory with custom class method and keyword arguments."""
    obj = factory(
        "tests.unit.factory.test_resolve.FakeClass",
        _init_="create_with_custom_arg2",
        arg2="kwarg_value",
    )
    assert isinstance(obj, FakeClass)


def test_factory_with_classmethod_init_and_kwargs_arg1() -> None:
    obj = factory(
        "tests.unit.factory.test_resolve.FakeClass",
        _init_="create_with_custom_arg2",
        arg2="kwarg_value",
    )
    assert obj.arg1 == 100


def test_factory_with_classmethod_init_and_kwargs_arg2() -> None:
    obj = factory(
        "tests.unit.factory.test_resolve.FakeClass",
        _init_="create_with_custom_arg2",
        arg2="kwarg_value",
    )
    assert obj.arg2 == "kwarg_value"


def test_factory_with_static_method_init_type() -> None:
    """Test factory with static method as initializer."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", _init_="static_create")
    assert isinstance(obj, FakeClass)


def test_factory_with_static_method_init_arg1() -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", _init_="static_create")
    assert obj.arg1 == 999


def test_factory_with_static_method_init_arg2() -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", _init_="static_create")
    assert obj.arg2 == "static"


def test_factory_with_new_init_is_refused() -> None:
    with pytest.raises(TypeError, match=r"`_init_` must be '__init__' or the name of a public"):
        factory("tests.unit.factory.test_resolve.FakeClass", 50, "new_test", _init_="__new__")


def test_factory_function_target_type() -> None:
    """Test factory with a function as target."""
    obj = factory("tests.unit.factory.test_resolve.fake_function", 60)
    assert isinstance(obj, FakeClass)


def test_factory_function_target_arg1() -> None:
    obj = factory("tests.unit.factory.test_resolve.fake_function", 60)
    assert obj.arg1 == 60


def test_factory_function_target_arg2() -> None:
    obj = factory("tests.unit.factory.test_resolve.fake_function", 60)
    assert obj.arg2 == "function_default"


def test_factory_function_with_kwargs_arg1() -> None:
    """Test factory with a function and keyword arguments."""
    obj = factory("tests.unit.factory.test_resolve.fake_function", arg1=70, arg2="function_kwargs")
    assert obj.arg1 == 70


def test_factory_function_with_kwargs_arg2() -> None:
    obj = factory("tests.unit.factory.test_resolve.fake_function", arg1=70, arg2="function_kwargs")
    assert obj.arg2 == "function_kwargs"


def test_factory_builtin_type_value() -> None:
    """Test factory with built-in types."""
    obj = factory("builtins.int", "42")
    assert obj == 42


def test_factory_builtin_type_type() -> None:
    obj = factory("builtins.int", "42")
    assert isinstance(obj, int)


def test_factory_builtin_list_value() -> None:
    """Test factory with built-in list type."""
    obj = factory("builtins.list", [1, 2, 3])
    assert obj == [1, 2, 3]


def test_factory_builtin_list_type() -> None:
    obj = factory("builtins.list", [1, 2, 3])
    assert isinstance(obj, list)


@pytest.mark.parametrize(("arg1", "arg2"), [(1, "a"), (10, "z"), (-5, "test")])
def test_factory_parametrized_arg1(arg1: int, arg2: str) -> None:
    """Test factory with parametrized inputs."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", arg1=arg1, arg2=arg2)
    assert obj.arg1 == arg1


@pytest.mark.parametrize(("arg1", "arg2"), [(1, "a"), (10, "z"), (-5, "test")])
def test_factory_parametrized_arg2(arg1: int, arg2: str) -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", arg1=arg1, arg2=arg2)
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


def test_factory_object_init_constant_matches_literal_key_type() -> None:
    """Test that the OBJECT_INIT constant is wired to the actual
    ``_init_`` behavior of factory, i.e. that using the constant as a
    dict key has the same effect as the literal ``"_init_"`` string."""
    obj = factory(
        **{
            OBJECT_TARGET: "tests.unit.factory.test_resolve.FakeClass",
            OBJECT_INIT: "create_with_custom_arg2",
            "arg2": "custom_value",
        }
    )
    assert isinstance(obj, FakeClass)


def test_factory_object_init_constant_matches_literal_key_arg1() -> None:
    obj = factory(
        **{
            OBJECT_TARGET: "tests.unit.factory.test_resolve.FakeClass",
            OBJECT_INIT: "create_with_custom_arg2",
            "arg2": "custom_value",
        }
    )
    assert obj.arg1 == 100


def test_factory_object_init_constant_matches_literal_key_arg2() -> None:
    obj = factory(
        **{
            OBJECT_TARGET: "tests.unit.factory.test_resolve.FakeClass",
            OBJECT_INIT: "create_with_custom_arg2",
            "arg2": "custom_value",
        }
    )
    assert obj.arg2 == "custom_value"


def test_factory_invalid_init_method() -> None:
    """Test that factory raises error for invalid _init_ method."""
    with pytest.raises(TypeError, match=r"does not have a callable `invalid_method` attribute"):
        factory("tests.unit.factory.test_resolve.FakeClass", _init_="invalid_method")


def test_factory_empty_target() -> None:
    """Test that factory raises error for empty target string."""
    with pytest.raises((ImportError, ValueError)):
        factory("")


def test_factory_malformed_target() -> None:
    """Test that factory raises error for malformed target."""
    with pytest.raises(ImportError):
        factory("collections.")


def test_factory_with_custom_class_arg1() -> None:
    """Test factory with custom test class."""
    obj = factory("tests.unit.factory.test_resolve.FakeClass", arg1=123, arg2="test")
    assert obj.arg1 == 123


def test_factory_with_custom_class_arg2() -> None:
    obj = factory("tests.unit.factory.test_resolve.FakeClass", arg1=123, arg2="test")
    assert obj.arg2 == "test"


def test_factory_preserves_type_is_instance() -> None:
    """Test that factory preserves the correct type."""
    counter = factory("collections.Counter")
    assert type(counter).__name__ == "Counter"


def test_factory_preserves_type_type() -> None:
    counter = factory("collections.Counter")
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
    with pytest.raises(TypeError, match=r"a configuration needs the `_target_` key"):
        resolve_object({"name": "Rex"}, cls=Dog)


# --- A dict with the key ``_target_`` is always a configuration ---


def test_resolve_object_plain_dict_with_dict_subclass_cls_is_built_type() -> None:
    result = resolve_object({OBJECT_TARGET: "collections.OrderedDict", "a": 1}, cls=OrderedDict)
    assert isinstance(result, OrderedDict)


def test_resolve_object_plain_dict_with_dict_subclass_cls_is_built_value() -> None:
    result = resolve_object({OBJECT_TARGET: "collections.OrderedDict", "a": 1}, cls=OrderedDict)
    assert result == OrderedDict(a=1)


def test_resolve_object_configuration_with_dict_cls_is_built() -> None:
    assert resolve_object({OBJECT_TARGET: "collections.OrderedDict", "a": 1}, dict) == OrderedDict(
        a=1
    )


def test_resolve_object_dict_subclass_instance_without_target_is_passed_through() -> None:
    od = OrderedDict(a=1)
    assert resolve_object(od, cls=OrderedDict) is od


def test_resolve_object_counter_instance_without_target_is_passed_through() -> None:
    counter = Counter(a=1, b=2)
    assert resolve_object(counter, cls=Counter) is counter


def test_resolve_object_dict_without_target_is_passed_through_for_dict() -> None:
    config = {"a": 1}
    assert resolve_object(config, dict) is config


def test_resolve_object_dict_without_target_is_passed_through_for_mapping() -> None:
    config = {"a": 1}
    assert resolve_object(config, Mapping) is config


def test_resolve_object_dict_without_target_is_passed_through_by_default() -> None:
    config = {"a": 1}
    assert resolve_object(config) is config


def test_resolve_object_dict_without_target_for_another_cls_raises() -> None:
    with pytest.raises(TypeError, match=r"received: <class 'dict'>; a configuration needs"):
        resolve_object({"a": 1}, cls=OrderedDict)


def test_resolve_object_dict_subclass_instance_not_matching_cls_raises() -> None:
    with pytest.raises(TypeError, match=r"a configuration needs the `_target_` key"):
        resolve_object(Counter(a=1), cls=OrderedDict)


def test_resolve_object_non_dict_error_has_no_configuration_hint() -> None:
    with pytest.raises(TypeError, match=r"received: <class 'int'>\)"):
        resolve_object(1, cls=str)


def test_resolve_object_dict_subclass_with_target_key_is_a_configuration() -> None:
    assert resolve_object(Counter({OBJECT_TARGET: "collections.Counter", "a": 2})) == Counter(a=2)


def test_resolve_object_dict_subclass_with_target_key_is_not_passed_through() -> None:
    with pytest.raises(TypeError, match=r"does not build one"):
        resolve_object(Counter({OBJECT_TARGET: "collections.OrderedDict"}), cls=Counter)


def test_resolve_object_generic_cls_raises_the_isinstance_error() -> None:
    with pytest.raises(TypeError, match=r"isinstance\(\) argument 2 cannot be a parameterized"):
        resolve_object([1], cls=list[int])


def test_resolve_object_imports_the_target_once() -> None:
    with patch(
        "coola.factory.instantiation.importlib.import_module", wraps=importlib.import_module
    ) as m:
        resolve_object({OBJECT_TARGET: "collections.Counter"}, Counter)
    assert m.call_count == 1


def test_resolve_object_explicit_init_is_checked_like_the_default() -> None:
    with pytest.raises(TypeError, match="it was not called"):
        resolve_object({OBJECT_TARGET: f"{__name__}.Tracked", OBJECT_INIT: "__init__"}, cls=str)
    assert CALLS == []


def test_resolve_object_does_not_modify_the_configuration() -> None:
    config = {
        OBJECT_TARGET: "collections.Counter",
        OBJECT_INIT: "__init__",
        OBJECT_PREFIXES: ["collections"],
        "a": 1,
    }
    resolve_object(config, Counter)
    assert config == {
        OBJECT_TARGET: "collections.Counter",
        OBJECT_INIT: "__init__",
        OBJECT_PREFIXES: ["collections"],
        "a": 1,
    }


def test_factory_allowed_prefixes() -> None:
    assert factory("collections.Counter", [1, 1], _allowed_prefixes_=["collections"])


def test_factory_not_allowed_prefixes() -> None:
    with pytest.raises(ImportError, match="is not allowed"):
        factory("os.getcwd", _allowed_prefixes_=["collections"])


def test_factory_allowed_prefixes_violation_is_not_reported_as_missing() -> None:
    with pytest.raises(ImportError) as info:
        factory("os.getcwd", _allowed_prefixes_=["collections"])
    assert "does not exist" not in str(info.value)


def test_factory_allowed_prefixes_string_is_a_single_prefix() -> None:
    assert factory("collections.Counter", [1, 1], _allowed_prefixes_="collections")


def test_factory_import_error_inside_the_target_module_is_not_reported_as_missing() -> None:
    error = ModuleNotFoundError("No module named 'missing_dependency'", name="missing_dependency")
    with (
        patch("coola.factory.instantiation.importlib.import_module", side_effect=error),
        pytest.raises(ModuleNotFoundError, match="missing_dependency"),
    ):
        factory("some_package.SomeClass")


def test_factory_allowed_prefixes_not_forwarded_to_target() -> None:
    # The reserved key must be consumed, not passed to the constructor.
    assert factory("collections.Counter", [1, 1, 2], _allowed_prefixes_=["collections"]) == Counter(
        [1, 1, 2]
    )


def test_factory_allowed_prefixes_none() -> None:
    assert factory("collections.Counter", [1], _allowed_prefixes_=None) == Counter([1])


def test_resolve_object_allowed_prefixes_match() -> None:
    config = {OBJECT_TARGET: "collections.Counter", "a": 2}
    assert resolve_object(config, Counter, allowed_prefixes=["collections"]) == Counter(a=2)


def test_resolve_object_allowed_prefixes_no_match() -> None:
    with pytest.raises(ImportError, match="is not allowed"):
        resolve_object({OBJECT_TARGET: "collections.Counter"}, allowed_prefixes=["os"])


def test_resolve_object_allowed_prefixes_no_match_does_not_call_the_target() -> None:
    with pytest.raises(ImportError, match="is not allowed"):
        resolve_object({OBJECT_TARGET: "os.getcwd"}, allowed_prefixes=["collections"])


def test_resolve_object_allowed_prefixes_must_match_a_whole_module_name() -> None:
    with pytest.raises(ImportError, match="is not allowed"):
        resolve_object({OBJECT_TARGET: "collections.Counter"}, allowed_prefixes=["collection"])


def test_resolve_object_allowed_prefixes_overrides_the_prefixes_of_the_config() -> None:
    # A configuration cannot widen the allow-list given by the caller.
    config = {OBJECT_TARGET: "os.getcwd", "_allowed_prefixes_": ["os"]}
    with pytest.raises(ImportError, match="is not allowed"):
        resolve_object(config, allowed_prefixes=["collections"])


def test_resolve_object_allowed_prefixes_does_not_modify_the_config() -> None:
    config = {OBJECT_TARGET: "collections.Counter", "a": 1}
    resolve_object(config, allowed_prefixes=["collections"])
    assert config == {OBJECT_TARGET: "collections.Counter", "a": 1}


def test_resolve_object_prefixes_of_the_config_are_ignored_without_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "collections.Counter", OBJECT_PREFIXES: ["os"]}
    assert resolve_object(config) == Counter()


def test_resolve_object_prefixes_of_the_config_cannot_widen_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "os.getcwd", OBJECT_PREFIXES: ["os"]}
    with pytest.raises(ImportError, match="is not allowed"):
        resolve_object(config, allowed_prefixes=["collections"])


def test_resolve_object_allowed_prefixes_ignored_for_an_instance() -> None:
    counter = Counter([1])
    assert resolve_object(counter, Counter, allowed_prefixes=["os"]) is counter


def test_resolve_object_without_allowed_prefixes_allows_any_target() -> None:
    assert resolve_object({OBJECT_TARGET: "collections.Counter", "a": 1}) == Counter(a=1)


def test_resolve_object_tuple_of_classes_instance() -> None:
    assert resolve_object(1, (int, str)) == 1


def test_resolve_object_tuple_of_classes_config() -> None:
    assert resolve_object({OBJECT_TARGET: "builtins.str", "object": "a"}, (int, str)) == "a"


def test_resolve_object_tuple_of_classes_wrong_type() -> None:
    with pytest.raises(TypeError, match=r"not a int \| str instance"):
        resolve_object(1.5, (int, str))


def test_resolve_object_tuple_of_classes_with_dict_subclass() -> None:
    counter = Counter(a=1)
    assert resolve_object(counter, (int, Counter)) is counter


#####################################################
#     Tests for the check of the target (resolve)   #
#####################################################

CALLS: list[str] = []


class Tracked(Animal):
    def __init__(self) -> None:
        CALLS.append("Tracked")


def make_tracked() -> Tracked:
    CALLS.append("make_tracked")
    return Tracked()


# The missing return annotation is the point of this helper, so ANN201 is ignored on purpose.
def make_tracked_unannotated():  # noqa: ANN201
    CALLS.append("make_tracked_unannotated")
    return Tracked()


def make_unresolvable() -> Tracked:
    CALLS.append("make_unresolvable")
    return Tracked()


make_unresolvable.__annotations__["return"] = "UndefinedName"


def make_animal_or_none() -> Tracked | None:
    CALLS.append("make_animal_or_none")
    return Tracked()


def make_str_or_int() -> str | int:
    CALLS.append("make_str_or_int")
    return "a"


@pytest.fixture(autouse=True)
def _clear_calls() -> None:
    CALLS.clear()


def test_resolve_object_wrong_class_target_is_not_called() -> None:
    with pytest.raises(TypeError, match="it was not called"):
        resolve_object({OBJECT_TARGET: f"{__name__}.Tracked"}, cls=str)
    assert CALLS == []


def test_resolve_object_wrong_class_target_message() -> None:
    with pytest.raises(TypeError, match="Received object is not a str instance"):
        resolve_object({OBJECT_TARGET: f"{__name__}.Tracked"}, cls=str)


def test_resolve_object_wrong_return_hint_target_is_not_called() -> None:
    with pytest.raises(TypeError, match="it was not called"):
        resolve_object({OBJECT_TARGET: f"{__name__}.make_tracked"}, cls=str)
    assert CALLS == []


def test_resolve_object_subclass_target_is_called() -> None:
    resolve_object({OBJECT_TARGET: f"{__name__}.Tracked"}, cls=Animal)
    assert CALLS == ["Tracked"]


def test_resolve_object_return_hint_subclass_is_called() -> None:
    resolve_object({OBJECT_TARGET: f"{__name__}.make_tracked"}, cls=Animal)
    assert CALLS == ["make_tracked", "Tracked"]


def test_resolve_object_tuple_of_classes_accepts_a_matching_target() -> None:
    resolve_object({OBJECT_TARGET: f"{__name__}.Tracked"}, cls=(str, Animal))
    assert CALLS == ["Tracked"]


def test_resolve_object_tuple_of_classes_refuses_a_wrong_target() -> None:
    with pytest.raises(TypeError, match=r"str \| int"):
        resolve_object({OBJECT_TARGET: f"{__name__}.Tracked"}, cls=(str, int))


def test_resolve_object_unannotated_function_is_called_by_default() -> None:
    resolve_object({OBJECT_TARGET: f"{__name__}.make_tracked_unannotated"}, cls=Animal)
    assert CALLS == ["make_tracked_unannotated", "Tracked"]


def test_resolve_object_unannotated_function_wrong_result_raises_after_the_call() -> None:
    with pytest.raises(TypeError, match="Received object is not a str instance"):
        resolve_object({OBJECT_TARGET: f"{__name__}.make_tracked_unannotated"}, cls=str)
    assert CALLS == ["make_tracked_unannotated", "Tracked"]


def test_resolve_object_strict_unannotated_function_is_not_called() -> None:
    with pytest.raises(TypeError, match="no return type hint"):
        resolve_object(
            {OBJECT_TARGET: f"{__name__}.make_tracked_unannotated"}, cls=Animal, strict=True
        )
    assert CALLS == []


def test_resolve_object_strict_unresolvable_hint_is_not_called() -> None:
    with pytest.raises(TypeError, match="no return type hint"):
        resolve_object({OBJECT_TARGET: f"{__name__}.make_unresolvable"}, cls=Animal, strict=True)
    assert CALLS == []


def test_resolve_object_strict_builtin_function_is_not_called() -> None:
    with pytest.raises(TypeError, match="no return type hint"):
        resolve_object({OBJECT_TARGET: "os.getpid"}, cls=int, strict=True)


def test_resolve_object_strict_class_target_is_called() -> None:
    resolve_object({OBJECT_TARGET: f"{__name__}.Tracked"}, cls=Animal, strict=True)
    assert CALLS == ["Tracked"]


def test_resolve_object_strict_annotated_function_is_called() -> None:
    resolve_object({OBJECT_TARGET: f"{__name__}.make_tracked"}, cls=Animal, strict=True)
    assert CALLS == ["make_tracked", "Tracked"]


def test_resolve_object_alternative_constructor_is_not_checked() -> None:
    resolve_object(
        {OBJECT_TARGET: "collections.OrderedDict", OBJECT_INIT: "fromkeys", "iterable": [1]},
        cls=OrderedDict,
    )


def test_resolve_object_default_cls_does_not_check_the_target() -> None:
    resolve_object({OBJECT_TARGET: f"{__name__}.make_tracked_unannotated"}, strict=True)
    assert CALLS == ["make_tracked_unannotated", "Tracked"]


def test_resolve_object_target_outside_allowed_prefixes_still_raises_import_error() -> None:
    with pytest.raises(ImportError, match="allowed"):
        resolve_object({OBJECT_TARGET: f"{__name__}.Tracked"}, cls=Animal, allowed_prefixes=["os"])


def test_resolve_object_nested_target_is_not_resolved() -> None:
    """Test that a nested configuration is passed to the target as a
    ``dict``, as documented."""
    config = {OBJECT_TARGET: "collections.OrderedDict", "a": {OBJECT_TARGET: "builtins.int"}}
    assert resolve_object(config) == OrderedDict(a={OBJECT_TARGET: "builtins.int"})


def test_resolve_object_nested_configuration_is_refused_with_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "collections.OrderedDict", "a": {OBJECT_TARGET: "os.getcwd"}}
    with pytest.raises(ValueError, match="An argument is a configuration"):
        resolve_object(config, allowed_prefixes=["collections"])


def test_resolve_object_nested_configuration_is_refused_before_the_target_is_called() -> None:
    CALLS.clear()
    config = {OBJECT_TARGET: f"{__name__}.Tracked", "a": {OBJECT_TARGET: "os.getcwd"}}
    with pytest.raises(ValueError, match="An argument is a configuration"):
        resolve_object(config, allowed_prefixes=[__name__])
    assert CALLS == []


def test_factory_nested_configuration_is_refused_with_allowed_prefixes() -> None:
    with pytest.raises(ValueError, match="An argument is a configuration"):
        factory(
            "collections.OrderedDict",
            a={OBJECT_TARGET: "os.getcwd"},
            _allowed_prefixes_=["collections"],
        )


def test_factory_nested_configuration_in_args_is_refused_with_allowed_prefixes() -> None:
    with pytest.raises(ValueError, match="An argument is a configuration"):
        factory("builtins.list", {OBJECT_TARGET: "os.getcwd"}, _allowed_prefixes_=["builtins"])


def test_factory_nested_configuration_deeper_than_the_top_level_is_not_scanned() -> None:
    assert factory(
        "builtins.list", [{OBJECT_TARGET: "os.getcwd"}], _allowed_prefixes_=["builtins"]
    ) == [{OBJECT_TARGET: "os.getcwd"}]


def test_factory_nested_configuration_is_not_refused_with_empty_dict_values() -> None:
    assert factory(
        "collections.OrderedDict", a={}, _allowed_prefixes_=["collections"]
    ) == OrderedDict(a={})


def test_factory_target_receives_an_argument_named_obj() -> None:
    assert factory("collections.OrderedDict", obj=1) == OrderedDict(obj=1)


def test_factory_target_receives_an_argument_named_cls() -> None:
    assert factory("collections.OrderedDict", cls=1) == OrderedDict(cls=1)


def test_factory_class_method_receives_an_argument_named_cls() -> None:
    assert factory(f"{__name__}.KwargsClass", cls=1, obj=2).kwargs == {"cls": 1, "obj": 2}


def test_resolve_object_target_receives_an_argument_named_cls() -> None:
    config = {OBJECT_TARGET: "collections.OrderedDict", "cls": 1, "obj": 2}
    assert resolve_object(config) == OrderedDict(cls=1, obj=2)


def test_factory_nested_target_is_not_resolved() -> None:
    assert factory("collections.OrderedDict", a={OBJECT_TARGET: "builtins.int"}) == OrderedDict(
        a={OBJECT_TARGET: "builtins.int"}
    )


@pytest.mark.parametrize("target", ["non_existing_module.SomeClass", "collections.NotACounter"])
def test_resolve_object_non_existing_target_with_cls(target: str) -> None:
    with pytest.raises(ImportError, match=r"The target object does not exist: "):
        resolve_object({OBJECT_TARGET: target}, cls=int)


class KwargsClass:
    def __init__(self, **kwargs: int) -> None:
        self.kwargs = kwargs


def test_factory_nested_configuration_of_resolve_loader_is_refused() -> None:
    with pytest.raises(ValueError, match="An argument is a configuration"):
        factory(
            "coola.io.resolve_loader",
            loader={OBJECT_TARGET: "os.getcwd"},
            _allowed_prefixes_=["coola.io"],
        )


def test_resolve_object_init_new_is_refused() -> None:
    config = {OBJECT_TARGET: "tests.unit.factory.test_resolve.FakeClass", OBJECT_INIT: "__new__"}
    with pytest.raises(TypeError, match=r"`_init_` must be '__init__' or the name of a public"):
        resolve_object(config)


def test_resolve_object_union_hint_without_subclass_is_not_called() -> None:
    with pytest.raises(TypeError, match="it was not called"):
        resolve_object({OBJECT_TARGET: f"{__name__}.make_str_or_int"}, cls=list)
    assert CALLS == []


def test_resolve_object_union_hint_with_subclass_is_called() -> None:
    resolve_object({OBJECT_TARGET: f"{__name__}.make_animal_or_none"}, cls=Animal)
    assert CALLS == ["make_animal_or_none", "Tracked"]


def test_resolve_object_strict_union_hint_with_classes_is_called() -> None:
    resolve_object({OBJECT_TARGET: f"{__name__}.make_animal_or_none"}, cls=Animal, strict=True)
    assert CALLS == ["make_animal_or_none", "Tracked"]
