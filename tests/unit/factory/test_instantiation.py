from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod
from collections import Counter

import pytest

from coola.factory import import_object, instantiate_object


class Fake:
    """Fake class to tests some functions."""

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

    @classmethod
    def class_method_with_arg(cls, arg2: str) -> Fake:
        return cls(arg1=333, arg2=arg2)

    @classmethod
    def class_method_wrong_type(cls) -> str:
        return "not a Fake instance"


def fake_func(arg1: int, arg2: str = "abc") -> Fake:
    """Fake function to tests some functions."""
    return Fake(arg1=arg1, arg2=arg2)


####################################
#     Tests for import_object      #
####################################


def test_import_object_class() -> None:
    assert import_object("collections.Counter") == Counter


def test_import_object_function() -> None:
    assert import_object("math.isclose") == math.isclose


def test_import_object() -> None:
    assert import_object("math.pi") == math.pi


def test_import_object_package() -> None:
    assert import_object("math") is math


def test_import_object_incorrect_invalid_qualified_name() -> None:
    with pytest.raises(ImportError, match=r"Module 'collections' has no attribute ''"):
        import_object("collections.")


def test_import_object_incorrect_object_does_not_exist() -> None:
    with pytest.raises(ImportError, match=r"Module 'collections' has no attribute 'NotACounter'"):
        import_object("collections.NotACounter")


def test_import_object_incorrect_package_does_not_exist() -> None:
    with pytest.raises(ImportError, match=r"No module named 'missing_package_bjbskfs'"):
        import_object("missing_package_bjbskfs.my_object")


def test_import_object_incorrect_invalid_package() -> None:
    with pytest.raises(ImportError, match=r"No module named 'missing_package_bjbskfs'"):
        import_object("missing_package_bjbskfs")


def test_import_object_incorrect_type() -> None:
    with pytest.raises(TypeError, match=r"`object_path` is not a string"):
        import_object(1)


########################################
#     Tests for instantiate_object     #
########################################


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_class_args(arg1: int, arg2: str) -> None:
    obj = instantiate_object(Fake, arg1, arg2)
    assert isinstance(obj, Fake)
    assert obj.arg1 == arg1
    assert obj.arg2 == arg2


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_class_kwargs(arg1: int, arg2: str) -> None:
    obj = instantiate_object(Fake, arg1=arg1, arg2=arg2)
    assert isinstance(obj, Fake)
    assert obj.arg1 == arg1
    assert obj.arg2 == arg2


def test_instantiate_object_class_init_classmethod() -> None:
    obj = instantiate_object(Fake, _init_="class_method")
    assert isinstance(obj, Fake)
    assert obj.arg1 == 35
    assert obj.arg2 == "bac"


def test_instantiate_object_class_init_classmethod_with_arg() -> None:
    obj = instantiate_object(
        Fake,
        "meow",
        _init_="class_method_with_arg",
    )
    assert isinstance(obj, Fake)
    assert obj.arg1 == 333
    assert obj.arg2 == "meow"


def test_instantiate_object_class_init_classmethod_with_kwarg() -> None:
    obj = instantiate_object(
        Fake,
        _init_="class_method_with_arg",
        arg2="meow",
    )
    assert isinstance(obj, Fake)
    assert obj.arg1 == 333
    assert obj.arg2 == "meow"


def test_instantiate_object_class_init_new() -> None:
    assert isinstance(instantiate_object(Fake, _init_="__new__"), Fake)


def test_instantiate_object_class_init_static_method() -> None:
    obj = instantiate_object(Fake, _init_="static_method")
    assert isinstance(obj, Fake)
    assert obj.arg1 == 1
    assert obj.arg2 == "qwerty"


def test_instantiate_object_init_not_exist() -> None:
    with pytest.raises(TypeError, match=r".* does not have `incorrect_init_method` attribute"):
        # Should fail because the init function does not exist.
        instantiate_object(Fake, _init_="incorrect_init_method")


def test_instantiate_object_init_regular_method() -> None:
    with pytest.raises(TypeError, match=r"missing 1 required positional argument: 'self'"):
        # Should fail because self is not defined.
        instantiate_object(Fake, _init_="method")


def test_instantiate_object_init_wrong_return_type() -> None:
    with pytest.raises(
        TypeError,
        match=r"`class_method_wrong_type` of .* did not return an instance of .*",
    ):
        # Should fail because the classmethod does not return a Fake instance.
        instantiate_object(Fake, _init_="class_method_wrong_type")


def test_instantiate_object_init_not_method() -> None:
    Fake.not_a_method = None
    with pytest.raises(TypeError, match=r"`not_a_method` attribute of .* is not callable"):
        # Should fail because the attribute not_a_method is not a method.
        instantiate_object(Fake, _init_="not_a_method")


def test_instantiate_object_abstract_class() -> None:
    class FakeAbstractClass(ABC):
        @abstractmethod
        def my_method(self) -> None:
            """Abstract method."""

    with pytest.raises(
        TypeError,
        match=r"Cannot instantiate the class .* because it is an abstract class.",
    ):
        instantiate_object(FakeAbstractClass, 42)


def test_instantiate_object_function() -> None:
    obj = instantiate_object(fake_func, 42)
    assert isinstance(obj, Fake)
    assert obj.arg1 == 42
    assert obj.arg2 == "abc"


def test_instantiate_object_function_init() -> None:
    obj = instantiate_object(fake_func, 42, _init_="my_init_function")
    assert isinstance(obj, Fake)
    assert obj.arg1 == 42
    assert obj.arg2 == "abc"


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_function_args(arg1: int, arg2: str) -> None:
    obj = instantiate_object(fake_func, arg1, arg2)
    assert isinstance(obj, Fake)
    assert obj.arg1 == arg1
    assert obj.arg2 == arg2


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_function_kwargs(arg1: int, arg2: str) -> None:
    obj = instantiate_object(fake_func, arg1=arg1, arg2=arg2)
    assert isinstance(obj, Fake)
    assert obj.arg1 == arg1
    assert obj.arg2 == arg2


def test_instantiate_object_incorrect_type() -> None:
    with pytest.raises(
        TypeError, match=r"Incorrect type: .*. The valid types are class and callable"
    ):
        instantiate_object(Fake(12))


def test_instantiate_object_builtin_function() -> None:
    assert instantiate_object(len, [1, 2, 3]) == 3


def test_instantiate_object_partial() -> None:
    obj = instantiate_object(functools.partial(fake_func, arg2="partial"), 42)
    assert isinstance(obj, Fake)
    assert obj.arg1 == 42
    assert obj.arg2 == "partial"


def test_instantiate_object_callable_instance() -> None:
    class CallableFactory:
        def __call__(self, arg1: int) -> Fake:
            return Fake(arg1=arg1, arg2="callable")

    obj = instantiate_object(CallableFactory(), 7)
    assert isinstance(obj, Fake)
    assert obj.arg1 == 7
    assert obj.arg2 == "callable"
