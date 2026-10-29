from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

import pytest

from coola.factory import import_object, instantiate_object

if TYPE_CHECKING:
    from pathlib import Path


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


class FakeAbstractClass(ABC):
    @abstractmethod
    def my_method(self) -> None:
        """Abstract method."""


def fake_func(arg1: int, arg2: str = "abc") -> Fake:
    """Fake function to tests some functions."""
    return Fake(arg1=arg1, arg2=arg2)


###################################
#     Tests for import_object     #
###################################


def test_import_object_class() -> None:
    assert import_object("collections.Counter") == Counter


def test_import_object_function() -> None:
    assert import_object("math.isclose") == math.isclose


def test_import_object() -> None:
    assert import_object("math.pi") == math.pi


def test_import_object_package() -> None:
    assert import_object("math") is math


def test_import_object_incorrect_invalid_qualified_name() -> None:
    with pytest.raises(ImportError, match=r"The target object does not exist: collections\."):
        import_object("collections.")


def test_import_object_incorrect_object_does_not_exist() -> None:
    with pytest.raises(
        ImportError, match=r"The target object does not exist: collections.NotACounter"
    ):
        import_object("collections.NotACounter")


def test_import_object_incorrect_package_does_not_exist() -> None:
    with pytest.raises(
        ImportError, match=r"The target object does not exist: missing_package_bjbskfs.my_object"
    ):
        import_object("missing_package_bjbskfs.my_object")


def test_import_object_incorrect_invalid_package() -> None:
    with pytest.raises(
        ImportError, match=r"The target object does not exist: missing_package_bjbskfs"
    ):
        import_object("missing_package_bjbskfs")


def test_import_object_incorrect_package_does_not_exist_name() -> None:
    with pytest.raises(ImportError, match="The target object does not exist") as info:
        import_object("missing_package_bjbskfs.my_object")
    assert info.value.name == "missing_package_bjbskfs"


def test_import_object_incorrect_object_does_not_exist_name() -> None:
    with pytest.raises(ImportError, match="The target object does not exist") as info:
        import_object("collections.NotACounter")
    assert info.value.name == "collections.NotACounter"


def test_import_object_missing_dependency_keeps_its_own_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "module_with_missing_dependency.py").write_text(
        "import missing_dependency_bjbskfs\n\nTarget = 1\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match=r"No module named 'missing_dependency_bjbskfs'"):
        import_object("module_with_missing_dependency.Target")


def test_import_object_incorrect_type() -> None:
    with pytest.raises(TypeError, match=r"`object_path` is not a string"):
        import_object(1)


########################################
#     Tests for instantiate_object     #
########################################


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_class_args_type(arg1: int, arg2: str) -> None:
    obj = instantiate_object(Fake, arg1, arg2)
    assert isinstance(obj, Fake)


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_class_args_arg1(arg1: int, arg2: str) -> None:
    obj = instantiate_object(Fake, arg1, arg2)
    assert obj.arg1 == arg1


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_class_args_arg2(arg1: int, arg2: str) -> None:
    obj = instantiate_object(Fake, arg1, arg2)
    assert obj.arg2 == arg2


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_class_kwargs_type(arg1: int, arg2: str) -> None:
    obj = instantiate_object(Fake, arg1=arg1, arg2=arg2)
    assert isinstance(obj, Fake)


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_class_kwargs_arg1(arg1: int, arg2: str) -> None:
    obj = instantiate_object(Fake, arg1=arg1, arg2=arg2)
    assert obj.arg1 == arg1


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_class_kwargs_arg2(arg1: int, arg2: str) -> None:
    obj = instantiate_object(Fake, arg1=arg1, arg2=arg2)
    assert obj.arg2 == arg2


def test_instantiate_object_class_init_classmethod_type() -> None:
    obj = instantiate_object(Fake, _init_="class_method")
    assert isinstance(obj, Fake)


def test_instantiate_object_class_init_classmethod_arg1() -> None:
    obj = instantiate_object(Fake, _init_="class_method")
    assert obj.arg1 == 35


def test_instantiate_object_class_init_classmethod_arg2() -> None:
    obj = instantiate_object(Fake, _init_="class_method")
    assert obj.arg2 == "bac"


def test_instantiate_object_class_init_classmethod_with_arg_type() -> None:
    obj = instantiate_object(
        Fake,
        "meow",
        _init_="class_method_with_arg",
    )
    assert isinstance(obj, Fake)


def test_instantiate_object_class_init_classmethod_with_arg_arg1() -> None:
    obj = instantiate_object(
        Fake,
        "meow",
        _init_="class_method_with_arg",
    )
    assert obj.arg1 == 333


def test_instantiate_object_class_init_classmethod_with_arg_arg2() -> None:
    obj = instantiate_object(
        Fake,
        "meow",
        _init_="class_method_with_arg",
    )
    assert obj.arg2 == "meow"


def test_instantiate_object_class_init_classmethod_with_kwarg_type() -> None:
    obj = instantiate_object(
        Fake,
        _init_="class_method_with_arg",
        arg2="meow",
    )
    assert isinstance(obj, Fake)


def test_instantiate_object_class_init_classmethod_with_kwarg_arg1() -> None:
    obj = instantiate_object(
        Fake,
        _init_="class_method_with_arg",
        arg2="meow",
    )
    assert obj.arg1 == 333


def test_instantiate_object_class_init_classmethod_with_kwarg_arg2() -> None:
    obj = instantiate_object(
        Fake,
        _init_="class_method_with_arg",
        arg2="meow",
    )
    assert obj.arg2 == "meow"


def test_instantiate_object_class_init_static_method_type() -> None:
    obj = instantiate_object(Fake, _init_="static_method")
    assert isinstance(obj, Fake)


def test_instantiate_object_class_init_static_method_arg1() -> None:
    obj = instantiate_object(Fake, _init_="static_method")
    assert obj.arg1 == 1


def test_instantiate_object_class_init_static_method_arg2() -> None:
    obj = instantiate_object(Fake, _init_="static_method")
    assert obj.arg2 == "qwerty"


def test_instantiate_object_init_not_exist() -> None:
    with pytest.raises(
        TypeError, match=r".* does not have a callable `incorrect_init_method` attribute"
    ):
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


def test_instantiate_object_init_not_method(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Fake, "not_a_method", None, raising=False)
    with pytest.raises(TypeError, match=r".* does not have a callable `not_a_method` attribute"):
        # Should fail because the attribute not_a_method is not a method.
        instantiate_object(Fake, _init_="not_a_method")


def test_instantiate_object_abstract_class() -> None:
    with pytest.raises(
        TypeError,
        match=r"Cannot instantiate the class .* because it is an abstract class.",
    ):
        instantiate_object(FakeAbstractClass, 42)


def test_instantiate_object_function_type() -> None:
    obj = instantiate_object(fake_func, 42)
    assert isinstance(obj, Fake)


def test_instantiate_object_function_arg1() -> None:
    obj = instantiate_object(fake_func, 42)
    assert obj.arg1 == 42


def test_instantiate_object_function_arg2() -> None:
    obj = instantiate_object(fake_func, 42)
    assert obj.arg2 == "abc"


def test_instantiate_object_function_init_type() -> None:
    obj = instantiate_object(fake_func, 42, _init_="my_init_function")
    assert isinstance(obj, Fake)


def test_instantiate_object_function_init_arg1() -> None:
    obj = instantiate_object(fake_func, 42, _init_="my_init_function")
    assert obj.arg1 == 42


def test_instantiate_object_function_init_arg2() -> None:
    obj = instantiate_object(fake_func, 42, _init_="my_init_function")
    assert obj.arg2 == "abc"


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_function_args_type(arg1: int, arg2: str) -> None:
    obj = instantiate_object(fake_func, arg1, arg2)
    assert isinstance(obj, Fake)


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_function_args_arg1(arg1: int, arg2: str) -> None:
    obj = instantiate_object(fake_func, arg1, arg2)
    assert obj.arg1 == arg1


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_function_args_arg2(arg1: int, arg2: str) -> None:
    obj = instantiate_object(fake_func, arg1, arg2)
    assert obj.arg2 == arg2


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_function_kwargs_type(arg1: int, arg2: str) -> None:
    obj = instantiate_object(fake_func, arg1=arg1, arg2=arg2)
    assert isinstance(obj, Fake)


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_function_kwargs_arg1(arg1: int, arg2: str) -> None:
    obj = instantiate_object(fake_func, arg1=arg1, arg2=arg2)
    assert obj.arg1 == arg1


@pytest.mark.parametrize("arg1", [-1, 1])
@pytest.mark.parametrize("arg2", ["arg2", "cba"])
def test_instantiate_object_function_kwargs_arg2(arg1: int, arg2: str) -> None:
    obj = instantiate_object(fake_func, arg1=arg1, arg2=arg2)
    assert obj.arg2 == arg2


def test_instantiate_object_incorrect_type() -> None:
    with pytest.raises(
        TypeError, match=r"Incorrect type: .*. The valid types are class and callable"
    ):
        instantiate_object(Fake(12))  # ty: ignore[invalid-argument-type]


def test_instantiate_object_builtin_function() -> None:
    assert instantiate_object(len, [1, 2, 3]) == 3


def test_instantiate_object_partial_type() -> None:
    obj = instantiate_object(functools.partial(fake_func, arg2="partial"), 42)
    assert isinstance(obj, Fake)


def test_instantiate_object_partial_arg1() -> None:
    obj = instantiate_object(functools.partial(fake_func, arg2="partial"), 42)
    assert obj.arg1 == 42


def test_instantiate_object_partial_arg2() -> None:
    obj = instantiate_object(functools.partial(fake_func, arg2="partial"), 42)
    assert obj.arg2 == "partial"


def test_instantiate_object_callable_instance_type() -> None:
    class CallableFactory:
        def __call__(self, arg1: int) -> Fake:
            return Fake(arg1=arg1, arg2="callable")

    obj = instantiate_object(CallableFactory(), 7)
    assert isinstance(obj, Fake)


def test_instantiate_object_callable_instance_arg1() -> None:
    class CallableFactory:
        def __call__(self, arg1: int) -> Fake:
            return Fake(arg1=arg1, arg2="callable")

    obj = instantiate_object(CallableFactory(), 7)
    assert obj.arg1 == 7


def test_instantiate_object_callable_instance_arg2() -> None:
    class CallableFactory:
        def __call__(self, arg1: int) -> Fake:
            return Fake(arg1=arg1, arg2="callable")

    obj = instantiate_object(CallableFactory(), 7)
    assert obj.arg2 == "callable"


def test_import_object_allowed_prefixes_class() -> None:
    assert import_object("collections.Counter", allowed_prefixes=["collections"]) is Counter


def test_import_object_allowed_prefixes_module() -> None:
    assert import_object("collections", allowed_prefixes=["collections"])


def test_import_object_not_allowed_prefixes() -> None:
    with pytest.raises(ImportError, match="not allowed"):
        import_object("os.system", allowed_prefixes=["collections"])


def test_import_object_not_allowed_prefixes_similar_prefix() -> None:
    with pytest.raises(ImportError, match="not allowed"):
        import_object("collectionsx.Counter", allowed_prefixes=["collections"])


def test_import_object_allowed_prefixes_none_allows_all() -> None:
    assert import_object("math.pi", allowed_prefixes=None) == math.pi


def test_import_object_allowed_prefixes_empty_rejects_all() -> None:
    with pytest.raises(ImportError, match="not allowed"):
        import_object("collections.Counter", allowed_prefixes=[])


def test_import_object_allowed_prefixes_multiple() -> None:
    assert import_object("math.pi", allowed_prefixes=["collections", "math"]) == math.pi


def test_import_object_allowed_prefixes_submodule() -> None:
    assert import_object("os.path.join", allowed_prefixes=["os.path"]) is not None


def test_import_object_not_allowed_prefixes_submodule() -> None:
    with pytest.raises(ImportError, match="not allowed"):
        import_object("os.getcwd", allowed_prefixes=["os.path"])


def test_import_object_allowed_prefixes_checked_before_import() -> None:
    # A non-existent module outside the allow-list must report "not allowed".
    with pytest.raises(ImportError, match="not allowed"):
        import_object("nonexistent_pkg_xyz.Foo", allowed_prefixes=["collections"])


def test_import_object_allowed_prefixes_non_string() -> None:
    with pytest.raises(TypeError, match="not a string"):
        import_object(42, allowed_prefixes=["collections"])


@pytest.mark.parametrize("object_path", ["", ".", ".Counter", "collections..Counter"])
def test_import_object_empty_part_raises_import_error(object_path: str) -> None:
    with pytest.raises(ImportError, match="Invalid object path"):
        import_object(object_path)


def test_import_object_allowed_prefixes_string_is_a_single_prefix() -> None:
    assert import_object("collections.Counter", allowed_prefixes="collections") is Counter


def test_import_object_allowed_prefixes_string_rejects_other_modules() -> None:
    with pytest.raises(ImportError, match="not allowed"):
        import_object("os.system", allowed_prefixes="collections")


def test_instantiate_object_class_receives_an_argument_named_obj() -> None:
    assert instantiate_object(dict, obj=1) == {"obj": 1}


def test_instantiate_object_class_receives_an_argument_named_cls() -> None:
    assert instantiate_object(dict, cls=1) == {"cls": 1}


def test_instantiate_object_function_receives_an_argument_named_obj() -> None:
    assert instantiate_object(functools.partial(dict, obj=1), cls=2) == {"obj": 1, "cls": 2}


@pytest.mark.parametrize("init", ["__new__", "__class__", "__subclasses__", "_private", "_"])
def test_instantiate_object_init_starting_with_underscore_is_refused(init: str) -> None:
    with pytest.raises(TypeError, match=r"`_init_` must be '__init__' or the name of a public"):
        instantiate_object(Fake, _init_=init)


def test_instantiate_object_init_not_a_string_is_refused() -> None:
    with pytest.raises(TypeError, match=r"`_init_` must be '__init__' or the name of a public"):
        instantiate_object(Fake, _init_=5)  # ty: ignore[invalid-argument-type]


def test_instantiate_object_init_underscore_is_refused_before_the_attribute_is_read() -> None:
    with pytest.raises(TypeError, match=r"received: '__new__'"):
        instantiate_object(Counter, _init_="__new__")


def test_instantiate_object_init_is_ignored_for_a_function() -> None:
    assert instantiate_object(math.isclose, 1.0, 1.0, _init_="__new__") is True


def test_instantiate_object_init_is_ignored_for_a_callable_instance() -> None:
    assert instantiate_object(functools.partial(int, "1"), _init_="_private") == 1
