from __future__ import annotations

from coola.utils.introspection import get_fully_qualified_name


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
