from __future__ import annotations

import pytest

from coola.utils.safe import safe_repr, safe_str


class BadStr:
    def __str__(self) -> str:
        msg = "broken"
        raise RuntimeError(msg)


class BadRepr:
    def __repr__(self) -> str:
        msg = "broken"
        raise RuntimeError(msg)


@pytest.mark.parametrize("obj", [42, "abc", [1, 2], None])
def test_safe_str(obj: object) -> None:
    assert safe_str(obj) == str(obj)


def test_safe_str_failing_str() -> None:
    assert safe_str(BadStr()).startswith("<BadStr object at 0x")


def test_safe_str_big_int() -> None:
    assert safe_str(10**5000).startswith("<int object at 0x")


@pytest.mark.parametrize("obj", [42, "abc", [1, 2], None])
def test_safe_repr(obj: object) -> None:
    assert safe_repr(obj) == repr(obj)


def test_safe_repr_failing_repr() -> None:
    assert safe_repr(BadRepr()).startswith("<BadRepr object at 0x")


# ``__str__`` is set from a dict because a ``__str__`` that returns an int is rejected by ruff.
NonStrStr = type("NonStrStr", (), {"__str__": lambda _: 1})


class NonStrRepr:
    def __repr__(self) -> str:
        return 1  # ty: ignore[invalid-return-type]


class Interrupt:
    def __str__(self) -> str:
        msg = "interrupted"
        raise KeyboardInterrupt(msg)

    def __repr__(self) -> str:
        msg = "interrupted"
        raise KeyboardInterrupt(msg)


class Outer:
    class Inner:
        def __str__(self) -> str:
            msg = "broken"
            raise ValueError(msg)

        def __repr__(self) -> str:
            msg = "broken"
            raise ValueError(msg)


@pytest.mark.parametrize("obj", ["", "é日本語", b"abc", 1.5, {"a": 1}])
def test_safe_str_returns_str(obj: object) -> None:
    assert safe_str(obj) == str(obj)


@pytest.mark.parametrize("obj", ["", "é日本語", b"abc", 1.5, {"a": 1}])
def test_safe_repr_returns_repr(obj: object) -> None:
    assert safe_repr(obj) == repr(obj)


def test_safe_str_non_str_result() -> None:
    assert safe_str(NonStrStr()).startswith("<NonStrStr object at 0x")


def test_safe_repr_non_str_result() -> None:
    assert safe_repr(NonStrRepr()).startswith("<NonStrRepr object at 0x")


def test_safe_str_placeholder_contains_id() -> None:
    obj = BadStr()
    assert safe_str(obj) == f"<BadStr object at {id(obj):#x}>"


def test_safe_repr_placeholder_contains_id() -> None:
    obj = BadRepr()
    assert safe_repr(obj) == f"<BadRepr object at {id(obj):#x}>"


def test_safe_str_uses_qualified_name() -> None:
    assert safe_str(Outer.Inner()).startswith("<Outer.Inner object at 0x")


def test_safe_repr_uses_qualified_name() -> None:
    assert safe_repr(Outer.Inner()).startswith("<Outer.Inner object at 0x")


def test_safe_repr_big_int() -> None:
    assert safe_repr(10**5000).startswith("<int object at 0x")


def test_safe_str_does_not_swallow_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt, match="interrupted"):
        safe_str(Interrupt())


def test_safe_repr_does_not_swallow_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt, match="interrupted"):
        safe_repr(Interrupt())


def test_safe_str_object_with_only_bad_repr_is_not_affected() -> None:
    assert safe_str(BadRepr()).startswith("<BadRepr object at 0x")


def test_safe_repr_object_with_only_bad_str_is_not_affected() -> None:
    assert "BadStr object at 0x" in safe_repr(BadStr())
