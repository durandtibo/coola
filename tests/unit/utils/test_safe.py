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
