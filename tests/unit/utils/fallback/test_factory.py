from __future__ import annotations

import pytest

from coola.utils.fallback.factory import make_fake_class, make_fake_function


def _raise_error() -> None:
    msg = "dependency is missing"
    raise RuntimeError(msg)


def test_make_fake_class_returns_type() -> None:
    assert isinstance(make_fake_class(_raise_error), type)


def test_make_fake_class_name() -> None:
    assert make_fake_class(_raise_error).__name__ == "FakeClass"


def test_make_fake_class_instantiation_raises_error() -> None:
    fake_class = make_fake_class(_raise_error)
    with pytest.raises(RuntimeError, match=r"dependency is missing"):
        fake_class()


def test_make_fake_class_instantiation_raises_error_with_args() -> None:
    fake_class = make_fake_class(_raise_error)
    with pytest.raises(RuntimeError, match=r"dependency is missing"):
        fake_class(1, 2, key="value")


def test_make_fake_class_uses_given_callable() -> None:
    calls = []

    def raise_error() -> None:
        calls.append(1)
        msg = "custom error"
        raise RuntimeError(msg)

    fake_class = make_fake_class(raise_error)
    with pytest.raises(RuntimeError, match=r"custom error"):
        fake_class()
    assert calls == [1]


def test_make_fake_class_independent_instances() -> None:
    fake_class1 = make_fake_class(_raise_error)
    fake_class2 = make_fake_class(_raise_error)
    assert fake_class1 is not fake_class2


def test_make_fake_function_returns_callable() -> None:
    assert callable(make_fake_function(_raise_error))


def test_make_fake_function_call_raises_error() -> None:
    fake_function = make_fake_function(_raise_error)
    with pytest.raises(RuntimeError, match=r"dependency is missing"):
        fake_function()


def test_make_fake_function_call_raises_error_with_args() -> None:
    fake_function = make_fake_function(_raise_error)
    with pytest.raises(RuntimeError, match=r"dependency is missing"):
        fake_function(1, 2, key="value")


def test_make_fake_function_uses_given_callable() -> None:
    calls = []

    def raise_error() -> None:
        calls.append(1)
        msg = "custom error"
        raise RuntimeError(msg)

    fake_function = make_fake_function(raise_error)
    with pytest.raises(RuntimeError, match=r"custom error"):
        fake_function()
    assert calls == [1]
