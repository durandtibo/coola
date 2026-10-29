from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from coola.equality import Tolerance

###############################
#     Tests for Tolerance     #
###############################


def test_tolerance_default() -> None:
    assert Tolerance() == Tolerance(rtol=0.0, atol=0.0)


def test_tolerance_repr() -> None:
    assert repr(Tolerance(rtol=0.1, atol=0.2)) == "Tolerance(rtol=0.1, atol=0.2)"


def test_tolerance_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Tolerance().rtol = 1.0  # ty: ignore[invalid-assignment]


def test_tolerance_is_hashable() -> None:
    assert hash(Tolerance(rtol=0.1)) == hash(Tolerance(rtol=0.1))


@pytest.mark.parametrize("name", ["rtol", "atol"])
def test_tolerance_negative(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        Tolerance(**{name: -1.0})


@pytest.mark.parametrize("name", ["rtol", "atol"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_tolerance_not_finite(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        Tolerance(**{name: value})


def test_tolerance_numpy_default() -> None:
    assert Tolerance.numpy_default() == Tolerance(rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("name", ["rtol", "atol"])
@pytest.mark.parametrize("value", [True, False, "a", None])
def test_tolerance_wrong_type(name: str, value: object) -> None:
    kwargs: dict[str, Any] = {name: value}
    with pytest.raises(TypeError, match=f"{name} must be a real number"):
        Tolerance(**kwargs)
