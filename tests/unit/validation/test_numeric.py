from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available
from coola.validation import validate_finite, validate_number

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = None


@pytest.mark.parametrize("value", [0, 1, -1.5, 1e300])
def test_validate_finite_valid(value: float) -> None:
    validate_finite(value)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_validate_finite_invalid(value: float) -> None:
    with pytest.raises(ValueError, match="value must be finite, got"):
        validate_finite(value)


def test_validate_finite_custom_name() -> None:
    with pytest.raises(ValueError, match="atol must be finite, got inf"):
        validate_finite(float("inf"), name="atol")


def test_validate_finite_huge_int() -> None:
    validate_finite(10**400)


@pytest.mark.parametrize("value", [None, "a"])
def test_validate_finite_not_a_number_raises(value: Any) -> None:
    with pytest.raises(TypeError, match="real number"):
        validate_finite(value)


@pytest.mark.parametrize("value", [0, 1, -1.5, 1e300, float("inf")])
def test_validate_number_valid(value: float) -> None:
    validate_number(value)


@pytest.mark.parametrize("value", [True, False, None, "a", [1], 1j])
def test_validate_number_invalid(value: Any) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        validate_number(value)


def test_validate_number_custom_name() -> None:
    with pytest.raises(TypeError, match="foo must be a real number"):
        validate_number("a", name="foo")


@pytest.mark.parametrize("value", [True, Decimal("1.5")])
def test_validate_finite_not_a_real_number_raises(value: Any) -> None:
    with pytest.raises(TypeError, match="value must be a real number"):
        validate_finite(value)


@numpy_available
def test_validate_finite_array_raises_a_named_type_error() -> None:
    with pytest.raises(TypeError, match=r"^value must be a real number, got ndarray$"):
        validate_finite(np.array([1.0, 2.0]))


def test_validate_finite_message_shows_the_value_with_repr() -> None:
    with pytest.raises(ValueError, match=r"^lr must be finite, got inf$"):
        validate_finite(float("inf"), name="lr")


def test_validate_number_message_names_the_type() -> None:
    with pytest.raises(TypeError, match=r"^rtol must be a real number, got NoneType$"):
        validate_number(None, name="rtol")
