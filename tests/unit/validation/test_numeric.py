from __future__ import annotations

from typing import Any

import pytest

from coola.validation import validate_finite


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
