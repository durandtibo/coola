from __future__ import annotations

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
