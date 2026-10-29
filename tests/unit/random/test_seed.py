from __future__ import annotations

import pytest

from coola.random import validate_seed


@pytest.mark.parametrize("seed", [0, 42, 2**63 - 1])
def test_validate_seed_valid(seed: int) -> None:
    validate_seed(seed)


@pytest.mark.parametrize("seed", [-1, 2**63, 2**70])
def test_validate_seed_out_of_range(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be in the range"):
        validate_seed(seed)


@pytest.mark.parametrize("seed", [1.5, "1", True, None])
def test_validate_seed_not_int(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an int"):
        validate_seed(seed)  # ty: ignore[invalid-argument-type]


def test_validate_seed_out_of_range_message() -> None:
    with pytest.raises(
        ValueError, match=r"^seed must be in the range \[0, 9223372036854775808\), got -1$"
    ):
        validate_seed(-1)
