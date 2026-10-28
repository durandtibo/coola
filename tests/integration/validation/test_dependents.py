r"""Test that the public arguments that call a validator raise the
message of the validator, with the real validators."""

from __future__ import annotations

import pytest

from coola.equality import EqualityConfig
from coola.hashing import hash_object
from coola.nested import merge_mappings
from coola.random import validate_seed
from coola.rounding import round_values


def test_hash_object_length_message() -> None:
    with pytest.raises(ValueError, match=r"^length must be an even number, got 3$"):
        hash_object([1], length=3)


def test_hash_object_length_type_message() -> None:
    with pytest.raises(TypeError, match=r"^length must be an int, got str$"):
        hash_object([1], length="3")  # ty: ignore[invalid-argument-type]


def test_round_values_decimals_type_message() -> None:
    with pytest.raises(TypeError, match=r"^decimals must be an int, got bool$"):
        round_values(1.5, decimals=True)


def test_round_values_decimals_value_message() -> None:
    with pytest.raises(ValueError, match=r"^decimals must be greater than or equal to 0, got -1$"):
        round_values(1.5, decimals=-1)


def test_merge_mappings_on_duplicate_message() -> None:
    with pytest.raises(
        ValueError,
        match=r"^on_duplicate must be one of \('raise', 'first', 'last', 'suffix'\), got 'x'$",
    ):
        merge_mappings([{"a": 1}], on_duplicate="x")  # ty: ignore[invalid-argument-type]


def test_validate_seed_message() -> None:
    with pytest.raises(TypeError, match=r"^seed must be an int, got float$"):
        validate_seed(1.5)  # ty: ignore[invalid-argument-type]


def test_equality_config_max_depth_message() -> None:
    with pytest.raises(ValueError, match=r"^max_depth must be greater than 0, got 0$"):
        EqualityConfig(max_depth=0)
