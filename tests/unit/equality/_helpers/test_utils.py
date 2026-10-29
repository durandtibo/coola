from __future__ import annotations

from typing import Any

import pytest

from coola.equality._helpers import comparison_key, is_exact_scalar_sequence
from coola.equality.config import EqualityConfig

###############################################
#     Tests for is_exact_scalar_sequence      #
###############################################


@pytest.mark.parametrize("seq", [[1, "a", None, True], (b"a", 2), [], ()])
def test_is_exact_scalar_sequence_true(seq: Any) -> None:
    assert is_exact_scalar_sequence(seq)


@pytest.mark.parametrize("seq", [[1.0], [1, [2]], {1, 2}, "abc", range(3)])
def test_is_exact_scalar_sequence_false(seq: Any) -> None:
    assert not is_exact_scalar_sequence(seq)


def test_is_exact_scalar_sequence_subclass_of_list() -> None:
    class MyList(list):
        pass

    assert not is_exact_scalar_sequence(MyList([1]))


###########################################
#     Tests for comparison_key            #
###########################################


def test_comparison_key_frozenset_with_nan() -> None:
    config = EqualityConfig(equal_nan=True)
    assert comparison_key(frozenset({float("nan")}), config) == comparison_key(
        frozenset({float("nan")}), config
    )


def test_comparison_key_frozenset_without_equal_nan() -> None:
    config = EqualityConfig()
    assert comparison_key(frozenset({float("nan")}), config) != comparison_key(
        frozenset({float("nan")}), config
    )


def test_comparison_key_keeps_the_type() -> None:
    assert comparison_key(1, EqualityConfig()) != comparison_key(True, EqualityConfig())
