from __future__ import annotations

from coola.random.interface import random_seed
from tests.helpers import draw_python_random

#################################
#     Tests for random_seed     #
#################################


def _draw_with_nested_contexts() -> tuple[float, float, float]:
    with random_seed(42):
        value1 = draw_python_random()

        with random_seed(123):
            value2 = draw_python_random()

        # After inner context exits, outer seed should still be active
        value3 = draw_python_random()
    return value1, value2, value3


def test_random_seed_nested_contexts_use_different_seeds() -> None:
    """Test that the inner context has a different seed."""
    value1, value2, _ = _draw_with_nested_contexts()
    assert value1 != value2


def test_random_seed_nested_contexts_reproduce_the_outer_sequence() -> None:
    value1, _, _ = _draw_with_nested_contexts()
    assert value1 == _draw_with_nested_contexts()[0]


def test_random_seed_nested_contexts_reproduce_the_inner_sequence() -> None:
    _, value2, _ = _draw_with_nested_contexts()
    assert value2 == _draw_with_nested_contexts()[1]


def test_random_seed_nested_contexts_reproduce_the_outer_sequence_after_the_inner_context() -> None:
    _, _, value3 = _draw_with_nested_contexts()
    assert value3 == _draw_with_nested_contexts()[2]
