r"""Property-based tests for the guarantees of ``summarize`` and of the
random seeds."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from hypothesis import assume, given, strategies as st

from coola.random import NumpyRandomManager, random_seed
from coola.summary import summarize
from tests.helpers import draw_numpy_rand, draw_python_random
from tests.properties.strategies import values

##################
#     Summary    #
##################


nested_ints = st.recursive(
    st.integers(), lambda children: st.lists(children, max_size=8), max_leaves=40
)


def max_lines(depth: int, max_items: int) -> int:
    r"""Return the maximum number of lines of a summary: a node has a
    header, at most ``max_items`` children and a line for the hidden
    children."""
    if depth == 0:
        return 1
    return 2 + max_items * max_lines(depth - 1, max_items)


@given(data=values, max_depth=st.integers(0, 4))
def test_summarize_returns_a_str(data: Any, max_depth: int) -> None:
    assert isinstance(summarize(data, max_depth=max_depth), str)


@given(data=values.filter(lambda value: value != ""), max_depth=st.integers(0, 4))
def test_summarize_returns_a_non_empty_string(data: Any, max_depth: int) -> None:
    assert summarize(data, max_depth=max_depth)


@given(data=nested_ints, max_depth=st.integers(0, 4), max_items=st.integers(0, 4))
def test_summarize_output_is_bounded(data: Any, max_depth: int, max_items: int) -> None:
    summary = summarize(data, max_depth=max_depth, max_items=max_items)
    assert len(summary.splitlines()) <= max_lines(max_depth, max_items)


@given(data=nested_ints, max_depth=st.integers(0, 4))
def test_summarize_hides_nothing_without_limits(data: Any, max_depth: int) -> None:
    summary = summarize(data, max_depth=max_depth, max_items=None, max_characters=None)
    assert "more)" not in summary


@given(data=values, max_depth=st.integers(0, 4))
def test_summarize_is_deterministic(data: Any, max_depth: int) -> None:
    assert summarize(data, max_depth=max_depth) == summarize(data, max_depth=max_depth)


@given(data=values)
def test_summarize_does_not_modify_the_data(data: Any) -> None:
    before = repr(data)
    summarize(data, max_depth=3)
    assert repr(data) == before


@given(data=st.lists(st.integers(), min_size=1, max_size=20))
def test_summarize_shows_the_length_of_a_list(data: list[int]) -> None:
    assert f"length={len(data)}" in summarize(data)


##################
#     Random     #
##################


@given(seed=st.integers(0, 2**32 - 1))
def test_random_seed_gives_the_same_numbers(seed: int) -> None:
    with random_seed(seed):
        first = draw_python_random()
    with random_seed(seed):
        second = draw_python_random()
    assert first == second


@given(seed=st.integers(0, 2**32 - 1))
def test_random_seed_restores_the_state(seed: int) -> None:
    state = random.getstate()
    with random_seed(seed):
        draw_python_random()
    assert random.getstate() == state


@given(
    seed=st.integers(0, 2**63 - 1),
    other=st.integers(0, 2**63 - 1),
)
def test_numpy_manager_different_seeds_give_different_numbers(seed: int, other: int) -> None:
    assume(seed != other)
    manager = NumpyRandomManager()
    with manager.seed(seed):
        first = draw_numpy_rand(4)
    with manager.seed(other):
        second = draw_numpy_rand(4)
    assert not np.array_equal(first, second)
