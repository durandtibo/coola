r"""Property-based tests for the guarantees of ``summarize`` and of the
random seeds."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from hypothesis import assume, given, strategies as st

from coola.random import NumpyRandomManager, random_seed
from coola.summary import summarize
from tests.properties.strategies import values

##################
#     Summary    #
##################


@given(data=values, max_depth=st.integers(0, 4))
def test_summarize_returns_a_non_empty_string(data: Any, max_depth: int) -> None:
    assert summarize(data, max_depth=max_depth)


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
        first = random.random()  # noqa: S311
    with random_seed(seed):
        second = random.random()  # noqa: S311
    assert first == second


@given(seed=st.integers(0, 2**32 - 1))
def test_random_seed_restores_the_state(seed: int) -> None:
    state = random.getstate()
    with random_seed(seed):
        random.random()  # noqa: S311
    assert random.getstate() == state


@given(
    seed=st.integers(0, 2**63 - 1),
    other=st.integers(0, 2**63 - 1),
)
def test_numpy_manager_different_seeds_give_different_numbers(seed: int, other: int) -> None:
    assume(seed != other)
    manager = NumpyRandomManager()
    with manager.seed(seed):
        first = np.random.rand(4)  # noqa: NPY002
    with manager.seed(other):
        second = np.random.rand(4)  # noqa: NPY002
    assert not np.array_equal(first, second)
