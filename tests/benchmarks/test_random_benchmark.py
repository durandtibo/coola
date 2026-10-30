r"""Benchmarks for ``coola.random``.

Run them explicitly with:

    inv test.benchmark

They only use the public functions of ``coola.random``, so that they
run unchanged on any design of the package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coola.random import (
    get_random_seed,
    get_rng_state,
    get_torch_generator,
    random_seed,
    set_rng_state,
)
from coola.testing.marks import torch_available

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


def seeded() -> None:
    with random_seed(42):
        pass


def test_random_seed_benchmark(benchmark: BenchmarkFixture) -> None:
    benchmark(seeded)


def test_get_random_seed_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(get_random_seed, 42) == get_random_seed(42)


def test_get_rng_state_benchmark(benchmark: BenchmarkFixture) -> None:
    assert isinstance(benchmark(get_rng_state), dict)


def test_set_rng_state_benchmark(benchmark: BenchmarkFixture) -> None:
    state = get_rng_state()
    benchmark(set_rng_state, state)


@torch_available
def test_get_torch_generator_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(get_torch_generator, 42).initial_seed() == 42
