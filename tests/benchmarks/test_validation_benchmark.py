r"""Benchmarks for the validators.

Run them explicitly with:

    inv test.benchmark

The validators run in constructors and in the entry points of
``hash_object``, ``round_values``... so the success path must stay as
cheap as a plain ``isinstance`` or comparison. The mean time of a call
on the success path, measured on an Apple M1 before the redesign of
``coola.validation``, is the reference: no row must be slower than 1.1
times its reference.

| Benchmark        | Reference |
| ---------------- | --------- |
| ``int``          | 67 ns     |
| ``isinstance``   | 66 ns     |
| ``number``       | 163 ns    |
| ``finite``       | 72 ns     |
| ``gt``           | 85 ns     |
| ``ge``           | 142 ns    |
| ``range``        | 256 ns    |
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coola.validation import (
    validate_finite,
    validate_ge,
    validate_gt,
    validate_int,
    validate_isinstance,
    validate_number,
    validate_range,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


def test_validate_int(benchmark: BenchmarkFixture) -> None:
    benchmark(validate_int, 1)


def test_validate_isinstance(benchmark: BenchmarkFixture) -> None:
    benchmark(validate_isinstance, 1.0, float)


def test_validate_number(benchmark: BenchmarkFixture) -> None:
    benchmark(validate_number, 1)


def test_validate_finite(benchmark: BenchmarkFixture) -> None:
    benchmark(validate_finite, 1.0)


def test_validate_gt(benchmark: BenchmarkFixture) -> None:
    benchmark(validate_gt, 1, 0)


def test_validate_ge(benchmark: BenchmarkFixture) -> None:
    benchmark(validate_ge, 1, 0)


def test_validate_range(benchmark: BenchmarkFixture) -> None:
    benchmark(validate_range, 1, 0, 2)
