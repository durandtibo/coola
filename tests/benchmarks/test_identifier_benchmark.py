r"""Benchmarks for ``coola.identifier``.

Run them explicitly with:

    inv test.benchmark

They exist to compare the performance of the implementations of
``coola.identifier`` over time. They only use the public generators and
extractors that exist in every design of the package, so that they run
unchanged on any of them. The generators are called 10,000 times per
round, so a result divided by 10,000 is the cost of one call. Do not use
the internals of the package in this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coola.identifier import (
    SnowflakeIdGenerator,
    extract_snowflake_timestamp_ms,
    extract_ulid_timestamp_ms,
    extract_uuid7_timestamp_ms,
    generate_checksummed_id,
    generate_object_id,
    generate_stable_content_id,
    generate_stable_uuid5,
    generate_ulid,
    generate_uuid7,
    verify_checksummed_id,
)

# The generator takes its worker_id in the constructor.
generate_snowflake_id = SnowflakeIdGenerator(worker_id=0).generate

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture

CALLS = 10_000

SMALL = {"source": "cats.txt", "page": 1}
NESTED = {
    "id": 1,
    "tags": ["a", "b", "c"],
    "values": [float(i) for i in range(100)],
    "children": [{"name": f"n{i}", "pos": (i, i + 1)} for i in range(100)],
}


def repeat(func: Callable[..., Any], *args: Any) -> list[Any]:
    r"""Call ``func(*args)`` ``CALLS`` times."""
    return [func(*args) for _ in range(CALLS)]


#######################################
#     Time-ordered, current time      #
#######################################


def test_generate_ulid_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(repeat, generate_ulid)) == CALLS


def test_generate_uuid7_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(repeat, generate_uuid7)) == CALLS


def test_generate_snowflake_id_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(repeat, generate_snowflake_id)) == CALLS


def test_generate_object_id_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(repeat, generate_object_id)) == CALLS


#######################
#     Extractors      #
#######################


def test_extract_ulid_timestamp_ms_benchmark(benchmark: BenchmarkFixture) -> None:
    value = generate_ulid()
    assert len(benchmark(repeat, extract_ulid_timestamp_ms, value)) == CALLS


def test_extract_uuid7_timestamp_ms_benchmark(benchmark: BenchmarkFixture) -> None:
    value = generate_uuid7()
    assert len(benchmark(repeat, extract_uuid7_timestamp_ms, value)) == CALLS


def test_extract_snowflake_timestamp_ms_benchmark(benchmark: BenchmarkFixture) -> None:
    value = generate_snowflake_id()
    assert len(benchmark(repeat, extract_snowflake_timestamp_ms, value)) == CALLS


#########################
#     Checksummed       #
#########################


def test_generate_checksummed_id_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(repeat, generate_checksummed_id)) == CALLS


def test_verify_checksummed_id_benchmark(benchmark: BenchmarkFixture) -> None:
    value = generate_checksummed_id()
    assert all(benchmark(repeat, verify_checksummed_id, value))


#############################
#     Stable identifiers    #
#############################


def test_generate_stable_content_id_small_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(generate_stable_content_id, SMALL)) == 64


def test_generate_stable_uuid5_small_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(generate_stable_uuid5, SMALL)) == 36


def test_generate_stable_content_id_nested_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(generate_stable_content_id, NESTED)) == 64


def test_generate_stable_uuid5_nested_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(generate_stable_uuid5, NESTED)) == 36
