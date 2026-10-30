r"""Benchmarks for ``coola.display``.

Run them explicitly with:

    inv test.benchmark

They only use the public functions of ``coola.display``, so that they
run unchanged on any design of the package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coola.display import repr_pydantic_model, str_pydantic_model
from coola.testing.marks import pydantic_available
from coola.utils.imports import is_pydantic_available

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

if is_pydantic_available():
    from pydantic import BaseModel

    class Item(BaseModel):
        name: str
        values: list[int]
        tags: dict[str, str]

    MODEL = Item(name="a", values=list(range(100)), tags={f"k{i}": "v" for i in range(50)})


@pydantic_available
def test_repr_pydantic_model_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(repr_pydantic_model, MODEL).startswith("Item(")


@pydantic_available
def test_str_pydantic_model_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(str_pydantic_model, MODEL).startswith("Item(")
