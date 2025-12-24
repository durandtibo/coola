from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola import objects_are_equal
from coola.iterator.dfs import IterableIterator, IteratorRegistry

if TYPE_CHECKING:
    from collections.abc import Generator

######################################
#     Tests for IterableIterator     #
######################################


def test_iterable_iterator_repr() -> None:
    assert repr(IterableIterator()) == "IterableIterator()"


def test_iterable_iterator_str() -> None:
    assert str(IterableIterator()) == "IterableIterator()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param([5, 3, 8, 1, 9, 2], [5, 3, 8, 1, 9, 2], id="list"),
        pytest.param([], [], id="empty list"),
        pytest.param((1, 2, 3), [1, 2, 3], id="tuple"),
        pytest.param((), [], id="empty tuple"),
        pytest.param("hello", ["h", "e", "l", "l", "o"], id="string"),
        pytest.param("", [], id="empty string"),
        pytest.param([[1, 2, 3], [4, 5, 6]], [1, 2, 3, 4, 5, 6], id="nested"),
        pytest.param(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [1, 2, 3, 4, 5, 6, 7, 8], id="deeply nested"
        ),
        pytest.param([1, "a", 2.5, None, True], [1, "a", 2.5, None, True], id="mixed types"),
        pytest.param(range(3), [0, 1, 2], id="range"),
    ],
)
def test_iterable_iterator_iterate(data: Any, expected: Any) -> None:
    iterator = IterableIterator()
    assert objects_are_equal(
        list(iterator.iterate(data, registry=IteratorRegistry({list: iterator}))), expected
    )


def test_iterable_iterator_iterate_set() -> None:
    result = list(IterableIterator().iterate({1, 2, 3}, registry=IteratorRegistry()))
    assert len(result)
    assert objects_are_equal(set(result), {1, 2, 3})


def test_iterable_iterator_iterate_custom_generator() -> None:
    def gen() -> Generator[int, None, None]:
        yield 1
        yield 2
        yield 3

    assert objects_are_equal(
        list(IterableIterator().iterate(gen(), registry=IteratorRegistry())), [1, 2, 3]
    )
