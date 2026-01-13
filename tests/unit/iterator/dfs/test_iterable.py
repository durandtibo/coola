from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import objects_are_equal
from coola.iterator.dfs import DefaultIterator, IterableIterator, IteratorRegistry

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def registry() -> IteratorRegistry:
    return IteratorRegistry({object: DefaultIterator()})


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
def test_iterable_iterator_iterate(data: Any, expected: Any, registry: IteratorRegistry) -> None:
    iterator = IterableIterator()
    registry.register(list, iterator)
    assert objects_are_equal(
        list(iterator.iterate(data, registry=registry)),
        expected,
    )


def test_iterable_iterator_iterate_set(registry: IteratorRegistry) -> None:
    result = list(IterableIterator().iterate({1, 2, 3}, registry=registry))
    assert len(result)
    assert objects_are_equal(set(result), {1, 2, 3})


def test_iterable_iterator_iterate_custom_generator(registry: IteratorRegistry) -> None:
    def gen() -> Generator[int, None, None]:
        yield 1
        yield 2
        yield 3

    assert objects_are_equal(
        list(IterableIterator().iterate(gen(), registry=registry)),
        [1, 2, 3],
    )
