from __future__ import annotations

from typing import Any

import pytest

from coola import objects_are_equal
from coola.iterator.dfs import DefaultIterator, IteratorRegistry

DEFAULT_SAMPLES = [
    pytest.param("abc", ["abc"], id="str"),
    pytest.param(42, [42], id="int"),
    pytest.param({"key": "value"}, [{"key": "value"}], id="dict"),
    pytest.param({}, [{}], id="empty dict"),
    pytest.param([1, 2, 3], [[1, 2, 3]], id="list"),
    pytest.param([], [[]], id="empty list"),
    pytest.param((1, 2, 3, 4), [(1, 2, 3, 4)], id="tuple"),
    pytest.param((), [()], id="empty tuple"),
    pytest.param({1, 2, 3}, [{1, 2, 3}], id="set"),
    pytest.param(set(), [set()], id="empty set"),
]


#####################################
#     Tests for DefaultIterator     #
#####################################


def test_default_iterator_repr() -> None:
    assert repr(DefaultIterator()) == "DefaultIterator()"


def test_default_iterator_str() -> None:
    assert str(DefaultIterator()) == "DefaultIterator()"


@pytest.mark.parametrize(("data", "expected"), DEFAULT_SAMPLES)
def test_default_iterator_iterate(data: Any, expected: Any) -> None:
    assert objects_are_equal(
        list(DefaultIterator().iterate(data, registry=IteratorRegistry())), expected
    )
