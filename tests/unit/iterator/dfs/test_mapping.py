from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pytest

from coola import objects_are_equal
from coola.iterator.dfs import IterableIterator, IteratorRegistry, MappingIterator

#####################################
#     Tests for MappingIterator     #
#####################################


def test_mapping_iterator_repr() -> None:
    assert repr(MappingIterator()) == "MappingIterator()"


def test_mapping_iterator_str() -> None:
    assert str(MappingIterator()) == "MappingIterator()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param({"a": 1, "b": 2}, [1, 2], id="dict"),
        pytest.param({"a": {"b": 1, "c": 2}, "d": 3}, [1, 2, 3], id="nested dict"),
        pytest.param({}, [], id="empty dict"),
        pytest.param({"a": {}}, [], id="empty nested dict"),
        pytest.param({"x": [1, 2], "y": [3, 4]}, [1, 2, 3, 4], id="nested dict list"),
        pytest.param(
            {"a": {"b": [1, 2], "c": 3}, "d": 4}, [1, 2, 3, 4], id="nested dict mixed types"
        ),
        pytest.param(OrderedDict({"a": 1, "b": 2}), [1, 2], id="ordered dict"),
    ],
)
def test_mapping_iterator_iterate(data: Any, expected: Any) -> None:
    iterator = MappingIterator()
    assert objects_are_equal(
        list(
            iterator.iterate(
                data, registry=IteratorRegistry({list: IterableIterator(), dict: iterator})
            )
        ),
        expected,
    )
