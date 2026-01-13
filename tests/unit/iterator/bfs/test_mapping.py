from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pytest

from coola.equality import objects_are_equal
from coola.iterator.bfs import MappingChildFinder

########################################
#     Tests for MappingChildFinder     #
########################################


def test_mapping_child_finder_repr() -> None:
    assert repr(MappingChildFinder()) == "MappingChildFinder()"


def test_mapping_child_finder_str() -> None:
    assert str(MappingChildFinder()) == "MappingChildFinder()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param({"a": 1, "b": 2}, [1, 2], id="dict"),
        pytest.param({"a": {"b": 1, "c": 2}, "d": 3}, [{"b": 1, "c": 2}, 3], id="nested dict"),
        pytest.param({}, [], id="empty dict"),
        pytest.param({"a": {}}, [{}], id="empty nested dict"),
        pytest.param({"x": [1, 2], "y": [3, 4]}, [[1, 2], [3, 4]], id="nested dict list"),
        pytest.param(OrderedDict({"a": 1, "b": 2}), [1, 2], id="ordered dict"),
    ],
)
def test_mapping_child_finder_find_children(data: Any, expected: Any) -> None:
    iterator = MappingChildFinder()
    assert objects_are_equal(list(iterator.find_children(data)), expected)
