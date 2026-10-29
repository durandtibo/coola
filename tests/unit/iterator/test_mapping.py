from __future__ import annotations

from collections import Counter, OrderedDict
from types import MappingProxyType
from typing import Any

import pytest

from coola.iterator import MappingChildFinder

########################################
#     Tests for MappingChildFinder     #
########################################


def test_mapping_child_finder_repr() -> None:
    assert repr(MappingChildFinder()) == "MappingChildFinder()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param({"a": 1, "b": 2}, [1, 2], id="dict"),
        pytest.param({}, [], id="empty dict"),
        pytest.param({"a": 1}, [1], id="single element"),
        pytest.param({"a": 1, "b": 1}, [1, 1], id="duplicate values"),
        pytest.param({"a": {"b": 1, "c": 2}, "d": 3}, [{"b": 1, "c": 2}, 3], id="does not recurse"),
        pytest.param({"b": 1, "a": 2}, [1, 2], id="insertion order"),
        pytest.param(OrderedDict([("b", 1), ("a", 2)]), [1, 2], id="ordered dict"),
        pytest.param(Counter({"a": 3}), [3], id="counter"),
        pytest.param(MappingProxyType({"a": 1}), [1], id="mapping proxy"),
    ],
)
def test_mapping_child_finder_find_children(data: Any, expected: list) -> None:
    assert list(MappingChildFinder().find_children(data)) == expected


def test_mapping_child_finder_find_children_does_not_modify_the_data() -> None:
    data = {"a": [1]}
    list(MappingChildFinder().find_children(data))
    assert data == {"a": [1]}


def test_mapping_child_finder_find_items() -> None:
    assert list(MappingChildFinder().find_items({"a": 1, "b": 2})) == [("a", 1), ("b", 2)]


def test_mapping_child_finder_find_items_keeps_non_str_keys() -> None:
    assert list(MappingChildFinder().find_items({1: "a", (2, 3): "b"})) == [(1, "a"), ((2, 3), "b")]


def test_mapping_child_finder_find_items_empty() -> None:
    assert list(MappingChildFinder().find_items({})) == []


def test_mapping_child_finder_find_items_mapping_proxy() -> None:
    assert list(MappingChildFinder().find_items(MappingProxyType({"a": 1}))) == [("a", 1)]
