from __future__ import annotations

import pytest

from coola.iterator import DefaultChildFinder

########################################
#     Tests for DefaultChildFinder     #
########################################


def test_default_child_finder_repr() -> None:
    assert repr(DefaultChildFinder()) == "DefaultChildFinder()"


@pytest.mark.parametrize("data", [42, "abc", b"abc", None, [1, 2], {"a": 1}, (), set()])
def test_default_child_finder_find_children_is_none(data: object) -> None:
    assert DefaultChildFinder().find_children(data) is None


def test_default_child_finder_find_items_is_none() -> None:
    assert DefaultChildFinder().find_items(42) is None
