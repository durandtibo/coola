from __future__ import annotations

from typing import Any

import pytest

from coola import objects_are_equal
from coola.iterator.bfs import DefaultChildFinder
from tests.unit.iterator.bfs.helpers import DEFAULT_FIND_CHILDREN_SAMPLES

########################################
#     Tests for DefaultChildFinder     #
########################################


def test_default_child_finder_repr() -> None:
    assert repr(DefaultChildFinder()) == "DefaultChildFinder()"


def test_default_child_finder_str() -> None:
    assert str(DefaultChildFinder()) == "DefaultChildFinder()"


@pytest.mark.parametrize(("data", "expected"), DEFAULT_FIND_CHILDREN_SAMPLES)
def test_default_child_finder_find_children(data: Any, expected: Any) -> None:
    assert objects_are_equal(list(DefaultChildFinder().find_children(data)), expected)
