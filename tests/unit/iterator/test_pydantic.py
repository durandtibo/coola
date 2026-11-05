from __future__ import annotations

from typing import Any
from unittest.mock import patch

from coola.iterator import dfs_items, dfs_iterate
from coola.iterator.pydantic import PydanticChildFinder, get_default_child_finders
from coola.testing.marks import pydantic_available
from coola.utils.imports import is_pydantic_available

if is_pydantic_available():
    from pydantic import BaseModel, ConfigDict, computed_field

    class Point(BaseModel):
        x: Any
        y: Any = 0.0

    class Extra(BaseModel):
        model_config = ConfigDict(extra="allow")

        a: Any

    class Computed(BaseModel):
        x: float

        @computed_field  # type: ignore[prop-decorator]
        @property
        def double(self) -> float:
            return self.x * 2

    class Nested(BaseModel):
        items: list
        point: Point


########################################
#     Tests for PydanticChildFinder    #
########################################


def test_pydantic_child_finder_repr() -> None:
    assert repr(PydanticChildFinder()) == "PydanticChildFinder()"


@pydantic_available
def test_pydantic_child_finder_find_children() -> None:
    assert list(PydanticChildFinder().find_children(Point(x=1, y=2))) == [1, 2]


@pydantic_available
def test_pydantic_child_finder_find_children_extra_fields_come_last() -> None:
    assert list(PydanticChildFinder().find_children(Extra(z=3, a=1))) == [1, 3]


@pydantic_available
def test_pydantic_child_finder_find_children_computed_field_is_not_a_child() -> None:
    assert list(PydanticChildFinder().find_children(Computed(x=1.0))) == [1.0]


@pydantic_available
def test_pydantic_child_finder_find_items() -> None:
    assert list(PydanticChildFinder().find_items(Point(x=1, y=2))) == [("x", 1), ("y", 2)]


@pydantic_available
def test_pydantic_child_finder_find_items_extra_fields() -> None:
    assert list(PydanticChildFinder().find_items(Extra(z=3, a=1))) == [("a", 1), ("z", 3)]


###############################################
#     Tests for get_default_child_finders     #
###############################################


@pydantic_available
def test_get_default_child_finders_keys() -> None:
    assert list(get_default_child_finders()) == [BaseModel]


def test_get_default_child_finders_without_pydantic() -> None:
    with patch("coola.iterator.pydantic.is_pydantic_available", lambda: False):
        assert get_default_child_finders() == {}


@pydantic_available
def test_dfs_iterate_enters_a_model() -> None:
    assert list(dfs_iterate(Nested(items=[1, 2], point=Point(x=3)))) == [1, 2, 3, 0.0]


@pydantic_available
def test_dfs_items_enters_a_model() -> None:
    assert list(dfs_items(Point(x=1, y=[2]))) == [(("x",), 1), (("y", 0), 2)]
