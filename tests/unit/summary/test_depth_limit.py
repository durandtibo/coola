from __future__ import annotations

from collections import deque

import pytest

from coola.summary import summarize

#########################################################
#     Tests for the collections at the depth limit      #
#########################################################


class Bad:
    r"""An object whose ``__repr__`` and ``__str__`` raise."""

    def __repr__(self) -> str:
        msg = "broken __repr__"
        raise ZeroDivisionError(msg)

    __str__ = __repr__

    def __hash__(self) -> int:
        return 0


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([Bad()], id="list"),
        pytest.param((Bad(),), id="tuple"),
        pytest.param(deque([Bad()]), id="deque"),
        pytest.param({"a": Bad()}, id="dict"),
        pytest.param({Bad()}, id="set"),
        pytest.param(frozenset({Bad()}), id="frozenset"),
    ],
)
def test_summarize_item_with_failing_repr_at_the_depth_limit(data: object) -> None:
    assert isinstance(summarize(data, max_depth=0), str)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([Bad()] * 20, id="list"),
        pytest.param({i: Bad() for i in range(20)}, id="dict"),
    ],
)
def test_summarize_items_with_failing_repr_above_the_preview_limit(data: object) -> None:
    assert isinstance(summarize(data, max_depth=0), str)


def test_summarize_nested_item_with_failing_repr_at_the_depth_limit() -> None:
    assert isinstance(summarize({"a": [Bad()]}, max_depth=1), str)


def test_summarize_item_with_failing_repr_shows_a_placeholder() -> None:
    assert summarize([Bad()], max_depth=0).startswith("<list object at 0x")


def test_summarize_item_with_working_repr_at_the_depth_limit() -> None:
    assert summarize([1, 2], max_depth=0) == "[1, 2]"
