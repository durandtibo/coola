from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, SameObjectHandler, handlers_are_equal
from coola.equality.handler.utils import check_recursion_depth

if TYPE_CHECKING:
    from coola.equality.handler.base import BaseEqualityHandler


########################################
#     Tests for handlers_are_equal     #
########################################


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        (SameObjectHandler(), SameObjectHandler()),
        (FalseHandler(), FalseHandler()),
        (None, None),
    ],
)
def test_handlers_are_equal_true(
    handler1: BaseEqualityHandler | None, handler2: BaseEqualityHandler | None
) -> None:
    assert handlers_are_equal(handler1, handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        (SameObjectHandler(), FalseHandler()),
        (SameObjectHandler(), None),
        (None, SameObjectHandler()),
    ],
)
def test_handlers_are_equal_false(
    handler1: BaseEqualityHandler | None, handler2: BaseEqualityHandler | None
) -> None:
    assert not handlers_are_equal(handler1, handler2)


###########################################
#     Tests for check_recursion_depth     #
###########################################


def test_check_recursion_depth_1() -> None:
    config = EqualityConfig()
    assert config.depth == 0
    with check_recursion_depth(config):
        assert config.depth == 1
    assert config.depth == 0


def test_check_recursion_depth_2() -> None:
    config = EqualityConfig()
    assert config.depth == 0
    with check_recursion_depth(config):
        assert config.depth == 1
        with check_recursion_depth(config):
            assert config.depth == 2
        assert config.depth == 1
    assert config.depth == 0


def test_check_recursion_depth_equal_max_depth() -> None:
    config = EqualityConfig(max_depth=100)
    config._current_depth = 100
    with (
        pytest.raises(RecursionError, match=r"Maximum recursion depth"),
        check_recursion_depth(config),
    ):
        pass


def test_check_recursion_depth_greater_than_max_depth() -> None:
    config = EqualityConfig(max_depth=100)
    config._current_depth = 101
    with (
        pytest.raises(RecursionError, match=r"Maximum recursion depth"),
        check_recursion_depth(config),
    ):
        pass
