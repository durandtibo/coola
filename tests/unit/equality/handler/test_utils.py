from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    ObjectEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
    check_recursion_depth,
    create_chain,
    handlers_are_equal,
)

if TYPE_CHECKING:
    from coola.equality.handler.base import BaseEqualityHandler


##################################
#     Tests for create_chain     #
##################################


def test_create_chain_1_item() -> None:
    handler = create_chain(SameObjectHandler())
    assert handler.equal(SameObjectHandler())


def test_create_chain_multiple_items() -> None:
    handler = create_chain(SameObjectHandler(), SameTypeHandler(), ObjectEqualHandler())
    assert handler.equal(SameObjectHandler(SameTypeHandler(ObjectEqualHandler())))


def test_create_chain_0_item() -> None:
    with pytest.raises(ValueError, match="At least one handler is required to create a chain"):
        create_chain()


def test_create_chain_with_none_handler() -> None:
    with pytest.raises(ValueError, match="Handler at position 0 is None"):
        create_chain(None)


def test_create_chain_with_none_handler_in_middle() -> None:
    with pytest.raises(ValueError, match="Handler at position 1 is None"):
        create_chain(SameObjectHandler(), None, ObjectEqualHandler())


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
