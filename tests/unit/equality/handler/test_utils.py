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
    report_difference,
)
from tests.unit.equality.utils import messages

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
    with pytest.raises(ValueError, match=r"At least one handler is required to create a chain"):
        create_chain()


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


def test_check_recursion_depth_1_initial_depth() -> None:
    assert EqualityConfig().depth == 0


def test_check_recursion_depth_1() -> None:
    config = EqualityConfig()
    with check_recursion_depth(config):
        assert config.depth == 1


def test_check_recursion_depth_1_restores_the_depth() -> None:
    config = EqualityConfig()
    with check_recursion_depth(config):
        pass
    assert config.depth == 0


def test_check_recursion_depth_2_outer_depth() -> None:
    config = EqualityConfig()
    with check_recursion_depth(config):
        assert config.depth == 1


def test_check_recursion_depth_2_inner_depth() -> None:
    config = EqualityConfig()
    with check_recursion_depth(config), check_recursion_depth(config):
        assert config.depth == 2


def test_check_recursion_depth_2_restores_the_outer_depth() -> None:
    config = EqualityConfig()
    with check_recursion_depth(config):
        with check_recursion_depth(config):
            pass
        assert config.depth == 1


def test_check_recursion_depth_2_restores_the_depth() -> None:
    config = EqualityConfig()
    with check_recursion_depth(config), check_recursion_depth(config):
        pass
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


#######################################
#     Tests for report_difference     #
#######################################


def test_report_difference_records_the_formatted_message() -> None:
    config = EqualityConfig(reports=[])
    report_difference(config, "different %s", "values")
    assert messages(config) == ["different values"]


def test_report_difference_records_the_path() -> None:
    config = EqualityConfig(reports=[])
    report_difference(config, "different", path=("a", 0))
    assert config.reports == [("different", ("a", 0))]


def test_report_difference_records_an_empty_path_by_default() -> None:
    config = EqualityConfig(reports=[])
    report_difference(config, "different")
    assert config.reports == [("different", ())]


def test_report_difference_records_the_innermost_difference_first() -> None:
    config = EqualityConfig(reports=[])
    report_difference(config, "inner")
    report_difference(config, "outer")
    assert messages(config) == ["inner", "outer"]


def test_report_difference_records_nothing_when_not_explaining() -> None:
    config = EqualityConfig()
    report_difference(config, "different")
    assert config.reports is None
