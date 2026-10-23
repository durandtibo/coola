from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    AttributeName,
    FalseHandler,
    MaxDepthExceededError,
    ObjectEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
    check_recursion_depth,
    create_chain,
    handlers_are_equal,
    report_difference,
    track_comparison,
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


#######################################
#     Tests for report_difference     #
#######################################


def test_report_difference_logs_without_reporter(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test_report_difference")
    with caplog.at_level(logging.INFO, logger=logger.name):
        report_difference(logger, EqualityConfig(), "different %s", "values")
    assert caplog.messages == ["different values"]


def test_report_difference_sends_formatted_message_and_path_to_reporter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reports: list[tuple[str, tuple[object, ...]]] = []
    config = EqualityConfig(reporter=lambda message, path: reports.append((message, path)))
    logger = logging.getLogger("test_report_difference")
    with caplog.at_level(logging.INFO, logger=logger.name):
        report_difference(logger, config, "%s vs %s", 1, 2, extra={"coola_path": ("a", 0)})
    assert reports == [("1 vs 2", ("a", 0))]
    assert not caplog.messages  # nothing is logged when there is a reporter


def test_report_difference_reporter_without_args_or_extra() -> None:
    reports: list[tuple[str, tuple[object, ...]]] = []
    config = EqualityConfig(reporter=lambda message, path: reports.append((message, path)))
    report_difference(logging.getLogger("x"), config, "100% different")
    assert reports == [("100% different", ())]


def test_max_depth_exceeded_error_is_recursion_error() -> None:
    assert issubclass(MaxDepthExceededError, RecursionError)


def test_check_recursion_depth_raises_max_depth_exceeded_error() -> None:
    config = EqualityConfig(max_depth=1)
    with (
        check_recursion_depth(config),
        pytest.raises(MaxDepthExceededError),
        check_recursion_depth(config),
    ):
        pass


#####################################
#     Tests for track_comparison    #
#####################################


def test_track_comparison_first_time() -> None:
    config = EqualityConfig()
    with track_comparison(config, [], []) as in_progress:
        assert not in_progress


def test_track_comparison_same_pair_is_in_progress() -> None:
    config = EqualityConfig()
    a, b = [], []
    with track_comparison(config, a, b) as outer:
        assert not outer
        with track_comparison(config, a, b) as inner:
            assert inner


def test_track_comparison_other_pair_is_not_in_progress() -> None:
    config = EqualityConfig()
    a, b = [], []
    with track_comparison(config, a, b):
        with track_comparison(config, b, a) as swapped:
            assert not swapped
        with track_comparison(config, a, []) as other:
            assert not other


def test_track_comparison_pair_is_removed_on_exit() -> None:
    config = EqualityConfig()
    a, b = [], []
    with track_comparison(config, a, b):
        pass
    assert not config._in_progress
    with track_comparison(config, a, b) as in_progress:
        assert not in_progress


def _fail() -> None:
    msg = "boom"
    raise RuntimeError(msg)


def test_track_comparison_pair_is_removed_on_error() -> None:
    config = EqualityConfig()
    a, b = [], []
    with pytest.raises(RuntimeError, match="boom"), track_comparison(config, a, b):
        _fail()
    assert not config._in_progress


def test_track_comparison_inner_exit_keeps_outer_pair() -> None:
    config = EqualityConfig()
    a, b = [], []
    with track_comparison(config, a, b):
        with track_comparison(config, a, b):
            pass
        with track_comparison(config, a, b) as in_progress:
            assert in_progress


def test_report_difference_attribute_kind_converts_path() -> None:
    reports = []
    config = EqualityConfig(reporter=lambda _msg, path: reports.append(path))
    report_difference(
        logging.getLogger(),
        config,
        "msg",
        extra={"coola_path": ("a", "b"), "coola_path_kind": "attribute"},
    )
    assert all(isinstance(key, AttributeName) for key in reports[0])
