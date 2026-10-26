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
    comparison_key,
    create_chain,
    handlers_are_equal,
    is_exact_scalar_sequence,
    report_difference,
    track_comparison,
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


###############################################
#     Tests for is_exact_scalar_sequence      #
###############################################


@pytest.mark.parametrize("seq", [[1, "a", None, True], (b"a", 2), [], ()])
def test_is_exact_scalar_sequence_true(seq: object) -> None:
    assert is_exact_scalar_sequence(seq)


@pytest.mark.parametrize("seq", [[1.0], [1, [2]], {1, 2}, "abc", range(3)])
def test_is_exact_scalar_sequence_false(seq: object) -> None:
    assert not is_exact_scalar_sequence(seq)


def test_is_exact_scalar_sequence_subclass_of_list() -> None:
    class MyList(list):
        pass

    assert not is_exact_scalar_sequence(MyList([1]))


############################################
#     Tests for track_comparison           #
############################################


def test_track_comparison_first_visit() -> None:
    config = EqualityConfig()
    with track_comparison(config, [], []) as in_progress:
        assert not in_progress


def test_track_comparison_nested_visit_of_the_same_pair() -> None:
    config = EqualityConfig()
    x, y = [], []
    with track_comparison(config, x, y), track_comparison(config, x, y) as in_progress:
        assert in_progress


def test_track_comparison_forgets_the_pair_on_exit() -> None:
    config = EqualityConfig()
    x, y = [], []
    with track_comparison(config, x, y):
        pass
    with track_comparison(config, x, y) as in_progress:
        assert not in_progress


###########################################
#     Tests for comparison_key            #
###########################################


def test_comparison_key_frozenset_with_nan() -> None:
    config = EqualityConfig(equal_nan=True)
    assert comparison_key(frozenset({float("nan")}), config) == comparison_key(
        frozenset({float("nan")}), config
    )


def test_comparison_key_frozenset_without_equal_nan() -> None:
    config = EqualityConfig()
    assert comparison_key(frozenset({float("nan")}), config) != comparison_key(
        frozenset({float("nan")}), config
    )


def test_comparison_key_keeps_the_type() -> None:
    assert comparison_key(1, EqualityConfig()) != comparison_key(True, EqualityConfig())
