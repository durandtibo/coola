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
    supports_methods,
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


######################################
#     Tests for supports_methods     #
######################################


class _WithMethods:
    def allclose(self) -> bool:
        return True

    def equal(self) -> bool:
        return True

    not_callable = 42


def test_supports_methods_no_method_names() -> None:
    assert supports_methods(_WithMethods())


def test_supports_methods_single_method_true() -> None:
    assert supports_methods(_WithMethods(), "allclose")


def test_supports_methods_multiple_methods_true() -> None:
    assert supports_methods(_WithMethods(), "allclose", "equal")


def test_supports_methods_missing_method_false() -> None:
    assert not supports_methods(_WithMethods(), "does_not_exist")


def test_supports_methods_one_missing_among_several_false() -> None:
    assert not supports_methods(_WithMethods(), "allclose", "does_not_exist")


def test_supports_methods_attribute_not_callable_false() -> None:
    assert not supports_methods(_WithMethods(), "not_callable")


def test_supports_methods_on_plain_object_false() -> None:
    assert not supports_methods(object(), "allclose")


def test_supports_methods_on_int_true() -> None:
    assert supports_methods(1, "bit_length")
