from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig2
from coola.equality.handler import (
    BaseEqualityHandler,
    ObjectEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.equality.tester import HandlerEqualityTester


@pytest.fixture
def config() -> EqualityConfig2:
    return EqualityConfig2()


@pytest.fixture
def handler() -> BaseEqualityHandler:
    handler = SameObjectHandler()
    handler.chain(SameTypeHandler()).chain(ObjectEqualHandler())
    return handler


###########################################
#     Tests for HandlerEqualityTester     #
###########################################


def test_handler_equality_tester_repr(handler: BaseEqualityHandler) -> None:
    assert repr(HandlerEqualityTester(handler)) == "HandlerEqualityTester()"


def test_handler_equality_tester_str(handler: BaseEqualityHandler) -> None:
    assert str(HandlerEqualityTester(handler)) == "HandlerEqualityTester()"


def test_handler_equality_tester_equal_true(handler: BaseEqualityHandler) -> None:
    assert HandlerEqualityTester(handler).equal(HandlerEqualityTester(handler))


def test_handler_equality_tester_equal_false_different_type(handler: BaseEqualityHandler) -> None:
    assert not HandlerEqualityTester(handler).equal(42)


def test_handler_equality_tester_equal_false_different_type_child(
    handler: BaseEqualityHandler,
) -> None:
    class Child(HandlerEqualityTester): ...

    assert not HandlerEqualityTester(handler).equal(Child(handler))


def test_handler_equality_tester_objects_are_equal_true(
    handler: BaseEqualityHandler, config: EqualityConfig2
) -> None:
    tester = HandlerEqualityTester(handler)
    assert tester.objects_are_equal(actual=42, expected=42, config=config)


def test_handler_equality_tester_objects_are_equal_false(
    handler: BaseEqualityHandler, config: EqualityConfig2
) -> None:
    tester = HandlerEqualityTester(handler)
    assert not tester.objects_are_equal(42, 1, config)
