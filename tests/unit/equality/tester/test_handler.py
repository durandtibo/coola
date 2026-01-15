from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    BaseEqualityHandler,
    ObjectEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
    create_chain,
)
from coola.equality.tester import HandlerEqualityTester


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


@pytest.fixture
def handler() -> BaseEqualityHandler:
    return create_chain(SameObjectHandler(), SameTypeHandler(), ObjectEqualHandler())


###########################################
#     Tests for HandlerEqualityTester     #
###########################################


def test_handler_equality_tester_repr(handler: BaseEqualityHandler) -> None:
    assert (
        repr(HandlerEqualityTester(handler))
        == "HandlerEqualityTester(handler=SameObjectHandler(next_handler="
        "SameTypeHandler(next_handler=ObjectEqualHandler())))"
    )


def test_handler_equality_tester_str(handler: BaseEqualityHandler) -> None:
    assert str(HandlerEqualityTester(handler)) == (
        "HandlerEqualityTester(\n"
        "  (0): SameObjectHandler()\n"
        "  (1): SameTypeHandler()\n"
        "  (2): ObjectEqualHandler()\n"
        ")"
    )


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
    handler: BaseEqualityHandler, config: EqualityConfig
) -> None:
    tester = HandlerEqualityTester(handler)
    assert tester.objects_are_equal(actual=42, expected=42, config=config)


def test_handler_equality_tester_objects_are_equal_false(
    handler: BaseEqualityHandler, config: EqualityConfig
) -> None:
    tester = HandlerEqualityTester(handler)
    assert not tester.objects_are_equal(42, 1, config)
