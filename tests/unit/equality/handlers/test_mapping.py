from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, SameKeysHandler, TrueHandler

if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#####################################
#     Tests for SameKeysHandler     #
#####################################


def test_same_keys_handler_eq_true() -> None:
    assert SameKeysHandler() == SameKeysHandler()


def test_same_keys_handler_eq_false() -> None:
    assert SameKeysHandler() != FalseHandler()


def test_same_keys_handler_str() -> None:
    assert str(SameKeysHandler()).startswith("SameKeysHandler(")


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({}, {}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
    ],
)
def test_same_keys_handler_handle_true(
    object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    assert SameKeysHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({"a": 1, "b": 2}, {}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 1}),
    ],
)
def test_same_keys_handler_handle_false(
    object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    assert not SameKeysHandler().handle(object1, object2, config)


def test_same_keys_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameKeysHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1={"a": 1, "b": 2}, object2={"a": 1, "b": 2, "c": 1}, config=config
        )
        assert caplog.messages[0].startswith("objects have different keys:")


def test_same_keys_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameKeysHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1={"a": 1, "b": 2}, object2={"a": 1, "b": 2}, config=config)


def test_same_keys_handler_set_next_handler() -> None:
    handler = SameKeysHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_same_keys_handler_set_next_handler_incorrect() -> None:
    handler = SameKeysHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)
