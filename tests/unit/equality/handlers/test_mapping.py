from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import (
    FalseHandler,
    MappingSameKeysHandler,
    MappingSameValuesHandler,
    TrueHandler,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


############################################
#     Tests for MappingSameKeysHandler     #
############################################


def test_mapping_same_keys_handler_eq_true() -> None:
    assert MappingSameKeysHandler() == MappingSameKeysHandler()


def test_mapping_same_keys_handler_eq_false() -> None:
    assert MappingSameKeysHandler() != FalseHandler()


def test_mapping_same_keys_handler_repr() -> None:
    assert repr(MappingSameKeysHandler()).startswith("MappingSameKeysHandler(")


def test_mapping_same_keys_handler_str() -> None:
    assert str(MappingSameKeysHandler()).startswith("MappingSameKeysHandler(")


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({}, {}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
    ],
)
def test_mapping_same_keys_handler_handle_true(
    object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    assert MappingSameKeysHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({"a": 1, "b": 2}, {}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 1}),
    ],
)
def test_mapping_same_keys_handler_handle_false(
    object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    assert not MappingSameKeysHandler().handle(object1, object2, config)


def test_mapping_same_keys_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = MappingSameKeysHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1={"a": 1, "b": 2}, object2={"a": 1, "b": 2, "c": 1}, config=config
        )
        assert caplog.messages[0].startswith("mappings have different keys:")


def test_mapping_same_keys_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = MappingSameKeysHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1={"a": 1, "b": 2}, object2={"a": 1, "b": 2}, config=config)


def test_mapping_same_keys_handler_set_next_handler() -> None:
    handler = MappingSameKeysHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_mapping_same_keys_handler_set_next_handler_incorrect() -> None:
    handler = MappingSameKeysHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)


###############################################
#     Tests for  MappingSameValuesHandler     #
###############################################


def test_mapping_same_values_handler_eq_true() -> None:
    assert MappingSameValuesHandler() == MappingSameValuesHandler()


def test_mapping_same_values_handler_eq_false() -> None:
    assert MappingSameValuesHandler() != FalseHandler()


def test_mapping_same_values_handler_repr() -> None:
    assert repr(MappingSameValuesHandler()).startswith("MappingSameValuesHandler(")


def test_mapping_same_values_handler_str() -> None:
    assert str(MappingSameValuesHandler()).startswith("MappingSameValuesHandler(")


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({}, {}),
        ({}, {"a": 1, "b": 2}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
        ({"a": 1, "b": 2}, {"0": 0, "a": 1, "b": 2, "c": 1}),
        ({"a": 1, "b": {"k": 1}}, {"a": 1, "b": {"k": 1}}),
    ],
)
def test_mapping_same_values_handler_handle_true(
    object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    assert MappingSameValuesHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}),
        ({"a": 1, "b": {"k": 1}}, {"a": 1, "b": {"k": 2}}),
    ],
)
def test_mapping_same_values_handler_handle_false(
    object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    assert not MappingSameValuesHandler().handle(object1, object2, config)


def test_mapping_same_values_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = MappingSameValuesHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(object1={"a": 1, "b": 2}, object2={"a": 1, "b": 3}, config=config)
        assert caplog.messages[-1].startswith("mappings have at least one different value:")


def test_mapping_same_values_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = MappingSameValuesHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1={"a": 1, "b": 2}, object2={"a": 1, "b": 2}, config=config)


def test_mapping_same_values_handler_set_next_handler() -> None:
    handler = MappingSameValuesHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_mapping_same_values_handler_set_next_handler_incorrect() -> None:
    handler = MappingSameValuesHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)
