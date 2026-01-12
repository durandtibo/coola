from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from coola.equality import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    MappingSameKeysHandler,
    MappingSameValuesHandler,
    TrueHandler,
)
from coola.equality.testers import EqualityTester

if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


############################################
#     Tests for MappingSameKeysHandler     #
############################################


def test_mapping_same_keys_handler__eq__true() -> None:
    assert MappingSameKeysHandler() == MappingSameKeysHandler()


def test_mapping_same_keys_handler__eq__false_different_type() -> None:
    assert MappingSameKeysHandler() != FalseHandler()


def test_mapping_same_keys_handler__eq__false_different_type_child() -> None:
    class Child(MappingSameKeysHandler): ...

    assert MappingSameKeysHandler() != Child()


def test_mapping_same_keys_handler_repr() -> None:
    assert repr(MappingSameKeysHandler()).startswith("MappingSameKeysHandler(")


def test_mapping_same_keys_handler_str() -> None:
    assert str(MappingSameKeysHandler()).startswith("MappingSameKeysHandler(")


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({}, {}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
    ],
)
def test_mapping_same_keys_handler_handle_true(
    actual: Mapping, expected: Mapping, config: EqualityConfig
) -> None:
    assert MappingSameKeysHandler(next_handler=TrueHandler()).handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({"a": 1, "b": 2}, {}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 1}),
    ],
)
def test_mapping_same_keys_handler_handle_false(
    actual: Mapping, expected: Mapping, config: EqualityConfig
) -> None:
    assert not MappingSameKeysHandler().handle(actual, expected, config)


def test_mapping_same_keys_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = MappingSameKeysHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual={"a": 1, "b": 2}, expected={"a": 1, "b": 2, "c": 1}, config=config
        )
        assert caplog.messages[0].startswith("mappings have different keys:")


def test_mapping_same_keys_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = MappingSameKeysHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual={"a": 1, "b": 2}, expected={"a": 1, "b": 2}, config=config)


def test_mapping_same_keys_handler_set_next_handler() -> None:
    handler = MappingSameKeysHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_mapping_same_keys_handler_set_next_handler_incorrect() -> None:
    handler = MappingSameKeysHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for `handler`."):
        handler.set_next_handler(None)


###############################################
#     Tests for  MappingSameValuesHandler     #
###############################################


def test_mapping_same_values_handler__eq__true() -> None:
    assert MappingSameValuesHandler() == MappingSameValuesHandler()


def test_mapping_same_values_handler__eq__false_different_type() -> None:
    assert MappingSameValuesHandler() != FalseHandler()


def test_mapping_same_values_handler__eq__false_different_type_child() -> None:
    class Child(MappingSameValuesHandler): ...

    assert MappingSameValuesHandler() != Child()


def test_mapping_same_values_handler_repr() -> None:
    assert repr(MappingSameValuesHandler()).startswith("MappingSameValuesHandler(")


def test_mapping_same_values_handler_str() -> None:
    assert str(MappingSameValuesHandler()).startswith("MappingSameValuesHandler(")


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({}, {}),
        ({}, {"a": 1, "b": 2}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
        ({"a": 1, "b": 2}, {"0": 0, "a": 1, "b": 2, "c": 1}),
        ({"a": 1, "b": {"k": 1}}, {"a": 1, "b": {"k": 1}}),
    ],
)
def test_mapping_same_values_handler_handle_true(
    actual: Mapping, expected: Mapping, config: EqualityConfig
) -> None:
    assert MappingSameValuesHandler(next_handler=TrueHandler()).handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}),
        ({"a": 1, "b": {"k": 1}}, {"a": 1, "b": {"k": 2}}),
    ],
)
def test_mapping_same_values_handler_handle_false(
    actual: Mapping, expected: Mapping, config: EqualityConfig
) -> None:
    assert not MappingSameValuesHandler().handle(actual, expected, config)


def test_mapping_same_values_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = MappingSameValuesHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(actual={"a": 1, "b": 2}, expected={"a": 1, "b": 3}, config=config)
        assert caplog.messages[-1].startswith("mappings have at least one different value:")


def test_mapping_same_values_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = MappingSameValuesHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual={"a": 1, "b": 2}, expected={"a": 1, "b": 2}, config=config)


def test_mapping_same_values_handler_set_next_handler() -> None:
    handler = MappingSameValuesHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_mapping_same_values_handler_set_next_handler_incorrect() -> None:
    handler = MappingSameValuesHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for `handler`."):
        handler.set_next_handler(None)
