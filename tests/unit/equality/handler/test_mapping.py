from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    MappingSameKeysHandler,
    MappingSameValuesHandler,
    TrueHandler,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


############################################
#     Tests for MappingSameKeysHandler     #
############################################


def test_mapping_same_keys_handler_repr() -> None:
    assert repr(MappingSameKeysHandler()) == "MappingSameKeysHandler()"


def test_mapping_same_keys_handler_str() -> None:
    assert str(MappingSameKeysHandler()) == "MappingSameKeysHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(MappingSameKeysHandler(), MappingSameKeysHandler(), id="without next handler"),
        pytest.param(
            MappingSameKeysHandler(FalseHandler()),
            MappingSameKeysHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_mapping_same_keys_handler_equal_true(
    handler1: MappingSameKeysHandler, handler2: MappingSameKeysHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            MappingSameKeysHandler(TrueHandler()),
            MappingSameKeysHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            MappingSameKeysHandler(),
            MappingSameKeysHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            MappingSameKeysHandler(FalseHandler()),
            MappingSameKeysHandler(),
            id="other next handler is none",
        ),
        pytest.param(MappingSameKeysHandler(), FalseHandler(), id="different type"),
    ],
)
def test_mapping_same_keys_handler_equal_false(
    handler1: MappingSameKeysHandler, handler2: object
) -> None:
    assert not handler1.equal(handler2)


def test_mapping_same_keys_handler_equal_false_different_type_child() -> None:
    class Child(MappingSameKeysHandler): ...

    assert not MappingSameKeysHandler().equal(Child())


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
    assert handler.next_handler.equal(FalseHandler())


def test_mapping_same_keys_handler_set_next_handler_none() -> None:
    handler = MappingSameKeysHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_mapping_same_keys_handler_set_next_handler_incorrect() -> None:
    handler = MappingSameKeysHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


###############################################
#     Tests for  MappingSameValuesHandler     #
###############################################


def test_mapping_same_values_handler_repr() -> None:
    assert repr(MappingSameValuesHandler()) == "MappingSameValuesHandler()"


def test_mapping_same_values_handler_str() -> None:
    assert str(MappingSameValuesHandler()) == "MappingSameValuesHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            MappingSameValuesHandler(), MappingSameValuesHandler(), id="without next handler"
        ),
        pytest.param(
            MappingSameValuesHandler(FalseHandler()),
            MappingSameValuesHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_mapping_same_values_handler_equal_true(
    handler1: MappingSameValuesHandler, handler2: MappingSameValuesHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            MappingSameValuesHandler(TrueHandler()),
            MappingSameValuesHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            MappingSameValuesHandler(),
            MappingSameValuesHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            MappingSameValuesHandler(FalseHandler()),
            MappingSameValuesHandler(),
            id="other next handler is none",
        ),
        pytest.param(MappingSameValuesHandler(), FalseHandler(), id="different type"),
    ],
)
def test_mapping_same_values_handler_equal_false(
    handler1: MappingSameValuesHandler, handler2: object
) -> None:
    assert not handler1.equal(handler2)


def test_mapping_same_values_handler_equal_false_different_type_child() -> None:
    class Child(MappingSameValuesHandler): ...

    assert not MappingSameValuesHandler().equal(Child())


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
    assert handler.next_handler.equal(FalseHandler())


def test_mapping_same_values_handler_set_next_handler_none() -> None:
    handler = MappingSameValuesHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_mapping_same_values_handler_set_next_handler_incorrect() -> None:
    handler = MappingSameValuesHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)
