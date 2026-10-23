from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    MappingSameKeyOrderHandler,
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


def test_mapping_same_keys_handler_repr_with_next_handler() -> None:
    assert (
        repr(MappingSameKeysHandler(FalseHandler()))
        == "MappingSameKeysHandler(next_handler=FalseHandler())"
    )


def test_mapping_same_keys_handler_str() -> None:
    assert str(MappingSameKeysHandler()) == "MappingSameKeysHandler()"


def test_mapping_same_keys_handler_str_with_next_handler() -> None:
    assert str(MappingSameKeysHandler(FalseHandler())) == "MappingSameKeysHandler()"


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


def test_mapping_same_keys_handler_handle_false_show_difference(config: EqualityConfig) -> None:
    config.show_difference = True
    assert not MappingSameKeysHandler().handle(
        actual={"a": 1, "b": 2}, expected={"a": 1, "b": 2, "c": 1}, config=config
    )


def test_mapping_same_keys_handler_handle_false_show_difference_logs(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        MappingSameKeysHandler().handle(
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


def test_mapping_same_values_handler_repr_with_next_handler() -> None:
    assert (
        repr(MappingSameValuesHandler(FalseHandler()))
        == "MappingSameValuesHandler(next_handler=FalseHandler())"
    )


def test_mapping_same_values_handler_str() -> None:
    assert str(MappingSameValuesHandler()) == "MappingSameValuesHandler()"


def test_mapping_same_values_handler_str_with_next_handler() -> None:
    assert str(MappingSameValuesHandler(FalseHandler())) == "MappingSameValuesHandler()"


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


def test_mapping_same_values_handler_handle_false_show_difference(config: EqualityConfig) -> None:
    config.show_difference = True
    handler = MappingSameValuesHandler()
    assert not handler.handle(actual={"a": 1, "b": 2}, expected={"a": 1, "b": 3}, config=config)


def test_mapping_same_values_handler_handle_false_show_difference_logs(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = MappingSameValuesHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(actual={"a": 1, "b": 2}, expected={"a": 1, "b": 3}, config=config)
    assert "mappings have different values for key 'b'" in caplog.messages[-1]


def test_mapping_same_values_handler_handle_false_log_record_path(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = MappingSameValuesHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(actual={"a": 1, "b": 2}, expected={"a": 1, "b": 3}, config=config)
    assert caplog.records[-1].coola_path == ("b",)


def test_mapping_same_values_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = MappingSameValuesHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual={"a": 1, "b": 2}, expected={"a": 1, "b": 2}, config=config)


def test_mapping_same_values_handler_handle_does_not_reuse_cache_across_different_objects(
    config: EqualityConfig,
) -> None:
    actual = {"a": 1, "b": 2}
    expected = {"a": 1, "b": 2}
    config.registry = Mock(wraps=config.registry)
    assert MappingSameValuesHandler(next_handler=TrueHandler()).handle(actual, expected, config)


def test_mapping_same_values_handler_handle_compares_each_value_for_different_objects(
    config: EqualityConfig,
) -> None:
    actual = {"a": 1, "b": 2}
    expected = {"a": 1, "b": 2}
    registry = Mock(wraps=config.registry)
    config.registry = registry
    MappingSameValuesHandler(next_handler=TrueHandler()).handle(actual, expected, config)
    assert registry.objects_are_equal.call_count == 2


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


def test_mapping_same_values_handler_handle_does_not_cache_by_identity(
    config: EqualityConfig,
) -> None:
    shared = [1, 2, 3]
    config.registry = Mock(wraps=config.registry)
    assert MappingSameValuesHandler(next_handler=TrueHandler()).handle(
        {"a": shared, "b": shared}, {"a": shared, "b": shared}, config
    )


def test_mapping_same_values_handler_handle_compares_each_shared_value(
    config: EqualityConfig,
) -> None:
    shared = [1, 2, 3]
    registry = Mock(wraps=config.registry)
    config.registry = registry
    MappingSameValuesHandler(next_handler=TrueHandler()).handle(
        {"a": shared, "b": shared}, {"a": shared, "b": shared}, config
    )
    assert registry.objects_are_equal.call_count == 2


################################################
#     Tests for MappingSameKeyOrderHandler     #
################################################


def test_mapping_same_key_order_handler_repr() -> None:
    assert repr(MappingSameKeyOrderHandler()) == "MappingSameKeyOrderHandler()"


def test_mapping_same_key_order_handler_str() -> None:
    assert str(MappingSameKeyOrderHandler()) == "MappingSameKeyOrderHandler()"


def test_mapping_same_key_order_handler_equal_true() -> None:
    assert MappingSameKeyOrderHandler().equal(MappingSameKeyOrderHandler())


def test_mapping_same_key_order_handler_equal_false_different_type() -> None:
    assert not MappingSameKeyOrderHandler().equal(FalseHandler())


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({}, {}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
        ({"a": 1, "b": 2}, {"a": 3, "b": 4}),
        ({1: "a", "x": 2}, {1: "b", "x": 3}),
    ],
)
def test_mapping_same_key_order_handler_handle_true(
    actual: Mapping, expected: Mapping, config: EqualityConfig
) -> None:
    assert MappingSameKeyOrderHandler(next_handler=TrueHandler()).handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
        ({1: "a", "x": 2}, {"x": 2, 1: "a"}),
    ],
)
def test_mapping_same_key_order_handler_handle_false(
    actual: Mapping, expected: Mapping, config: EqualityConfig
) -> None:
    assert not MappingSameKeyOrderHandler(next_handler=TrueHandler()).handle(
        actual, expected, config
    )


def test_mapping_same_key_order_handler_handle_false_show_difference(
    config: EqualityConfig,
) -> None:
    config.show_difference = True
    handler = MappingSameKeyOrderHandler(next_handler=TrueHandler())
    assert not handler.handle(actual={"a": 1, "b": 2}, expected={"b": 2, "a": 1}, config=config)


def _log_key_order_difference(config: EqualityConfig, caplog: pytest.LogCaptureFixture) -> str:
    config.show_difference = True
    handler = MappingSameKeyOrderHandler(next_handler=TrueHandler())
    with caplog.at_level(logging.INFO):
        handler.handle(actual={"a": 1, "b": 2}, expected={"b": 2, "a": 1}, config=config)
    return caplog.messages[0]


def test_mapping_same_key_order_handler_handle_false_show_difference_logs(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    message = _log_key_order_difference(config, caplog)
    assert message.startswith("mappings have the same keys in a different order:")


def test_mapping_same_key_order_handler_handle_false_show_difference_logs_actual_keys(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    assert "['a', 'b']" in _log_key_order_difference(config, caplog)


def test_mapping_same_key_order_handler_handle_false_show_difference_logs_expected_keys(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    assert "['b', 'a']" in _log_key_order_difference(config, caplog)


def test_mapping_same_key_order_handler_handle_true_show_difference(config: EqualityConfig) -> None:
    config.show_difference = True
    handler = MappingSameKeyOrderHandler(next_handler=TrueHandler())
    assert handler.handle(actual={"a": 1}, expected={"a": 1}, config=config)


def test_mapping_same_key_order_handler_handle_true_does_not_log(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = MappingSameKeyOrderHandler(next_handler=TrueHandler())
    with caplog.at_level(logging.INFO):
        handler.handle(actual={"a": 1}, expected={"a": 1}, config=config)
    assert not caplog.messages


def test_mapping_same_key_order_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = MappingSameKeyOrderHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle({"a": 1}, {"a": 1}, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param({1: "a"}, {True: "a"}, id="int vs bool"),
        pytest.param({1: "a"}, {1.0: "a"}, id="int vs float"),
    ],
)
def test_mapping_same_keys_handler_handle_keys_of_different_types(
    actual: Mapping, expected: Mapping, config: EqualityConfig
) -> None:
    assert not MappingSameKeysHandler().handle(actual, expected, config)


def test_mapping_same_keys_handler_handle_keys_of_different_types_show_difference(
    config: EqualityConfig,
) -> None:
    config.show_difference = True
    assert not MappingSameKeysHandler().handle({1: "a"}, {True: "a"}, config)


def test_mapping_same_keys_handler_handle_keys_of_different_types_logs_missing_keys(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        MappingSameKeysHandler().handle({1: "a"}, {True: "a"}, config)
    assert "missing keys    : [1]" in caplog.text


def test_mapping_same_keys_handler_handle_keys_of_different_types_logs_additional_keys(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        MappingSameKeysHandler().handle({1: "a"}, {True: "a"}, config)
    assert "additional keys : [True]" in caplog.text
