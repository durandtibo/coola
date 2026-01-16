from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, SequenceSameValuesHandler, TrueHandler
from coola.testing.fixtures import numpy_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


###############################################
#     Tests for SequenceSameValuesHandler     #
###############################################


def test_sequence_same_values_handler_repr() -> None:
    assert repr(SequenceSameValuesHandler()) == "SequenceSameValuesHandler()"


def test_sequence_same_values_handler_repr_with_next_handler() -> None:
    assert (
        repr(SequenceSameValuesHandler(FalseHandler()))
        == "SequenceSameValuesHandler(next_handler=FalseHandler())"
    )


def test_sequence_same_values_handler_str() -> None:
    assert str(SequenceSameValuesHandler()) == "SequenceSameValuesHandler()"


def test_sequence_same_values_handler_str_with_next_handler() -> None:
    assert str(SequenceSameValuesHandler(FalseHandler())) == "SequenceSameValuesHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            SequenceSameValuesHandler(), SequenceSameValuesHandler(), id="without next handler"
        ),
        pytest.param(
            SequenceSameValuesHandler(FalseHandler()),
            SequenceSameValuesHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_sequence_same_values_handler_equal_true(
    handler1: SequenceSameValuesHandler, handler2: SequenceSameValuesHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            SequenceSameValuesHandler(TrueHandler()),
            SequenceSameValuesHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            SequenceSameValuesHandler(),
            SequenceSameValuesHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            SequenceSameValuesHandler(FalseHandler()),
            SequenceSameValuesHandler(),
            id="other next handler is none",
        ),
        pytest.param(SequenceSameValuesHandler(), FalseHandler(), id="different type"),
    ],
)
def test_sequence_same_values_handler_equal_false(
    handler1: SequenceSameValuesHandler, handler2: object
) -> None:
    assert not handler1.equal(handler2)


def test_sequence_same_values_handler_equal_false_different_type_child() -> None:
    class Child(SequenceSameValuesHandler): ...

    assert not SequenceSameValuesHandler().equal(Child())


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ([0, 1, 2], [0, 1, 2]),
        ((0, 1, 2), (0, 1, 2)),
        ([0, ("a", "b", "c"), 2], [0, ("a", "b", "c"), 2]),
        ([0, 1, 2], [0, 1, 2, 3]),
        ([0, 1, 2, 3], [0, 1, 2]),
    ],
)
def test_sequence_same_values_handler_handle_true(
    actual: Sequence, expected: Sequence, config: EqualityConfig
) -> None:
    assert SequenceSameValuesHandler(next_handler=TrueHandler()).handle(actual, expected, config)


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.zeros(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.zeros(2))),
    ],
)
def test_sequence_same_values_handler_handle_true_numpy(
    actual: Sequence, expected: Sequence, config: EqualityConfig
) -> None:
    assert SequenceSameValuesHandler(next_handler=TrueHandler()).handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ([1, 2, 3], [1, 2, 4]),
        ((1, 2, 3), (1, 2, 4)),
        ([0, ("a", "b", "c"), 2], [0, ("a", "b", "d"), 2]),
    ],
)
def test_sequence_same_values_handler_handle_false(
    actual: Sequence, expected: Sequence, config: EqualityConfig
) -> None:
    assert not SequenceSameValuesHandler().handle(actual, expected, config)


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.ones(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.ones(2))),
    ],
)
def test_sequence_same_values_handler_handle_false_numpy(
    actual: Sequence, expected: Sequence, config: EqualityConfig
) -> None:
    assert not SequenceSameValuesHandler(next_handler=TrueHandler()).handle(
        actual, expected, config
    )


def test_sequence_same_values_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SequenceSameValuesHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(actual=[1, 2, 3], expected=[1, 2, 4], config=config)
        assert caplog.messages[-1].startswith("sequences have at least one different value:")


def test_sequence_same_values_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SequenceSameValuesHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)


def test_sequence_same_values_handler_set_next_handler() -> None:
    handler = SequenceSameValuesHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_sequence_same_values_handler_set_next_handler_none() -> None:
    handler = SequenceSameValuesHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_sequence_same_values_handler_set_next_handler_incorrect() -> None:
    handler = SequenceSameValuesHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


def test_sequence_same_values_handler_circular_reference() -> None:
    config = EqualityConfig()
    handler = SequenceSameValuesHandler(TrueHandler())
    # Create a circular reference - list containing itself
    obj = []
    obj.append(obj)
    # Should not hang or raise RecursionError
    assert handler.handle(obj, obj, config)


def test_sequence_same_values_handler_nested_circular_reference() -> None:
    config = EqualityConfig()
    handler = SequenceSameValuesHandler(TrueHandler())
    # Create nested circular references
    obj1 = [1, 2]
    obj2 = [3, 4]
    obj1.append(obj2)
    obj2.append(obj1)
    # Should not hang or raise RecursionError
    assert handler.handle(obj1, obj1, config)
