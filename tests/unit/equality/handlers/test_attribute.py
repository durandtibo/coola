from __future__ import annotations

import logging
from typing import Any
from unittest.mock import Mock

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, SameAttributeHandler, TrueHandler
from coola.testing import numpy_available
from coola.utils import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


##########################################
#     Tests for SameAttributeHandler     #
##########################################


def test_same_attribute_handler_eq_true() -> None:
    assert SameAttributeHandler(name="name") == SameAttributeHandler(name="name")


def test_same_attribute_handler_eq_false_different_type() -> None:
    assert SameAttributeHandler(name="name") != FalseHandler()


def test_same_attribute_handler_eq_false_different_name() -> None:
    assert SameAttributeHandler(name="name1") != SameAttributeHandler(name="name2")


def test_same_attribute_handler_repr() -> None:
    assert repr(SameAttributeHandler(name="name")).startswith("SameAttributeHandler(")


def test_same_attribute_handler_str() -> None:
    assert str(SameAttributeHandler(name="name")).startswith("SameAttributeHandler(")


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (Mock(data=1), Mock(data=1)),
        (Mock(data="abc"), Mock(data="abc")),
        (Mock(data=[1, 2, 3]), Mock(data=[1, 2, 3])),
    ],
)
def test_same_attribute_handler_handle_true(
    object1: Any, object2: Any, config: EqualityConfig
) -> None:
    assert SameAttributeHandler(name="data", next_handler=TrueHandler()).handle(
        object1, object2, config
    )


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (Mock(data=1), Mock(data=2)),
        (Mock(data="abc"), Mock(data="abcd")),
        (Mock(data=[1, 2, 3]), Mock(data=[1, 2, 4])),
    ],
)
def test_same_attribute_handler_handle_false(
    object1: Any, object2: Any, config: EqualityConfig
) -> None:
    assert not SameAttributeHandler(name="data").handle(object1, object2, config)


@numpy_available
def test_same_attribute_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameAttributeHandler(name="data")
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1=Mock(data=1),
            object2=Mock(data=2),
            config=config,
        )
        assert caplog.messages[-1].startswith("objects have different data:")


@numpy_available
def test_same_attribute_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameAttributeHandler(name="data")
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(
            object1=Mock(spec=Any, data=1), object2=Mock(spec=Any, data=1), config=config
        )


def test_same_attribute_handler_set_next_handler() -> None:
    handler = SameAttributeHandler(name="data")
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_same_attribute_handler_set_next_handler_incorrect() -> None:
    handler = SameAttributeHandler(name="data")
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)


@numpy_available
def test_same_attribute_handler_handle_true_numpy(config: EqualityConfig) -> None:
    assert SameAttributeHandler(name="dtype", next_handler=TrueHandler()).handle(
        np.ones(shape=(2, 3)), np.ones(shape=(2, 3)), config
    )


@numpy_available
def test_same_attribute_handler_handle_false_numpy(config: EqualityConfig) -> None:
    assert not SameAttributeHandler(name="dtype", next_handler=TrueHandler()).handle(
        np.ones(shape=(2, 3), dtype=float), np.ones(shape=(2, 3), dtype=int), config
    )
