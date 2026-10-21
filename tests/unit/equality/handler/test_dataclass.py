from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import DataclassSameFieldsHandler, FalseHandler, TrueHandler


@dataclass
class Point:
    x: float
    y: float = 0.0


@dataclass
class Ignored:
    x: float
    note: str = field(default="", compare=False)


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


def test_dataclass_same_fields_handler_equal_true() -> None:
    assert DataclassSameFieldsHandler().equal(DataclassSameFieldsHandler())


def test_dataclass_same_fields_handler_equal_false_different_type() -> None:
    assert not DataclassSameFieldsHandler().equal(TrueHandler())


def test_dataclass_same_fields_handler_equal_false_different_next() -> None:
    assert not DataclassSameFieldsHandler(TrueHandler()).equal(DataclassSameFieldsHandler())


def test_dataclass_same_fields_handler_str() -> None:
    assert str(DataclassSameFieldsHandler()) == "DataclassSameFieldsHandler()"


def test_dataclass_same_fields_handler_handle_true(config: EqualityConfig) -> None:
    handler = DataclassSameFieldsHandler(next_handler=TrueHandler())
    assert handler.handle(Point(1.0, 2.0), Point(1.0, 2.0), config)


def test_dataclass_same_fields_handler_handle_false(config: EqualityConfig) -> None:
    handler = DataclassSameFieldsHandler(next_handler=TrueHandler())
    assert not handler.handle(Point(1.0, 2.0), Point(1.0, 3.0), config)


def test_dataclass_same_fields_handler_handle_calls_next(config: EqualityConfig) -> None:
    handler = DataclassSameFieldsHandler(next_handler=FalseHandler())
    assert not handler.handle(Point(1.0), Point(1.0), config)


def test_dataclass_same_fields_handler_handle_ignores_compare_false(
    config: EqualityConfig,
) -> None:
    handler = DataclassSameFieldsHandler(next_handler=TrueHandler())
    assert handler.handle(Ignored(1.0, "a"), Ignored(1.0, "b"), config)


def test_dataclass_same_fields_handler_handle_tolerance() -> None:
    config = EqualityConfig(atol=0.1)
    handler = DataclassSameFieldsHandler(next_handler=TrueHandler())
    assert handler.handle(Point(1.0), Point(1.05), config)


def test_dataclass_same_fields_handler_handle_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = EqualityConfig(show_difference=True)
    handler = DataclassSameFieldsHandler(next_handler=TrueHandler())
    with caplog.at_level(logging.INFO):
        assert not handler.handle(Point(1.0, 2.0), Point(1.0, 3.0), config)
    assert caplog.messages[-1] == "objects have different y: 2.0 vs 3.0"
