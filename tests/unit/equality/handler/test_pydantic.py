from __future__ import annotations

import functools
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, PydanticModelSameFieldsHandler, TrueHandler
from coola.testing.fixtures import pydantic_available
from coola.utils.imports import is_pydantic_available
from tests.unit.equality.utils import BadStr, messages

if is_pydantic_available():
    from pydantic import BaseModel, ConfigDict, PrivateAttr
else:
    BaseModel, ConfigDict, PrivateAttr = object, dict, Mock()


class Point(BaseModel):
    x: float
    y: float = 0.0


class Cached(BaseModel):
    x: int

    @functools.cached_property
    def double(self) -> int:
        return self.x * 2


class Secret(BaseModel):
    x: float
    _cache: int = PrivateAttr(default=0)


class Holder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: object


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


def test_pydantic_model_same_fields_handler_equal_true() -> None:
    assert PydanticModelSameFieldsHandler().equal(PydanticModelSameFieldsHandler())


def test_pydantic_model_same_fields_handler_equal_false_different_type() -> None:
    assert not PydanticModelSameFieldsHandler().equal(TrueHandler())


def test_pydantic_model_same_fields_handler_equal_false_different_next() -> None:
    assert not PydanticModelSameFieldsHandler(TrueHandler()).equal(PydanticModelSameFieldsHandler())


def test_pydantic_model_same_fields_handler_str() -> None:
    assert str(PydanticModelSameFieldsHandler()) == "PydanticModelSameFieldsHandler()"


@pydantic_available
def test_pydantic_model_same_fields_handler_handle_true(config: EqualityConfig) -> None:
    handler = PydanticModelSameFieldsHandler(next_handler=TrueHandler())
    assert handler.handle(Point(x=1.0, y=2.0), Point(x=1.0, y=2.0), config)


@pydantic_available
def test_pydantic_model_same_fields_handler_handle_false(config: EqualityConfig) -> None:
    handler = PydanticModelSameFieldsHandler(next_handler=TrueHandler())
    assert not handler.handle(Point(x=1.0, y=2.0), Point(x=1.0, y=3.0), config)


@pydantic_available
def test_pydantic_model_same_fields_handler_handle_calls_next(config: EqualityConfig) -> None:
    handler = PydanticModelSameFieldsHandler(next_handler=FalseHandler())
    assert not handler.handle(Point(x=1.0), Point(x=1.0), config)


@pydantic_available
def test_pydantic_model_same_fields_handler_handle_ignores_cached_property(
    config: EqualityConfig,
) -> None:
    """Test that the value of a cached property read on one model only
    does not make the models different."""
    actual, expected = Cached(x=1), Cached(x=1)
    assert actual.double == 2
    handler = PydanticModelSameFieldsHandler(next_handler=TrueHandler())
    assert handler.handle(actual, expected, config)


@pydantic_available
def test_pydantic_model_same_fields_handler_handle_ignores_private_attributes(
    config: EqualityConfig,
) -> None:
    actual, expected = Secret(x=1.0), Secret(x=1.0)
    actual._cache = 1
    handler = PydanticModelSameFieldsHandler(next_handler=TrueHandler())
    assert handler.handle(actual, expected, config)


@pydantic_available
def test_pydantic_model_same_fields_handler_handle_tolerance() -> None:
    config = EqualityConfig(atol=0.1)
    handler = PydanticModelSameFieldsHandler(next_handler=TrueHandler())
    assert handler.handle(Point(x=1.0), Point(x=1.05), config)


@pydantic_available
def test_pydantic_model_same_fields_handler_handle_explaining() -> None:
    config = EqualityConfig(reports=[])
    handler = PydanticModelSameFieldsHandler(next_handler=TrueHandler())
    assert not handler.handle(Point(x=1.0, y=2.0), Point(x=1.0, y=3.0), config)


@pydantic_available
def test_pydantic_model_same_fields_handler_handle_explaining_reports() -> None:
    config = EqualityConfig(reports=[])
    handler = PydanticModelSameFieldsHandler(next_handler=TrueHandler())
    handler.handle(Point(x=1.0, y=2.0), Point(x=1.0, y=3.0), config)
    assert messages(config)[-1] == "objects have different y: 2.0 vs 3.0"


@pydantic_available
def test_pydantic_model_same_fields_handler_handle_field_with_failing_str() -> None:
    config = EqualityConfig(reports=[])
    handler = PydanticModelSameFieldsHandler(next_handler=TrueHandler())
    assert not handler.handle(Holder(value=BadStr(1)), Holder(value=BadStr(2)), config)


@pydantic_available
def test_pydantic_model_same_fields_handler_handle_cycle(config: EqualityConfig) -> None:
    actual, expected = Holder(value=None), Holder(value=None)
    actual.value = actual
    expected.value = expected
    handler = PydanticModelSameFieldsHandler(next_handler=TrueHandler())
    assert handler.handle(actual, expected, config)
