from __future__ import annotations

import functools
from unittest.mock import Mock, patch

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import PydanticModelEqualityTester
from coola.equality.tester.pydantic import get_default_testers
from coola.equality.tolerance import Tolerance
from coola.testing.marks import numpy_available, pydantic_available
from coola.utils.imports import MissingPackageError, is_numpy_available, is_pydantic_available
from tests.unit.equality.utils import explaining, messages, run_tester

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_pydantic_available():
    from pydantic import BaseModel, ConfigDict, PrivateAttr
else:
    BaseModel, ConfigDict, PrivateAttr = object, dict, Mock()


class Point(BaseModel):
    x: float
    y: float


class OtherPoint(BaseModel):
    x: float
    y: float


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")

    x: float


class Secret(BaseModel):
    x: float
    _cache: int = PrivateAttr(default=0)


class Cached(BaseModel):
    x: int

    @functools.cached_property
    def double(self) -> int:
        return self.x * 2


class Nested(BaseModel):
    point: Point
    values: list[float]


class WithArray(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: np.ndarray


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


#################################################
#     Tests for PydanticModelEqualityTester     #
#################################################


@pydantic_available
def test_pydantic_model_equality_tester_str() -> None:
    assert str(PydanticModelEqualityTester()).startswith("PydanticModelEqualityTester(")


@pydantic_available
def test_pydantic_model_equality_tester_equal(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    assert run_tester(tester, Point(x=1.0, y=2.0), Point(x=1.0, y=2.0), config)


@pydantic_available
def test_pydantic_model_equality_tester_equal_same_object(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    point = Point(x=1.0, y=2.0)
    assert run_tester(tester, point, point, config)


@pydantic_available
def test_pydantic_model_equality_tester_different_value(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    assert not run_tester(tester, Point(x=1.0, y=2.0), Point(x=1.0, y=3.0), config)


@pydantic_available
def test_pydantic_model_equality_tester_different_type(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    assert not run_tester(tester, Point(x=1.0, y=2.0), OtherPoint(x=1.0, y=2.0), config)


@pydantic_available
def test_pydantic_model_equality_tester_different_extra(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    assert not run_tester(tester, Open(x=1.0, e=1), Open(x=1.0, e=2), config)


@pydantic_available
def test_pydantic_model_equality_tester_equal_extra(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    assert run_tester(tester, Open(x=1.0, e=1), Open(x=1.0, e=1), config)


@pydantic_available
def test_pydantic_model_equality_tester_missing_extra(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    assert not run_tester(tester, Open(x=1.0, e=1), Open(x=1.0), config)


@pydantic_available
def test_pydantic_model_equality_tester_ignores_private_attributes(
    config: EqualityConfig,
) -> None:
    first, second = Secret(x=1.0), Secret(x=1.0)
    second._cache = 5
    assert run_tester(PydanticModelEqualityTester(), first, second, config)


@pydantic_available
def test_pydantic_model_equality_tester_nested(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    actual = Nested(point=Point(x=1.0, y=2.0), values=[1.0, 2.0])
    expected = Nested(point=Point(x=1.0, y=2.0), values=[1.0, 2.0])
    assert run_tester(tester, actual, expected, config)


@pydantic_available
def test_pydantic_model_equality_tester_nested_different(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    actual = Nested(point=Point(x=1.0, y=2.0), values=[1.0, 2.0])
    expected = Nested(point=Point(x=1.0, y=5.0), values=[1.0, 2.0])
    assert not run_tester(tester, actual, expected, config)


@pydantic_available
def test_pydantic_model_equality_tester_atol() -> None:
    tester = PydanticModelEqualityTester()
    assert run_tester(
        tester,
        Point(x=1.0, y=2.0),
        Point(x=1.0, y=2.05),
        EqualityConfig(tolerance=Tolerance(atol=0.1)),
    )


@pydantic_available
def test_pydantic_model_equality_tester_without_tolerance() -> None:
    tester = PydanticModelEqualityTester()
    assert not run_tester(tester, Point(x=1.0, y=2.0), Point(x=1.0, y=2.05), EqualityConfig())


@pydantic_available
def test_pydantic_model_equality_tester_equal_nan() -> None:
    tester = PydanticModelEqualityTester()
    nan = float("nan")
    assert run_tester(
        tester, Point(x=nan, y=2.0), Point(x=nan, y=2.0), EqualityConfig(equal_nan=True)
    )


@pydantic_available
def test_pydantic_model_equality_tester_nan_without_equal_nan(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    assert not run_tester(
        tester, Point(x=float("nan"), y=2.0), Point(x=float("nan"), y=2.0), config
    )


@pydantic_available
@numpy_available
def test_pydantic_model_equality_tester_array_field(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    assert run_tester(tester, WithArray(x=np.ones(2)), WithArray(x=np.ones(2)), config)


@pydantic_available
@numpy_available
def test_pydantic_model_equality_tester_array_field_different(config: EqualityConfig) -> None:
    tester = PydanticModelEqualityTester()
    assert not run_tester(tester, WithArray(x=np.ones(2)), WithArray(x=np.zeros(2)), config)


@pydantic_available
def test_pydantic_model_equality_tester_explaining(config: EqualityConfig) -> None:
    config = explaining(config)
    tester = PydanticModelEqualityTester()
    run_tester(tester, Point(x=1.0, y=2.0), Point(x=1.0, y=3.0), config)
    assert "different" in "\n".join(messages(config))


@pydantic_available
def test_pydantic_model_equality_tester_cached_property_read_on_one_model(
    config: EqualityConfig,
) -> None:
    """Test that a cached property read on one model does not make equal
    models different."""
    actual, expected = Cached(x=1), Cached(x=1)
    assert actual.double == 2
    assert run_tester(PydanticModelEqualityTester(), actual, expected, config)


def test_pydantic_model_equality_tester_missing_package() -> None:
    with (
        patch("coola.utils.imports.optional.is_pydantic_available", lambda: False),
        pytest.raises(
            MissingPackageError, match=r"'pydantic' package is required but not installed."
        ),
    ):
        PydanticModelEqualityTester()


#########################################
#     Tests for get_default_testers     #
#########################################


@pydantic_available
def test_get_default_testers_keys() -> None:
    assert list(get_default_testers()) == [BaseModel]


@pydantic_available
def test_get_default_testers_tester_type() -> None:
    assert isinstance(get_default_testers()[BaseModel], PydanticModelEqualityTester)


def test_get_default_testers_without_pydantic() -> None:
    with patch("coola.equality.tester.pydantic.is_pydantic_available", lambda: False):
        assert get_default_testers() == {}
