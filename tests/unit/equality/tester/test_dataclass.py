from __future__ import annotations

from dataclasses import dataclass

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import DataclassEqualityTester
from tests.unit.equality.utils import run_tester


@dataclass
class Point:
    x: float


@dataclass
class Other:
    x: float


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


def test_dataclass_equality_tester_str() -> None:
    assert str(DataclassEqualityTester()) == "DataclassEqualityTester()"


def test_dataclass_equality_tester_equal_true() -> None:
    assert DataclassEqualityTester().equal(DataclassEqualityTester())


def test_dataclass_equality_tester_equal_false() -> None:
    assert not DataclassEqualityTester().equal(42)


def test_dataclass_equality_tester_objects_are_equal_true(config: EqualityConfig) -> None:
    assert run_tester(DataclassEqualityTester(), Point(1.0), Point(1.0), config)


def test_dataclass_equality_tester_objects_are_equal_same_object(config: EqualityConfig) -> None:
    point = Point(1.0)
    assert run_tester(DataclassEqualityTester(), point, point, config)


def test_dataclass_equality_tester_objects_are_equal_false_value(config: EqualityConfig) -> None:
    assert not run_tester(DataclassEqualityTester(), Point(1.0), Point(2.0), config)


def test_dataclass_equality_tester_objects_are_equal_false_type(config: EqualityConfig) -> None:
    assert not run_tester(DataclassEqualityTester(), Point(1.0), Other(1.0), config)
