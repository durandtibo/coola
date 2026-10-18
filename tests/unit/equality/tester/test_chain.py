from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    BaseEqualityHandler,
    ObjectEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
    create_chain,
)
from coola.equality.tester import BaseChainEqualityTester
from coola.utils.imports import MissingPackageError


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


class MyTester(BaseChainEqualityTester[object]):
    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(SameObjectHandler(), SameTypeHandler(), ObjectEqualHandler())


class MyOtherTester(MyTester):
    pass


class RequiresPackageTester(MyTester):
    def _check_dependencies(self) -> None:
        msg = "'missing' package is required but not installed."
        raise MissingPackageError(msg, name="missing")


class CountingTester(MyTester):
    created = 0

    def _create_handler(self) -> BaseEqualityHandler:
        type(self).created += 1
        return super()._create_handler()


###########################################
#     Tests for BaseChainEqualityTester     #
###########################################


def test_base_chain_equality_tester_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseChainEqualityTester()  # type: ignore[abstract]


def test_base_chain_equality_tester_repr() -> None:
    assert repr(MyTester()) == "MyTester()"


def test_base_chain_equality_tester_str() -> None:
    assert str(MyTester()) == "MyTester()"


def test_base_chain_equality_tester_equal_true() -> None:
    assert MyTester().equal(MyTester())


def test_base_chain_equality_tester_equal_false_different_type() -> None:
    assert not MyTester().equal(MyOtherTester())
    assert not MyTester().equal(1)


def test_base_chain_equality_tester_objects_are_equal_true(config: EqualityConfig) -> None:
    assert MyTester().objects_are_equal([1], [1], config)


def test_base_chain_equality_tester_objects_are_equal_false(config: EqualityConfig) -> None:
    assert not MyTester().objects_are_equal(1, 2, config)
    assert not MyTester().objects_are_equal(1, 1.0, config)


def test_base_chain_equality_tester_check_dependencies_error_is_raised() -> None:
    with pytest.raises(MissingPackageError, match="'missing' package is required"):
        RequiresPackageTester()


def test_base_chain_equality_tester_creates_the_chain_once_at_init(config: EqualityConfig) -> None:
    CountingTester.created = 0
    tester = CountingTester()
    tester.objects_are_equal(1, 1, config)
    tester.objects_are_equal(1, 2, config)
    assert CountingTester.created == 1
