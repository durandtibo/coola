from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.equality.tester import get_default_registry as get_equality_registry
from coola.hashing import get_default_registry as get_hasher_registry
from coola.iterator import get_default_registry as get_iterator_registry
from coola.random import get_default_registry as get_random_registry
from coola.recursive import get_default_registry as get_transformer_registry
from coola.summary import get_default_registry as get_summarizer_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.registry import TypeRegistry

REGISTRIES = [
    pytest.param(get_equality_registry, id="equality"),
    pytest.param(get_hasher_registry, id="hashing"),
    pytest.param(get_iterator_registry, id="iterator"),
    pytest.param(get_random_registry, id="random"),
    pytest.param(get_transformer_registry, id="recursive"),
    pytest.param(get_summarizer_registry, id="summary"),
]


class Point:
    pass


###########################################
#     Tests for isolated_default_registries     #
###########################################

TEST_FILE = """
import pytest

from coola.equality.tester import DefaultEqualityTester, get_default_registry


class Point:
    pass


@pytest.mark.usefixtures("isolated_default_registries")
def test_registers_then_fails():
    get_default_registry().register(Point, DefaultEqualityTester())
    raise RuntimeError("fail")


def test_registration_is_gone():
    assert not get_default_registry().has_equality_tester(Point)
"""


def test_fixture_is_restored_when_the_test_fails(pytester: pytest.Pytester) -> None:
    pytester.makeconftest('pytest_plugins = ["coola.testing.fixtures"]')
    pytester.makepyfile(TEST_FILE)
    pytester.runpytest().assert_outcomes(passed=1, failed=1)


@pytest.mark.parametrize("get_registry", REGISTRIES)
@pytest.mark.usefixtures("isolated_default_registries")
def test_fixture_registers_in_each_registry(get_registry: Callable[[], TypeRegistry]) -> None:
    registry = get_registry()
    registry.register(Point, next(iter(registry.values())))
    assert registry.has(Point)


@pytest.mark.parametrize("get_registry", REGISTRIES)
def test_fixture_does_not_leak_to_the_next_test(get_registry: Callable[[], TypeRegistry]) -> None:
    """Test that the type registered by the previous test is gone."""
    assert not get_registry().has(Point)
