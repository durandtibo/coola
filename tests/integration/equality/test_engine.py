r"""Check the engine with the default registry and with deep
structures."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from coola.equality import MaxDepthExceededError, objects_are_equal
from coola.equality.config import EqualityConfig
from coola.equality.engine import Engine
from coola.equality.tester import BaseEqualityTester, EqualityTesterRegistry, get_default_registry
from tests.unit.equality.test_engine import DictTester, LeafTester, ListTester


def _nested(depth: int) -> dict | int:
    obj: dict | int = 0
    for _ in range(depth):
        obj = {"child": obj}
    return obj


@pytest.fixture
def engine_registry() -> EqualityTesterRegistry:
    registry = EqualityTesterRegistry()
    registry.register(object, LeafTester())
    registry.register(list, ListTester())
    registry.register(dict, DictTester())
    return registry


def _recurse_then_run(depth: int, func: Any) -> Any:
    if depth == 0:
        return func()
    return _recurse_then_run(depth - 1, func)


def test_default_registry_still_compares_nested_structures() -> None:
    registry = get_default_registry()
    assert objects_are_equal([1, {"a": [2.0]}], [1, {"a": [2.0]}], registry=registry)


def test_default_registry_still_reports_the_difference() -> None:
    registry = get_default_registry()
    assert not objects_are_equal([1, {"a": [2.0]}], [1, {"a": [3.0]}], registry=registry)


def test_engine_compares_100_000_nested_dicts(engine_registry: EqualityTesterRegistry) -> None:
    config = EqualityConfig(registry=engine_registry, max_depth=200_000)
    assert Engine(config).run(_nested(100_000), _nested(100_000))


def test_engine_compares_nested_dicts_from_a_deep_caller(
    engine_registry: EqualityTesterRegistry,
) -> None:
    config = EqualityConfig(registry=engine_registry)
    depth = sys.getrecursionlimit() - 300
    assert _recurse_then_run(
        min(depth, 500), lambda: Engine(config).run(_nested(100), _nested(100))
    )


def test_engine_depth_limit_with_a_deep_structure(engine_registry: EqualityTesterRegistry) -> None:
    config = EqualityConfig(registry=engine_registry, max_depth=99)
    with pytest.raises(MaxDepthExceededError, match=r"Maximum recursion depth \(99\) exceeded"):
        Engine(config).run(_nested(100), _nested(100))


class Wrapper:
    def __init__(self, value: int) -> None:
        self.value = value


class WrapperTester(BaseEqualityTester[Wrapper]):
    engine_protocol = True

    def objects_are_equal(self, actual: Wrapper, expected: Wrapper, ctx: Any) -> bool:
        return actual.value == expected.value or ctx.fail("values differ")


def test_engine_tester_inside_a_chain_tester_of_the_default_registry() -> None:
    registry = get_default_registry().copy()
    registry.register(Wrapper, WrapperTester())
    config = EqualityConfig(registry=registry)
    assert Engine(config).run([{"a": Wrapper(1)}], [{"a": Wrapper(1)}])


def test_engine_tester_difference_inside_a_chain_tester_of_the_default_registry() -> None:
    registry = get_default_registry().copy()
    registry.register(Wrapper, WrapperTester())
    config = EqualityConfig(registry=registry)
    assert not Engine(config).run([{"a": Wrapper(1)}], [{"a": Wrapper(2)}])
