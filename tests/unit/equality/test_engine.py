from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import Element, MaxDepthExceededError
from coola.equality.config import EqualityConfig
from coola.equality.engine import Engine
from coola.equality.tester import BaseEqualityTester, EqualityTesterRegistry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class LeafTester(BaseEqualityTester[Any]):
    r"""Compare the values with ``==`` and explain the difference."""

    engine_protocol = True

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(self, actual: Any, expected: Any, ctx: Any) -> bool:
        return actual == expected or ctx.fail(lambda: f"{actual!r} vs {expected!r}")


class ListTester(BaseEqualityTester[list]):
    r"""Return the pairs of the items of two lists."""

    engine_protocol = True

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self, actual: list, expected: list, ctx: Any
    ) -> bool | Iterator[tuple[object, object, object]]:
        if len(actual) != len(expected):
            return ctx.fail("lengths differ")
        return ((a, e, i) for i, (a, e) in enumerate(zip(actual, expected, strict=True)))


class DictTester(BaseEqualityTester[dict]):
    r"""Return the pairs of the values of two dicts."""

    engine_protocol = True

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self, actual: dict, expected: dict, ctx: Any
    ) -> bool | Iterator[tuple[object, object, object]]:
        if actual.keys() != expected.keys():
            return ctx.fail("keys differ")
        return ((actual[k], expected[k], k) for k in actual)


class IndexTester(BaseEqualityTester[tuple]):
    r"""Fail with a path part given to ``ctx.fail``."""

    engine_protocol = True

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(self, actual: tuple, expected: tuple, ctx: Any) -> bool:
        return actual == expected or ctx.fail("values differ", 1, 2)


class FuncTester(BaseEqualityTester[Any]):
    r"""A tester that uses the engine and calls a function."""

    engine_protocol = True

    def __init__(self, func: Callable[[Any, Any, Any], Any]) -> None:
        self.func = func

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(self, actual: Any, expected: Any, ctx: Any) -> Any:
        return self.func(actual, expected, ctx)


class LegacyTester(BaseEqualityTester[Any]):
    r"""A tester that receives the config and returns a ``bool``."""

    def __init__(self) -> None:
        self.configs: list[EqualityConfig] = []

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        self.configs.append(config)
        return actual == expected


class RecursiveLegacyTester(BaseEqualityTester[list]):
    r"""A tester that recurses through the registry, like the chain
    testers."""

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(self, actual: list, expected: list, config: EqualityConfig) -> bool:
        config.increment_depth()
        try:
            return all(
                config.registry.objects_are_equal(a, e, config)
                for a, e in zip(actual, expected, strict=True)
            )
        finally:
            config.decrement_depth()


@pytest.fixture
def registry() -> EqualityTesterRegistry:
    reg = EqualityTesterRegistry()
    reg.register(object, LeafTester())
    reg.register(list, ListTester())
    reg.register(dict, DictTester())
    reg.register(tuple, IndexTester())
    return reg


def _engine(registry: EqualityTesterRegistry, **kwargs: Any) -> Engine:
    config = EqualityConfig(registry=registry, **kwargs)
    return Engine(registry, config)


def _nested(depth: int) -> dict | int:
    obj: dict | int = 0
    for _ in range(depth):
        obj = {"child": obj}
    return obj


#####################
#     Tests run     #
#####################


def test_run_leaf_equal(registry: EqualityTesterRegistry) -> None:
    assert _engine(registry).run(1, 1)


def test_run_leaf_different(registry: EqualityTesterRegistry) -> None:
    assert not _engine(registry).run(1, 2)


def test_run_container_equal(registry: EqualityTesterRegistry) -> None:
    assert _engine(registry).run([1, {"a": [2, 3]}], [1, {"a": [2, 3]}])


def test_run_container_different(registry: EqualityTesterRegistry) -> None:
    assert not _engine(registry).run([1, {"a": [2, 3]}], [1, {"a": [2, 4]}])


def test_run_empty_container(registry: EqualityTesterRegistry) -> None:
    assert _engine(registry).run([], [])


def test_run_identical_objects_do_not_call_the_tester() -> None:
    tester = LegacyTester()
    reg = EqualityTesterRegistry()
    reg.register(object, tester)
    obj = object()
    Engine(reg, EqualityConfig(registry=reg)).run(obj, obj)
    assert tester.configs == []


def test_run_different_types_are_different(registry: EqualityTesterRegistry) -> None:
    assert not _engine(registry).run(1, 1.0)


def test_run_different_types_in_a_container_are_different(
    registry: EqualityTesterRegistry,
) -> None:
    assert not _engine(registry).run([1], [1.0])


def test_run_subclass_is_a_different_type(registry: EqualityTesterRegistry) -> None:
    class MyList(list):
        pass

    assert not _engine(registry).run([1], MyList([1]))


def test_run_returns_a_bool(registry: EqualityTesterRegistry) -> None:
    assert _engine(registry).run([1], [1]) is True


def test_run_stops_at_the_first_difference(registry: EqualityTesterRegistry) -> None:
    visited = []

    def spy(actual: Any, expected: Any, ctx: Any) -> bool:
        visited.append(actual)
        return actual == expected or ctx.fail("differ")

    registry.register(object, FuncTester(spy), exist_ok=True)
    _engine(registry).run([1, 2, 3, 4], [1, 9, 3, 4])
    assert visited == [2]


def test_run_container_children_are_lazy(registry: EqualityTesterRegistry) -> None:
    consumed = []

    def lazy(actual: list, expected: list, _ctx: Any) -> Iterator:
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            consumed.append(i)
            yield a, e, i

    registry.register(list, FuncTester(lazy), exist_ok=True)
    _engine(registry).run([1, 2, 3], [1, 5, 3])
    assert consumed == [0, 1]


def test_run_first_difference_in_iteration_order(registry: EqualityTesterRegistry) -> None:
    engine = _engine(registry, show_difference=True)
    engine.run({"a": 1, "b": 2}, {"a": 5, "b": 6})
    assert engine.failure == (("a",), "1 vs 5")


def test_run_does_not_mutate_the_inputs(registry: EqualityTesterRegistry) -> None:
    actual, expected = [1, {"a": [2]}], [1, {"a": [3]}]
    _engine(registry).run(actual, expected)
    assert (actual, expected) == ([1, {"a": [2]}], [1, {"a": [3]}])


def test_run_engine_tester_receives_the_engine(registry: EqualityTesterRegistry) -> None:
    received = []

    def spy(_actual: Any, _expected: Any, ctx: Any) -> bool:
        received.append(ctx)
        return True

    registry.register(object, FuncTester(spy), exist_ok=True)
    engine = _engine(registry)
    engine.run(1, 2)
    assert received == [engine]


def test_run_legacy_tester_receives_the_config() -> None:
    tester = LegacyTester()
    reg = EqualityTesterRegistry()
    reg.register(object, tester)
    config = EqualityConfig(registry=reg)
    Engine(reg, config).run(1, 2)
    assert tester.configs == [config]


def test_run_legacy_tester_verdict_is_a_bool() -> None:
    reg = EqualityTesterRegistry()
    reg.register(object, LegacyTester())
    assert Engine(reg, EqualityConfig(registry=reg)).run(1, 2) is False


def test_run_legacy_tester_sees_the_depth_of_the_stack(registry: EqualityTesterRegistry) -> None:
    depths = []

    class Spy(LegacyTester):
        def objects_are_equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
            depths.append(config.depth)
            return actual == expected

    registry.register(int, Spy())
    _engine(registry).run([[1]], [[2]])
    assert depths == [2]


def test_run_restores_the_depth_of_the_config(registry: EqualityTesterRegistry) -> None:
    engine = _engine(registry)
    engine.run([[1]], [[2]])
    assert engine.config.depth == 0


def test_run_nested_in_a_legacy_tester_adds_the_depth() -> None:
    reg = EqualityTesterRegistry()
    reg.register(list, RecursiveLegacyTester())
    reg.register(dict, DictTester())
    reg.register(object, LeafTester())
    config = EqualityConfig(registry=reg, max_depth=3)
    with pytest.raises(MaxDepthExceededError, match="Maximum recursion depth"):
        Engine(reg, config).run([[[{"a": {"b": {"c": 1}}}]]], [[[{"a": {"b": {"c": 2}}}]]])


def test_registry_runs_a_nested_comparison_with_the_engine(
    registry: EqualityTesterRegistry,
) -> None:
    config = EqualityConfig(registry=registry)
    assert registry.objects_are_equal([1, [2]], [1, [3]], config) is False


#####################
#     Tests cycle   #
#####################


def test_run_cycle_in_a_list_is_equal(registry: EqualityTesterRegistry) -> None:
    a, b = [1], [1]
    a.append(a)
    b.append(b)
    assert _engine(registry).run(a, b)


def test_run_cycle_in_a_dict_is_equal(registry: EqualityTesterRegistry) -> None:
    a: dict[str, Any] = {"x": 1}
    b: dict[str, Any] = {"x": 1}
    a["self"] = a
    b["self"] = b
    assert _engine(registry).run(a, b)


def test_run_cycle_with_a_difference_is_different(registry: EqualityTesterRegistry) -> None:
    a, b = [1], [2]
    a.append(a)
    b.append(b)
    assert not _engine(registry).run(a, b)


def test_run_same_pair_twice_in_siblings_is_compared_each_time(
    registry: EqualityTesterRegistry,
) -> None:
    shared_a, shared_b = [1], [2]
    assert not _engine(registry).run([shared_a, shared_b], [[1], [3]])


##################################
#     Tests depth and stack      #
##################################


def test_run_depth_exceeded_raises(registry: EqualityTesterRegistry) -> None:
    with pytest.raises(MaxDepthExceededError, match="Maximum recursion depth \\(5\\) exceeded"):
        _engine(registry, max_depth=5).run(_nested(10), _nested(10))


def test_run_depth_at_the_limit_is_accepted(registry: EqualityTesterRegistry) -> None:
    assert _engine(registry, max_depth=10).run(_nested(10), _nested(10))


def test_registry_nested_comparison_adds_the_depth(registry: EqualityTesterRegistry) -> None:
    config = EqualityConfig(registry=registry, max_depth=2)
    config.depth = 2
    with pytest.raises(MaxDepthExceededError):
        registry.objects_are_equal([[1]], [[1]], config)


def test_run_very_deep_nesting_has_no_recursion_error(registry: EqualityTesterRegistry) -> None:
    assert _engine(registry, max_depth=100_000).run(_nested(5_000), _nested(5_000))


def test_run_very_deep_nesting_difference(registry: EqualityTesterRegistry) -> None:
    other = _nested(5_000)
    node = other
    while node["child"] != 0:
        node = node["child"]
    node["child"] = 1
    assert not _engine(registry, max_depth=100_000).run(_nested(5_000), other)


#####################
#     Tests path    #
#####################


def test_failure_is_none_when_equal(registry: EqualityTesterRegistry) -> None:
    engine = _engine(registry)
    engine.run([1], [1])
    assert engine.failure is None


def test_failure_at_the_root(registry: EqualityTesterRegistry) -> None:
    engine = _engine(registry)
    engine.run(1, 2)
    assert engine.failure == ((), None)


def test_failure_path_of_a_nested_difference(registry: EqualityTesterRegistry) -> None:
    engine = _engine(registry, show_difference=True)
    engine.run({"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}]})
    assert engine.failure == (("a", 1, "b"), "2 vs 3")


def test_failure_path_of_a_container_that_fails(registry: EqualityTesterRegistry) -> None:
    engine = _engine(registry, show_difference=True)
    engine.run({"a": [1, 2]}, {"a": [1]})
    assert engine.failure == (("a",), "lengths differ")


def test_failure_path_of_a_container_that_fails_at_the_root(
    registry: EqualityTesterRegistry,
) -> None:
    engine = _engine(registry, show_difference=True)
    engine.run([1, 2], [1])
    assert engine.failure == ((), "lengths differ")


def test_failure_path_parts_of_fail_are_added(registry: EqualityTesterRegistry) -> None:
    engine = _engine(registry, show_difference=True)
    engine.run({"w": (1, 2)}, {"w": (1, 3)})
    assert engine.failure == (("w", 1, 2), "values differ")


def test_failure_types_differ_reason(registry: EqualityTesterRegistry) -> None:
    engine = _engine(registry, show_difference=True)
    engine.run({"a": 1}, {"a": 1.0})
    assert engine.failure == (("a",), "types differ: int vs float")


def test_failure_of_a_legacy_tester_has_no_reason() -> None:
    reg = EqualityTesterRegistry()
    reg.register(object, LegacyTester())
    engine = Engine(reg, EqualityConfig(registry=reg))
    engine.run(1, 2)
    assert engine.failure == ((), None)


#####################################
#     Tests lazy reasons / cache    #
#####################################


def test_reason_is_not_built_when_not_explaining(registry: EqualityTesterRegistry) -> None:
    calls = []
    registry.register(
        object,
        FuncTester(lambda _a, _e, ctx: ctx.fail(lambda: calls.append(1) or "reason")),
        exist_ok=True,
    )
    _engine(registry).run(1, 2)
    assert calls == []


def test_reason_is_built_when_explaining(registry: EqualityTesterRegistry) -> None:
    calls = []
    registry.register(
        object,
        FuncTester(lambda _a, _e, ctx: ctx.fail(lambda: calls.append(1) or "reason")),
        exist_ok=True,
    )
    _engine(registry, show_difference=True).run(1, 2)
    assert calls == [1]


def test_explaining_with_a_reporter(registry: EqualityTesterRegistry) -> None:
    assert _engine(registry, reporter=lambda *_: None).explaining


def test_not_explaining_by_default(registry: EqualityTesterRegistry) -> None:
    assert not _engine(registry).explaining


def test_find_looks_the_registry_up_once_per_type() -> None:
    calls = []

    class SpyRegistry(EqualityTesterRegistry):
        def find_equality_tester(self, data_type: type) -> BaseEqualityTester[Any]:
            calls.append(data_type)
            return super().find_equality_tester(data_type)

    registry = SpyRegistry()
    registry.register(object, LeafTester())
    registry.register(list, ListTester())
    _engine(registry).run([1, 2, 3], [1, 2, 4])
    assert calls == [list, int]


def test_engine_reads_the_settings_of_the_config(registry: EqualityTesterRegistry) -> None:
    engine = _engine(registry, atol=0.5, rtol=0.25, equal_nan=True)
    assert (engine.atol, engine.rtol, engine.equal_nan) == (0.5, 0.25, True)


###########################
#     Tests warn_once     #
###########################


def test_warn_once_emits_a_warning(registry: EqualityTesterRegistry) -> None:
    with pytest.warns(RuntimeWarning, match="meow"):
        assert _engine(registry).warn_once("key", "meow")


def test_warn_once_emits_once(registry: EqualityTesterRegistry) -> None:
    engine = _engine(registry)
    with pytest.warns(RuntimeWarning):
        engine.warn_once("key", "meow")
    assert not engine.warn_once("key", "meow")


def test_warn_once_different_keys(registry: EqualityTesterRegistry) -> None:
    engine = _engine(registry)
    with pytest.warns(RuntimeWarning):
        engine.warn_once("a", "meow")
    with pytest.warns(RuntimeWarning):
        assert engine.warn_once("b", "meow")


######################
#     Tests Element  #
######################


def test_element_repr() -> None:
    assert repr(Element(2)) == "Element(value=2)"


def test_element_equal() -> None:
    assert Element(2) == Element(2)


def test_element_is_hashable() -> None:
    assert len({Element(2), Element(2)}) == 1
