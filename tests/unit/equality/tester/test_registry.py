from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    BaseEqualityTester,
    DataclassEqualityTester,
    DefaultEqualityTester,
    EqualityTesterRegistry,
    MappingEqualityTester,
    ScalarEqualityTester,
    SequenceEqualityTester,
)
from coola.equality.tolerance import Tolerance
from coola.registry import AlreadyRegisteredError
from coola.utils.introspection import has_generated_dataclass_eq


class CustomList(list):
    r"""Create a custom class that inherits from list."""


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


############################################
#     Tests for EqualityTesterRegistry     #
############################################


def test_equality_tester_registry_init_empty() -> None:
    registry = EqualityTesterRegistry()
    assert len(registry._state) == 0


def test_equality_tester_registry_init_with_registry() -> None:
    registry = EqualityTesterRegistry({list: SequenceEqualityTester()})
    assert list in registry._state


def test_equality_tester_registry_init_with_registry_keeps_the_tester() -> None:
    tester = SequenceEqualityTester()
    registry = EqualityTesterRegistry({list: tester})
    assert registry._state[list] is tester


def test_equality_tester_registry_init_with_registry_copies_the_registry() -> None:
    initial_registry: dict[type, BaseEqualityTester[Any]] = {list: SequenceEqualityTester()}
    registry = EqualityTesterRegistry(initial_registry)
    initial_registry[tuple] = DefaultEqualityTester()
    assert tuple not in registry._state


def test_equality_tester_registry_repr() -> None:
    assert repr(EqualityTesterRegistry()).startswith("EqualityTesterRegistry(")


def test_equality_tester_registry_str() -> None:
    assert str(EqualityTesterRegistry()).startswith("EqualityTesterRegistry(")


def test_equality_tester_registry_register_new_type() -> None:
    registry = EqualityTesterRegistry()
    registry.register(list, SequenceEqualityTester())
    assert registry.has_equality_tester(list)


def test_equality_tester_registry_register_new_type_keeps_the_tester() -> None:
    registry = EqualityTesterRegistry()
    tester = SequenceEqualityTester()
    registry.register(list, tester)
    assert registry._state[list] is tester


def test_equality_tester_registry_register_existing_type_without_exist_ok() -> None:
    registry = EqualityTesterRegistry()
    tester1 = SequenceEqualityTester()
    tester2 = DefaultEqualityTester()
    registry.register(list, tester1)
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register(list, tester2, exist_ok=False)


def test_equality_tester_registry_register_existing_type_with_exist_ok() -> None:
    registry = EqualityTesterRegistry()
    tester1 = SequenceEqualityTester()
    tester2 = DefaultEqualityTester()

    registry.register(list, tester1)
    registry.register(list, tester2, exist_ok=True)

    assert registry._state[list] is tester2


def _register_many() -> EqualityTesterRegistry:
    registry = EqualityTesterRegistry()
    registry.register_many(
        {
            list: SequenceEqualityTester(),
            dict: MappingEqualityTester(),
            set: DefaultEqualityTester(),
        }
    )
    return registry


@pytest.mark.parametrize("dtype", [list, dict, set])
def test_equality_tester_registry_register_many(dtype: type) -> None:
    assert _register_many().has_equality_tester(dtype)


def test_equality_tester_registry_register_many_with_existing_type() -> None:
    registry = EqualityTesterRegistry({list: SequenceEqualityTester()})
    testers: dict[type, BaseEqualityTester[Any]] = {
        list: MappingEqualityTester(),
        dict: MappingEqualityTester(),
    }
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register_many(testers, exist_ok=False)


def test_equality_tester_registry_register_many_with_exist_ok() -> None:
    registry = EqualityTesterRegistry()
    registry.register(list, SequenceEqualityTester())

    tester = DefaultEqualityTester()
    testers: dict[type, BaseEqualityTester[Any]] = {
        list: tester,
        dict: MappingEqualityTester(),
    }

    registry.register_many(testers, exist_ok=True)
    assert registry._state[list] is tester


def test_equality_tester_registry_has_equality_tester_true() -> None:
    assert EqualityTesterRegistry({list: SequenceEqualityTester()}).has_equality_tester(list)


def test_equality_tester_registry_has_equality_tester_false() -> None:
    assert not EqualityTesterRegistry().has_equality_tester(list)


def test_equality_tester_registry_find_equality_tester_direct_match() -> None:
    tester = SequenceEqualityTester()
    registry = EqualityTesterRegistry({list: tester})
    assert registry.find_equality_tester(list) is tester


def test_equality_tester_registry_find_equality_tester_mro_lookup() -> None:
    tester = SequenceEqualityTester()
    registry = EqualityTesterRegistry({list: tester})
    assert registry.find_equality_tester(CustomList) is tester


def test_equality_tester_registry_find_equality_tester_most_specific() -> None:
    tester = MappingEqualityTester()
    registry = EqualityTesterRegistry(
        {
            object: DefaultEqualityTester(),
            list: SequenceEqualityTester(),
            CustomList: tester,
        }
    )

    assert registry.find_equality_tester(CustomList) is tester


def test_equality_tester_registry_objects_are_equal_true(config: EqualityConfig) -> None:
    registry = EqualityTesterRegistry(
        {object: DefaultEqualityTester(), list: SequenceEqualityTester()}
    )
    assert registry.objects_are_equal([1, 2, 3], [1, 2, 3], config)


def test_equality_tester_registry_objects_are_equal_true_atol() -> None:
    registry = EqualityTesterRegistry(
        {
            object: DefaultEqualityTester(),
            list: SequenceEqualityTester(),
            int: ScalarEqualityTester(),
        }
    )
    config = EqualityConfig(tolerance=Tolerance(atol=1.1))
    assert registry.objects_are_equal([1, 2, 3], [0, 1, 2], config)


def test_equality_tester_registry_objects_are_equal_false(config: EqualityConfig) -> None:
    registry = EqualityTesterRegistry(
        {object: DefaultEqualityTester(), list: SequenceEqualityTester()}
    )
    assert not registry.objects_are_equal([1, 2, 3], [1, 1, 1], config)


def test_equality_tester_registry_registry_isolation() -> None:
    tester1 = SequenceEqualityTester()
    registry1 = EqualityTesterRegistry({list: tester1})
    EqualityTesterRegistry({list: MappingEqualityTester()})
    assert registry1._state[list] is tester1


def test_equality_tester_registry_registry_isolation_other_registry() -> None:
    tester2 = MappingEqualityTester()
    EqualityTesterRegistry({list: SequenceEqualityTester()})
    registry2 = EqualityTesterRegistry({list: tester2})
    assert registry2._state[list] is tester2


@dataclass
class _Point:
    x: float


@dataclass(eq=False)
class _NoEq:
    x: float


@dataclass
class _CustomEq:
    x: float

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _CustomEq)

    def __hash__(self) -> int:
        return 0


def _make_registry() -> EqualityTesterRegistry:
    registry = EqualityTesterRegistry({object: DefaultEqualityTester()})
    registry.register_fallback(has_generated_dataclass_eq, DataclassEqualityTester())
    return registry


def test_equality_tester_registry_find_equality_tester_dataclass_custom_eq() -> None:
    assert isinstance(_make_registry().find_equality_tester(_CustomEq), DefaultEqualityTester)


def test_equality_tester_registry_find_equality_tester_dataclass() -> None:
    assert isinstance(_make_registry().find_equality_tester(_Point), DataclassEqualityTester)


def test_equality_tester_registry_find_equality_tester_dataclass_registered_type() -> None:
    tester = SequenceEqualityTester()
    registry = _make_registry()
    registry.register(_Point, tester)
    assert registry.find_equality_tester(_Point) is tester


def test_equality_tester_registry_find_equality_tester_dataclass_eq_false() -> None:
    assert isinstance(_make_registry().find_equality_tester(_NoEq), DefaultEqualityTester)


def test_equality_tester_registry_find_equality_tester_dataclass_no_fallback() -> None:
    registry = EqualityTesterRegistry({object: DefaultEqualityTester()})
    assert isinstance(registry.find_equality_tester(_Point), DefaultEqualityTester)


def test_equality_tester_registry_find_equality_tester_dataclass_with_registered_base() -> None:
    """Regression test: a tester shared by ``object`` and a base class
    is used for a dataclass that inherits from the base class."""

    class Base:
        pass

    @dataclass
    class Derived(Base):
        x: int = 1

    shared = DefaultEqualityTester()
    registry = EqualityTesterRegistry({object: shared, Base: shared})
    registry.register_fallback(has_generated_dataclass_eq, DataclassEqualityTester())
    assert registry.find_equality_tester(Derived) is shared


def test_equality_tester_registry_find_equality_tester_not_dataclass() -> None:
    assert isinstance(_make_registry().find_equality_tester(int), DefaultEqualityTester)
