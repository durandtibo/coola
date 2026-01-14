from __future__ import annotations

from typing import Any

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    BaseEqualityTester,
    DefaultEqualityTester,
    EqualityTesterRegistry,
    MappingEqualityTester,
    SequenceEqualityTester,
)


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
    tester = SequenceEqualityTester()
    initial_registry: dict[type, BaseEqualityTester[Any]] = {list: tester}
    registry = EqualityTesterRegistry(initial_registry)

    assert list in registry._state
    assert registry._state[list] is tester
    # Verify it's a copy
    initial_registry[tuple] = DefaultEqualityTester()
    assert tuple not in registry._state


def test_equality_tester_registry_repr() -> None:
    assert repr(EqualityTesterRegistry()).startswith("EqualityTesterRegistry(")


def test_equality_tester_registry_str() -> None:
    assert str(EqualityTesterRegistry()).startswith("EqualityTesterRegistry(")


def test_equality_tester_registry_register_new_type() -> None:
    registry = EqualityTesterRegistry()
    tester = SequenceEqualityTester()
    registry.register(list, tester)
    assert registry.has_equality_tester(list)
    assert registry._state[list] is tester


def test_equality_tester_registry_register_existing_type_without_exist_ok() -> None:
    registry = EqualityTesterRegistry()
    tester1 = SequenceEqualityTester()
    tester2 = DefaultEqualityTester()
    registry.register(list, tester1)
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register(list, tester2, exist_ok=False)


def test_equality_tester_registry_register_existing_type_with_exist_ok() -> None:
    registry = EqualityTesterRegistry()
    tester1 = SequenceEqualityTester()
    tester2 = DefaultEqualityTester()

    registry.register(list, tester1)
    registry.register(list, tester2, exist_ok=True)

    assert registry._state[list] is tester2


def test_equality_tester_registry_register_many() -> None:
    registry = EqualityTesterRegistry()
    registry.register_many(
        {
            list: SequenceEqualityTester(),
            dict: MappingEqualityTester(),
            set: DefaultEqualityTester(),
        }
    )
    assert registry.has_equality_tester(list)
    assert registry.has_equality_tester(dict)
    assert registry.has_equality_tester(set)


def test_equality_tester_registry_register_many_with_existing_type() -> None:
    registry = EqualityTesterRegistry({list: SequenceEqualityTester()})
    testers = {
        list: MappingEqualityTester(),
        dict: MappingEqualityTester(),
    }
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register_many(testers, exist_ok=False)


def test_equality_tester_registry_register_many_with_exist_ok() -> None:
    registry = EqualityTesterRegistry()
    registry.register(list, SequenceEqualityTester())

    tester = DefaultEqualityTester()
    testers = {
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
        {object: DefaultEqualityTester(), list: SequenceEqualityTester()}
    )
    config = EqualityConfig(atol=1.1)
    assert registry.objects_are_equal([1, 2, 3], [0, 1, 2], config)


def test_equality_tester_registry_objects_are_equal_false(config: EqualityConfig) -> None:
    registry = EqualityTesterRegistry(
        {object: DefaultEqualityTester(), list: SequenceEqualityTester()}
    )
    assert not registry.objects_are_equal([1, 2, 3], [1, 1, 1], config)


def test_equality_tester_registry_registry_isolation() -> None:
    tester1 = SequenceEqualityTester()
    tester2 = MappingEqualityTester()

    registry1 = EqualityTesterRegistry({list: tester1})
    registry2 = EqualityTesterRegistry({list: tester2})

    assert registry1._state[list] is tester1
    assert registry2._state[list] is tester2
