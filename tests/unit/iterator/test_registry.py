from __future__ import annotations

from dataclasses import dataclass

import pytest

from coola.iterator import (
    ChildFinderRegistry,
    DataclassChildFinder,
    DefaultChildFinder,
    IterableChildFinder,
    MappingChildFinder,
)
from coola.registry import DataclassMarker, TypeNotRegisteredError


@dataclass
class Point:
    x: int


class CustomList(list):
    r"""Create a custom class that inherits from list."""


@pytest.fixture
def registry() -> ChildFinderRegistry:
    return ChildFinderRegistry(
        {
            object: DefaultChildFinder(),
            list: IterableChildFinder(),
            dict: MappingChildFinder(),
            DataclassMarker: DataclassChildFinder(),
        }
    )


#########################################
#     Tests for ChildFinderRegistry     #
#########################################


def test_child_finder_registry_repr_contains_the_class_name() -> None:
    assert repr(ChildFinderRegistry()).startswith("ChildFinderRegistry(")


def test_child_finder_registry_has_child_finder_true(registry: ChildFinderRegistry) -> None:
    assert registry.has_child_finder(list)


def test_child_finder_registry_has_child_finder_false(registry: ChildFinderRegistry) -> None:
    assert not registry.has_child_finder(tuple)


def test_child_finder_registry_find_child_finder_exact_type(
    registry: ChildFinderRegistry,
) -> None:
    assert type(registry.find_child_finder(list)) is IterableChildFinder


def test_child_finder_registry_find_child_finder_uses_the_mro(
    registry: ChildFinderRegistry,
) -> None:
    assert type(registry.find_child_finder(CustomList)) is IterableChildFinder


def test_child_finder_registry_find_child_finder_falls_back_to_object(
    registry: ChildFinderRegistry,
) -> None:
    assert type(registry.find_child_finder(tuple)) is DefaultChildFinder


def test_child_finder_registry_find_child_finder_dataclass(registry: ChildFinderRegistry) -> None:
    assert type(registry.find_child_finder(Point)) is DataclassChildFinder


def test_child_finder_registry_find_child_finder_registered_type_has_priority_over_marker(
    registry: ChildFinderRegistry,
) -> None:
    registry.register(Point, DefaultChildFinder())
    assert type(registry.find_child_finder(Point)) is DefaultChildFinder


def test_child_finder_registry_find_child_finder_not_registered_raises() -> None:
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        ChildFinderRegistry().find_child_finder(int)


def test_child_finder_registry_copy_is_independent(registry: ChildFinderRegistry) -> None:
    copy = registry.copy()
    copy.register(tuple, IterableChildFinder())
    assert not registry.has_child_finder(tuple)
