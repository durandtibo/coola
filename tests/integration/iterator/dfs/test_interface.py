from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola import objects_are_equal
from coola.iterator.dfs import (
    BaseIterator,
    IteratorRegistry,
    dfs_iterate,
    get_default_registry,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


class LinkedListNode:
    """A simple linked list node for testing custom iterators.

    Args:
        value: The value stored in this node. Can be any data type.
        next_node: Optional reference to the next node in the list.
    """

    def __init__(self, value: Any, next_node: LinkedListNode | None = None) -> None:
        self.value = value
        self.next = next_node


class LinkedListIterator(BaseIterator):
    """Custom iterator for traversing linked list structures.

    This iterator demonstrates how to extend the system to handle custom
    data structures. It traverses the linked list by following the
    'next' pointers and recursively searches each node's value for numpy
    arrays.
    """

    def iterate(self, data: LinkedListNode, registry: IteratorRegistry) -> Generator[np.ndarray]:
        current = data
        while current is not None:
            yield from registry.iterate(current.value)
            current = current.next


def test_dfs_iterate_with_custom_iterator() -> None:
    """Test that users can easily extend the system with custom data
    structures."""
    get_default_registry().register(LinkedListNode, LinkedListIterator())
    data = LinkedListNode(1, LinkedListNode(2, LinkedListNode([3, 4])))
    assert objects_are_equal(list(dfs_iterate(data)), [1, 2, 3, 4])


def test_dfs_iterate_performance_wide_shallow() -> None:
    # Performance test: wide structure (many items at same level)
    assert objects_are_equal(list(dfs_iterate(list(range(1000)))), list(range(1000)))


def test_dfs_iterate_performance_narrow_deep() -> None:
    # Performance test: deep nesting
    data = 1
    for _ in range(100):
        data = [data]
    assert objects_are_equal(list(dfs_iterate(data)), [1])
