from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import objects_are_equal
from coola.iterator.bfs import (
    BaseChildFinder,
    bfs_iterate,
    get_default_registry,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


class LinkedListNode:
    r"""A simple linked list node for testing custom iterators.

    This class represents a node in a singly-linked list structure,
    used to test that custom iterators can be registered and work
    correctly with the BFS traversal system.

    Args:
        value: The value stored in this node. Can be any type,
            including numpy arrays or nested structures.
        next_node: The next node in the linked list, or ``None``
            if this is the last node.
    """

    def __init__(self, value: Any, next_node: LinkedListNode | None = None) -> None:
        self.value = value
        self.next = next_node

    def __repr__(self) -> str:
        return f"LinkedListNode({self.value})"


class LinkedListChildFinder(BaseChildFinder):
    r"""Iterator for traversing linked list structures in BFS.

    This iterator handles LinkedListNode objects by exposing both the
    node's value and the next node as children. This allows BFS to
    process the linked list level by level, exploring values before
    moving to the next nodes.

    The iterator returns the current node's value first, followed by the
    next node (if it exists). This ensures that arrays stored in values
    are found before moving deeper into the list.
    """

    def find_children(self, data: LinkedListNode) -> Iterator[Any]:
        yield data.value
        if data.next is not None:
            yield data.next


def test_bfs_iterate_with_linked_list() -> None:
    # Test custom iterator for linked list
    get_default_registry().register(LinkedListNode, LinkedListChildFinder())

    # Create linked list: [1] -> [2, 3] -> [4]
    node3 = LinkedListNode(4)
    node2 = LinkedListNode([2, 3], node3)
    node1 = LinkedListNode(1, node2)

    assert objects_are_equal(list(bfs_iterate(node1)), [1, 2, 3, 4])


def test_bfs_iterate_performance_wide_shallow() -> None:
    # Performance test: wide structure (many items at same level)
    assert objects_are_equal(list(bfs_iterate(list(range(1000)))), list(range(1000)))


def test_bfs_iterate_performance_narrow_deep() -> None:
    # Performance test: deep nesting
    data = 1
    for _ in range(100):
        data = [data]
    assert objects_are_equal(list(bfs_iterate(data)), [1])
