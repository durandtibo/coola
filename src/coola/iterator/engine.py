r"""Define the engines that walk nested data in depth-first and
breadth-first order.

The engines own the invariants of a traversal (the leaf rule, the
cycles, the depth, the order and the laziness). A child finder only
describes the shape of a type, so both orders visit the same leaves.
The state of a traversal (stack or queue, cache, ancestors) is local to
the call.
"""

from __future__ import annotations

__all__ = ["breadth_first", "depth_first"]

from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from coola.iterator.registry import ChildFinderRegistry


def _make_find(
    registry: ChildFinderRegistry,
    cache: dict[type, Callable[[Any], Iterable[Any] | None]],
) -> Callable[[Any], Iterable[Any] | None]:
    r"""Create a function that finds the children of an object.

    The child finder of a type is resolved once per type and per call.
    A registration made during the traversal is seen by the next call.

    Args:
        registry: The registry used to resolve the child finders.
        cache: The ``find_children`` method of the types already
            resolved in this call. It is filled by the returned
            function.

    Returns:
        A function that returns the children of an object, or ``None``
            for a leaf.
    """

    def find(data: Any) -> Iterable[Any] | None:
        data_type = type(data)
        try:
            find_children = cache[data_type]
        except KeyError:
            find_children = cache[data_type] = registry.find_child_finder(data_type).find_children
        return find_children(data)

    return find


def _cycle_error(data: object) -> ValueError:
    return ValueError(f"Cycle detected: an object of type {type(data)} contains itself")


def depth_first(data: object, registry: ChildFinderRegistry) -> Iterator[Any]:
    r"""Yield the leaves of nested data in depth-first (pre-order) order.

    The traversal uses an explicit stack, so the depth is bounded by
    the memory only.

    Args:
        data: The nested data to traverse.
        registry: The registry used to resolve the child finders.

    Yields:
        The leaves of ``data``, once per occurrence.

    Raises:
        ValueError: If a container contains itself.
    """
    # The root is resolved before the cache is built: a leaf or an empty container is common and
    # the cache is not needed for it. The finder of the root is then put in the cache.
    find_root = registry.find_child_finder(type(data)).find_children
    children = find_root(data)
    if children is None:
        yield data
        return
    find = _make_find(registry, {type(data): find_root})
    # The containers on the path are kept in the stack, so their ids cannot be reused.
    ancestors = {id(data)}
    stack = [(data, iter(children))]
    while stack:
        owner, iterator = stack[-1]
        for item in iterator:
            children = find(item)
            if children is None:
                yield item
                continue
            if id(item) in ancestors:
                raise _cycle_error(item)
            ancestors.add(id(item))
            stack.append((item, iter(children)))
            break
        else:
            stack.pop()
            ancestors.discard(id(owner))


def breadth_first(data: object, registry: ChildFinderRegistry) -> Iterator[Any]:
    r"""Yield the leaves of nested data in breadth-first (level) order.

    Args:
        data: The nested data to traverse.
        registry: The registry used to resolve the child finders.

    Yields:
        The leaves of ``data``, once per occurrence.

    Raises:
        ValueError: If a container contains itself.
    """
    # The root is resolved before the cache is built: a leaf or an empty container is common and
    # the cache is not needed for it. The finder of the root is then put in the cache.
    find_root = registry.find_child_finder(type(data)).find_children
    children = find_root(data)
    if children is None:
        yield data
        return
    find = _make_find(registry, {type(data): find_root})
    # Each item comes with the chain of its ancestors: ``(container, parent chain)``. The
    # containers are kept, not their ids, so an id cannot be reused by a temporary object.
    chain = (data, None)
    queue: deque[tuple[Any, Any]] = deque((child, chain) for child in children)
    # A container can be its own ancestor only if it is expanded twice, so the (linear)
    # walk over the chain is needed only for the ids already expanded.
    expanded = {id(data)}
    while queue:
        item, chain = queue.popleft()
        children = find(item)
        if children is None:
            yield item
            continue
        if id(item) in expanded:
            node = chain
            while node is not None:
                if node[0] is item:
                    raise _cycle_error(item)
                node = node[1]
        expanded.add(id(item))
        chain = (item, chain)
        queue.extend((child, chain) for child in children)
