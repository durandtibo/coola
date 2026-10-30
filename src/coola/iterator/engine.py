r"""Define the engines that walk nested data in depth-first and breadth-
first order.

The engines own the invariants of a traversal (the leaf rule, the
cycles, the depth, the order and the laziness). A child finder only
describes the shape of a type, so both orders visit the same leaves. The
state of a traversal (stack or queue, cache, ancestors) is local to the
call.
"""

from __future__ import annotations

__all__ = ["breadth_first", "depth_first", "depth_first_items"]

from collections import deque
from itertools import chain
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from coola.iterator.registry import ChildFinderRegistry


def _get_method(finder: Any, method: str) -> Callable[[Any], Iterable[Any] | None]:
    r"""Get a method of a child finder.

    A child finder that follows the ``ChildFinder`` protocol has no
    ``find_items`` method: its children are numbered.

    Args:
        finder: The child finder.
        method: The name of the method: ``"find_children"`` or
            ``"find_items"``.

    Returns:
        The method of the finder, or a function that numbers the
            children if the finder has no ``find_items`` method.
    """
    find = getattr(finder, method, None)
    if find is not None:
        return find

    def find_items(data: Any) -> Iterable[Any] | None:
        r"""Return the items of an object, as ``(key, child)`` pairs.

        Args:
            data: The object.

        Returns:
            The ``(index, child)`` pairs, or ``None`` for a leaf.
        """
        children = finder.find_children(data)
        return None if children is None else enumerate(children)

    return find_items


def _make_find(
    registry: ChildFinderRegistry,
    cache: dict[type, Callable[[Any], Iterable[Any] | None]],
    method: str = "find_children",
) -> Callable[[Any], Iterable[Any] | None]:
    r"""Create a function that finds the children of an object.

    The child finder of a type is resolved once per type and per call.
    A registration made during the traversal is seen by the next call.

    Args:
        registry: The registry used to resolve the child finders.
        cache: The method of the types already resolved in this
            call. It is filled by the returned function.
        method: The name of the method of the child finder to use,
            ``"find_children"`` or ``"find_items"``.

    Returns:
        A function that returns the children of an object, or ``None``
            for a leaf.
    """

    def find(data: Any) -> Iterable[Any] | None:
        r"""Find the children or the items of an object.

        Args:
            data: The object.

        Returns:
            The children or the items, or ``None`` for a leaf.
        """
        data_type = type(data)
        try:
            find_children = cache[data_type]
        except KeyError:
            finder = registry.find_child_finder(data_type)
            find_children = cache[data_type] = _get_method(finder, method)
        return find_children(data)

    return find


def _cycle_error(data: object) -> ValueError:
    r"""Create the error of a cycle.

    Args:
        data: The object that contains itself.

    Returns:
        The error to raise.
    """
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


_END = object()


def _peek(items: Iterable[Any]) -> Iterator[Any] | None:
    r"""Return an iterator over the items, or ``None`` if there is no
    item.

    Args:
        items: The items.

    Returns:
        An iterator over all the items, or ``None`` if there is no item.
    """
    iterator = iter(items)
    first = next(iterator, _END)
    if first is _END:
        return None
    return chain((first,), iterator)


def depth_first_items(
    data: object, registry: ChildFinderRegistry, *, empty: bool = False
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    r"""Yield the leaves of nested data with their path, in depth-first
    (pre-order) order.

    It visits the same leaves, in the same order, as ``depth_first``.
    The path is the tuple of the labels of the containers to cross to
    reach the leaf.

    Args:
        data: The nested data to traverse.
        registry: The registry used to resolve the child finders.
        empty: If ``True``, an empty container is also yielded, as a
            leaf.

    Yields:
        The pairs ``(path, leaf)`` of ``data``, once per occurrence.

    Raises:
        ValueError: If a container contains itself.
    """
    find = _make_find(registry, {}, "find_items")
    items = find(data)
    iterator = None if items is None else _peek(items)
    if iterator is None:
        # A leaf, or an empty container
        if items is None or empty:
            yield (), data
        return
    # The containers on the path are kept in the stack, so their ids cannot be reused.
    ancestors = {id(data)}
    stack = [(data, iterator)]
    # ``labels[i]`` is the label of the child of ``stack[i]`` that is being visited.
    labels: list[Any] = []
    while stack:
        owner, iterator = stack[-1]
        for label, item in iterator:
            children = find(item)
            if children is None:
                yield (*labels, label), item
                continue
            if id(item) in ancestors:
                raise _cycle_error(item)
            iterator = _peek(children)
            if iterator is None:
                if empty:
                    yield (*labels, label), item
                continue
            ancestors.add(id(item))
            labels.append(label)
            stack.append((item, iterator))
            break
        else:
            stack.pop()
            ancestors.discard(id(owner))
            if labels:
                labels.pop()
