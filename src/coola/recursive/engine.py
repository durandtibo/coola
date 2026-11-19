r"""Define the engine that applies a function to the leaves of nested
data.

The engine owns the invariants of the traversal (type dispatch, leaf
call, cycle detection, depth and order). A transformer only describes
the shape of a type: its children and how to rebuild it.
"""

from __future__ import annotations

__all__ = ["transform", "transform_zip"]

from collections.abc import Mapping, Set as AbstractSet
from typing import TYPE_CHECKING, Any

from coola.utils.recursion import raise_recursion_limit_error

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from coola.recursive.registry import TransformerRegistry


def transform(data: object, func: Callable[[Any], Any], registry: TransformerRegistry) -> Any:
    r"""Apply a function to the leaves of nested data.

    The traversal uses an explicit stack, so the depth of the data is
    bounded by the memory only. The state (the stack, the cache of the
    transformers and the ancestors) is local to the call, so the calls
    are independent and can run in several threads.

    Args:
        data: The data to transform.
        func: The function applied to each leaf, once per occurrence.
        registry: The registry used to find the transformer of each type.
            The transformers found during the call are cached for the
            duration of the call.

    Returns:
        The transformed data, rebuilt with the transformed leaves.

    Raises:
        ValueError: If a container contains itself, directly or through
            other containers (a cycle).

    Example:
        ```pycon
        >>> from coola.recursive import get_default_registry
        >>> from coola.recursive.engine import transform
        >>> transform({"a": [1, 2], "b": 3}, str, get_default_registry())
        {'a': ['1', '2'], 'b': '3'}

        ```
    """
    find_transformer = registry.find_transformer
    data_type = type(data)
    transformer = find_transformer(data_type)
    expansion = transformer.expand(data)
    if expansion is None:
        return func(data)
    # map: type -> transformer, for this call only
    cache: dict[type, Any] = {data_type: transformer}
    key = id(data)
    ancestors = {key}  # the ids of the containers being transformed
    children, rebuild = expansion
    # A frame is [id of the object, rebuild, iterator over the children, transformed children].
    stack: list[list[Any]] = [[key, rebuild, iter(children), []]]
    while True:
        top = stack[-1]
        results = top[3]
        for child in top[2]:
            data_type = type(child)
            transformer = cache.get(data_type)
            if transformer is None:
                transformer = cache[data_type] = find_transformer(data_type)
            expansion = transformer.expand(child)
            if expansion is None:
                results.append(func(child))
                continue
            key = id(child)
            if key in ancestors:
                msg = f"Cycle detected: an object of type {data_type} contains itself"
                raise ValueError(msg)
            ancestors.add(key)
            children, rebuild = expansion
            stack.append([key, rebuild, iter(children), []])
            break
        else:  # the children are exhausted: rebuild and give the result to the parent
            stack.pop()
            ancestors.discard(top[0])
            result = top[1](results)
            if not stack:
                return result
            stack[-1][3].append(result)


def transform_zip(
    items: Sequence[object], func: Callable[[list[Any]], Any], registry: TransformerRegistry
) -> Any:
    r"""Apply a function to the leaves found at the same position in
    several nested structures.

    The structure, the container types and the order of the output
    follow the first item. The children of each item come from
    ``Transformer.expand``, so a transformer registered for a custom type
    is supported without change. The traversal is recursive: a structure
    nested about a thousand levels reaches the interpreter limit.

    Args:
        items: The structures to combine. They must have the same layout.
        func: The function applied to the list of the leaves found at
            the same position in every item, once per position.
        registry: The registry used to find the transformer of each type.

    Returns:
        The structure of the first item, where each leaf is the output
        of ``func``.

    Raises:
        ValueError: If ``items`` is empty, if the items do not have the
            same container kinds, keys or lengths, or if a container
            contains itself.
        TypeError: If a position holds sets, which have no order to align.
        RecursionError: If the structure is too deep.

    Example:
        ```pycon
        >>> from coola.recursive import get_default_registry
        >>> from coola.recursive.engine import transform_zip
        >>> transform_zip([{"a": 1, "b": [2, 3]}, {"a": 10, "b": [20, 30]}], sum, get_default_registry())
        {'a': 11, 'b': [22, 33]}

        ```
    """
    if len(items) == 0:
        msg = "items must contain at least one structure"
        raise ValueError(msg)
    try:
        return _zip(list(items), func, registry, set())
    except RecursionError as exc:
        raise_recursion_limit_error("transforming", exc)


def _zip(
    items: list[Any],
    func: Callable[[list[Any]], Any],
    registry: TransformerRegistry,
    ancestors: set[int],
) -> Any:
    r"""Apply a function to the leaves found at the same position.

    Args:
        items: The objects found at the same position in every structure.
        func: The function applied to the list of leaves.
        registry: The registry used to find the transformers.
        ancestors: The ids of the first-item containers being transformed.

    Returns:
        The rebuilt object, or the output of ``func`` for a leaf.

    Raises:
        ValueError: If the objects do not have the same layout, or if a
            container contains itself.
        TypeError: If the objects are sets.
    """
    maybe = [registry.find_transformer(type(item)).expand(item) for item in items]
    if maybe[0] is None:
        for index, expansion in enumerate(maybe):
            if expansion is not None:
                msg = (
                    f"item 0 is a leaf of type {type(items[0])} but item {index} is a "
                    f"container of type {type(items[index])}"
                )
                raise ValueError(msg)
        return func(items)
    expansions: list[Any] = []
    for index, expansion in enumerate(maybe):
        if expansion is None:
            msg = (
                f"item 0 is a container of type {type(items[0])} but item {index} is a "
                f"leaf of type {type(items[index])}"
            )
            raise ValueError(msg)
        expansions.append(expansion)
    if isinstance(items[0], AbstractSet):
        msg = "Sets cannot be combined: their elements have no position to align"
        raise TypeError(msg)
    _check_layout(items, expansions)
    key = id(items[0])
    if key in ancestors:
        msg = f"Cycle detected: an object of type {type(items[0])} contains itself"
        raise ValueError(msg)
    ancestors.add(key)
    children = [
        _zip(list(group), func, registry, ancestors)
        for group in zip(*(expansion[0] for expansion in expansions), strict=True)
    ]
    ancestors.discard(key)
    return expansions[0][1](children)


def _check_layout(items: list[Any], expansions: list[Any]) -> None:
    r"""Check that the containers found at the same position are aligned.

    Args:
        items: The containers found at the same position.
        expansions: The expansion of each container.

    Raises:
        ValueError: If a container is a mapping and another is not, if
            the mappings have different keys, or if the containers have
            different lengths.
    """
    first = items[0]
    for index, item in enumerate(items[1:], start=1):
        if isinstance(first, Mapping) != isinstance(item, Mapping):
            msg = (
                f"The containers must have the same kind but item 0 is a {type(first)} "
                f"and item {index} is a {type(item)}"
            )
            raise ValueError(msg)
        if isinstance(first, Mapping):
            if set(first) != set(item) or len(first) != len(item):
                msg = (
                    f"The mappings must have the same keys but item 0 has keys "
                    f"{list(first)} and item {index} has keys {list(item)}"
                )
                raise ValueError(msg)
            if list(first) != list(item):
                # Same keys in another order: reorder the values of the item by the first keys.
                expansions[index] = ([item[k] for k in first], expansions[index][1])
        elif len(expansions[index][0]) != len(expansions[0][0]):
            msg = (
                f"The containers must have the same length but item 0 has "
                f"{len(expansions[0][0])} elements and item {index} has "
                f"{len(expansions[index][0])}"
            )
            raise ValueError(msg)
