r"""Define the engine that applies a function to the leaves of nested
data.

The engine owns the invariants of the traversal (type dispatch, leaf
call, cycle detection, depth and order). A transformer only describes
the shape of a type: its children and how to rebuild it.
"""

from __future__ import annotations

__all__ = ["transform"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

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
