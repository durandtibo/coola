r"""Define helpers shared by the tests."""

from __future__ import annotations

from typing import Any

__all__ = ["nest"]


def nest(leaf: Any, depth: int) -> Any:
    r"""Create deeply nested data.

    Args:
        leaf: The innermost value.
        depth: The number of lists that wrap ``leaf``. If ``0``, the
            leaf is returned unchanged.

    Returns:
        ``leaf`` wrapped in ``depth`` lists, for example
            ``nest(1, 2)`` returns ``[[1]]``.
    """
    data = leaf
    for _ in range(depth):
        data = [data]
    return data
