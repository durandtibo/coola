r"""Define helpers shared by the tests of ``coola.recursive``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coola.recursive.base import Expansion, Transformer

__all__ = ["expand_not_none"]


def expand_not_none(transformer: Transformer[Any], data: Any) -> Expansion:
    r"""Expand ``data`` with a transformer that must describe its children.

    Args:
        transformer: The transformer to use.
        data: The data to expand.

    Returns:
        The ``(children, rebuild)`` pair returned by
            ``transformer.expand(data)``.

    Raises:
        AssertionError: If the transformer treats ``data`` as a leaf.
    """
    expansion = transformer.expand(data)
    assert expansion is not None, f"{transformer!r} treats {data!r} as a leaf"
    return expansion
