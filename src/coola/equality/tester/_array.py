r"""Implement the helpers shared by the testers of the arrays and the
tensors."""

from __future__ import annotations

__all__ = ["check_dtype_and_shape"]

from typing import TYPE_CHECKING, Any

from coola.equality._helpers.format import format_shape_difference

if TYPE_CHECKING:
    from coola.equality.engine import Engine


def check_dtype_and_shape(actual: Any, expected: Any, ctx: Engine) -> bool | None:
    r"""Check that two arrays have the same data type and shape.

    Args:
        actual: The actual array (or tensor).
        expected: The expected array, with the same type.
        ctx: The engine of the comparison.

    Returns:
        ``False`` (the difference is recorded with ``ctx.fail``) if the
            data types or the shapes are different, otherwise ``None``.
    """
    if actual.dtype != expected.dtype:
        return ctx.fail(
            lambda: f"objects have different data types: {actual.dtype} vs {expected.dtype}"
        )
    if actual.shape != expected.shape:
        return ctx.fail(lambda: format_shape_difference(actual.shape, expected.shape))
    return None
