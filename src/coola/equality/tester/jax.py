r"""Implement equality testers for JAX arrays.

This module provides equality testers for jax.numpy.ndarray with support
for NaN equality, tolerance-based comparisons, and dtype/shape checking.
"""

from __future__ import annotations

__all__ = ["JaxArrayEqualityTester"]

from typing import TYPE_CHECKING, Any

from coola.equality._helpers.format import format_value_difference
from coola.equality._helpers.jax import array_close, array_equal
from coola.equality.tester._array import check_dtype_and_shape
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_jax, is_jax_available

if TYPE_CHECKING or is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    from coola.utils.fallback.jax import jnp


if TYPE_CHECKING:
    from coola.equality.engine import Engine


class JaxArrayEqualityTester(BaseEqualityTester[jnp.ndarray]):
    r"""Implement an equality tester for ``jax.numpy.ndarray``.

    Two arrays are equal if they have the same data type, the same
    shape and equal elements. The numbers are compared with the
    tolerance and ``equal_nan`` of the comparison, and a boolean array
    is exact. The location of a difference is the index of the first
    element that differs. Without 64-bit mode, JAX has no ``float64``:
    the tolerance is computed in the widest dtype JAX has.

    Raises:
        MissingPackageError: If ``jax`` is not installed.

    Example:
        ```pycon
        >>> import jax.numpy as jnp
        >>> from coola.equality import compare
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import JaxArrayEqualityTester, get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> tester = JaxArrayEqualityTester()
        >>> tester.objects_are_equal(jnp.ones((2, 3)), jnp.ones((2, 3)), engine)
        True
        >>> tester.objects_are_equal(jnp.ones((2, 3)), jnp.zeros((2, 3)), engine)
        False
        >>> compare(jnp.ones((2, 3)), jnp.zeros((2, 3))).path
        (0, 0)

        ```
    """

    def __init__(self) -> None:
        check_jax()

    def objects_are_equal(self, actual: jnp.ndarray, expected: jnp.ndarray, ctx: Engine) -> bool:
        r"""Test two JAX arrays.

        The arrays must have the same dtype and shape; the elements are
        compared with the tolerance and ``equal_nan``.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the objects are equal, otherwise ``False`` (the
                reason is recorded in the engine).
        """
        failure = check_dtype_and_shape(actual, expected, ctx)
        if failure is not None:
            return failure
        if array_equal(actual, expected, ctx.config):
            return True
        if not ctx.explaining:
            return False
        # The index is looked up only when the difference is explained.
        index = tuple(int(i) for i in jnp.argwhere(~array_close(actual, expected, ctx.config))[0])
        return ctx.fail(
            lambda: format_value_difference(actual[index], expected[index], name="values"), *index
        )


def get_default_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the default equality testers of the JAX arrays.

    Returns:
        The equality testers of the JAX arrays and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the equality testers of the JAX arrays.
    """
    if not is_jax_available():
        return {}

    tester = JaxArrayEqualityTester()
    # ``jnp.ndarray`` is ``jax.Array``, which is in the MRO of the concrete array classes.
    return {jnp.ndarray: tester}
