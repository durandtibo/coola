r"""Implement equality testers for JAX arrays.

This module provides equality testers for jax.numpy.ndarray with support
for NaN equality, tolerance-based comparisons, and dtype/shape checking.
"""

from __future__ import annotations

__all__ = ["JaxArrayEqualityTester"]

from typing import TYPE_CHECKING, Any

from coola.equality.handler import (
    JaxArrayEqualHandler,
    SameDTypeHandler,
    SameObjectHandler,
    SameShapeHandler,
    SameTypeHandler,
    create_chain,
)
from coola.equality.tester.chain import BaseChainEqualityTester
from coola.utils.imports import check_jax, is_jax_available

if TYPE_CHECKING or is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    from coola.utils.fallback.jax import jnp


if TYPE_CHECKING:
    from coola.equality.handler import BaseEqualityHandler
    from coola.equality.tester.base import BaseEqualityTester


class JaxArrayEqualityTester(BaseChainEqualityTester[jnp.ndarray]):
    r"""Implement an equality tester for ``jax.numpy.ndarray``.

    This tester compares JAX arrays element-wise with support for NaN equality
    and tolerance-based comparisons. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are JAX arrays
    3. SameDTypeHandler: Check arrays have the same dtype
    4. SameShapeHandler: Verify arrays have the same shape
    5. JaxArrayEqualHandler: Element-wise comparison with tolerance support

    The tester respects config.equal_nan for NaN comparisons and config.atol/rtol
    for floating-point tolerance.

    Example:
        Basic array comparison:

        ```pycon
        >>> import jax.numpy as jnp
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import JaxArrayEqualityTester
        >>> config = EqualityConfig()
        >>> tester = JaxArrayEqualityTester()
        >>> tester.objects_are_equal(jnp.ones((2, 3)), jnp.ones((2, 3)), config)
        True
        >>> tester.objects_are_equal(jnp.ones((2, 3)), jnp.zeros((2, 3)), config)
        False

        ```
    """

    def _check_dependencies(self) -> None:
        check_jax()

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameDTypeHandler(),
            SameShapeHandler(),
            JaxArrayEqualHandler(),
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
