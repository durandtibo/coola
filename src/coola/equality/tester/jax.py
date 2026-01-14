r"""Implement equality testers for JAX arrays.

This module provides equality testers for jax.numpy.ndarray with support
for NaN equality, tolerance-based comparisons, and dtype/shape checking.
"""

from __future__ import annotations

__all__ = ["JaxArrayEqualityTester"]

from functools import lru_cache
from typing import TYPE_CHECKING

from coola.equality.handler import (
    JaxArrayEqualHandler,
    SameDTypeHandler,
    SameObjectHandler,
    SameShapeHandler,
    SameTypeHandler,
)
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_jax, is_jax_available

if TYPE_CHECKING or is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    from coola.utils.fallback.jax import jnp


if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class JaxArrayEqualityTester(BaseEqualityTester[jnp.ndarray]):
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

    def __init__(self) -> None:
        check_jax()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameDTypeHandler()).chain(
            SameShapeHandler()
        ).chain(JaxArrayEqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: jnp.ndarray,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


@lru_cache(maxsize=1)
def get_array_impl_class() -> type:
    r"""Get the JAX array implementation class.

    JAX uses different concrete array implementations depending on the backend
    and compilation status. This function creates a simple array and returns
    its class, which is then registered in the equality tester registry.

    Returns:
        The JAX array implementation class (e.g., ArrayImpl).

    Note:
        This function is cached to avoid creating arrays repeatedly. The result
        is used to register the concrete JAX array type in the default registry.
    """
    return jnp.ones(1).__class__
