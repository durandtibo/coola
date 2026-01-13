r"""Implement an equality tester for ``jax.numpy.ndarray``s."""

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

    Example:
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
    r"""Get the array implementation class.

    Returns:
        The array implementation class.
    """
    return jnp.ones(1).__class__
