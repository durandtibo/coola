r"""Implement some handlers for ``jax.numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["JaxArrayEqualHandler"]

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.equality.handlers.base import BaseEqualityHandler
from coola.utils.imports import is_jax_available

if is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = Mock()

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class JaxArrayEqualHandler(BaseEqualityHandler):
    r"""Check if the two JAX arrays are equal.

    This handler returns ``True`` if the two arrays are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example usage:

    ```pycon

    >>> import jax.numpy as jnp
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import JaxArrayEqualHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = JaxArrayEqualHandler()
    >>> handler.handle(jnp.ones((2, 3)), jnp.ones((2, 3)), config)
    True
    >>> handler.handle(jnp.ones((2, 3)), jnp.zeros((2, 3)), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        actual: jnp.ndarray,
        expected: jnp.ndarray,
        config: EqualityConfig,
    ) -> bool:
        object_equal = array_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            logger.info(
                f"jax.numpy.ndarrays have different elements:\n"
                f"actual=\n{actual}\nexpected=\n{expected}"
            )
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


def array_equal(array1: jnp.ndarray, array2: jnp.ndarray, config: EqualityConfig) -> bool:
    r"""Indicate if the two arrays are equal within a tolerance.

    Args:
        array1: The first array to compare.
        array2: The second array to compare.
        config: The equality configuration.

    Returns:
        ``True``if the two arrays are equal within a tolerance,
            otherwise ``False``.
    """
    if config.atol > 0 or config.rtol > 0:
        return jnp.allclose(
            array1, array2, rtol=config.rtol, atol=config.atol, equal_nan=config.equal_nan
        ).item()
    return jnp.array_equal(array1, array2, equal_nan=config.equal_nan).item()
