r"""Implement handlers for ``jax.numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["JaxArrayEqualHandler"]

from typing import TYPE_CHECKING

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.elementwise import elementwise_close
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import report_difference
from coola.utils.imports import is_jax_available

if TYPE_CHECKING or is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    from coola.utils.fallback.jax import jnp

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class JaxArrayEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two JAX arrays are equal.

    This handler returns ``True`` if the two arrays are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> import jax.numpy as jnp
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import JaxArrayEqualHandler
        >>> config = EqualityConfig()
        >>> handler = JaxArrayEqualHandler()
        >>> handler.handle(jnp.ones((2, 3)), jnp.ones((2, 3)), config)
        True
        >>> handler.handle(jnp.ones((2, 3)), jnp.zeros((2, 3)), config)
        False

        ```
    """

    def handle(
        self,
        actual: jnp.ndarray,
        expected: jnp.ndarray,
        config: EqualityConfig,
    ) -> bool:
        object_equal = array_equal(actual, expected, config)
        if config.explaining and not object_equal:
            report_difference(
                config,
                format_value_difference(
                    actual=actual, expected=expected, name="jax.numpy.ndarrays"
                ),
            )
        return object_equal


def array_equal(array1: jnp.ndarray, array2: jnp.ndarray, config: EqualityConfig) -> bool:
    r"""Indicate if the two arrays are equal within a tolerance.

    Args:
        array1: The first array to compare.
        array2: The second array to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two arrays are equal within a tolerance,
            otherwise ``False``.

    Example:
        ```pycon
        >>> import jax.numpy as jnp
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.jax import array_equal
        >>> config = EqualityConfig()
        >>> array_equal(jnp.ones((2, 3)), jnp.ones((2, 3)), config)
        True
        >>> array_equal(jnp.ones((2, 3)), jnp.zeros((2, 3)), config)
        False

        ```
    """
    if jnp.bool_ in (array1.dtype, array2.dtype):
        # A boolean is a label, not a quantity: no tolerance.
        return jnp.array_equal(array1, array2).item()
    if config.atol > 0 or config.rtol > 0:
        return _arrays_close(array1, array2, config)
    return jnp.array_equal(array1, array2, equal_nan=config.equal_nan).item()


def _arrays_close(array1: jnp.ndarray, array2: jnp.ndarray, config: EqualityConfig) -> bool:
    if array1.shape != array2.shape or array1.dtype != array2.dtype:
        return False
    close = elementwise_close(
        array1, array2, atol=config.atol, rtol=config.rtol, equal_nan=config.equal_nan, xp=jnp
    )
    return bool(close.all().item())
