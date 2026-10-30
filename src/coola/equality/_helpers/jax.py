r"""Implement handlers for ``jax.numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["array_close", "array_equal"]

from typing import TYPE_CHECKING

from coola.equality._helpers.elementwise import elementwise_close
from coola.utils.imports import is_jax_available

if TYPE_CHECKING or is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    from coola.utils.fallback.jax import jnp

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


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
        >>> from coola.equality._helpers.jax import array_equal
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


def array_close(array1: jnp.ndarray, array2: jnp.ndarray, config: EqualityConfig) -> jnp.ndarray:
    r"""Indicate for each element if two arrays are equal within a
    tolerance.

    It is the rule of ``array_equal`` for each pair of elements, so it
    gives the location of the first difference. The two arrays must
    have the same shape and dtype.

    Args:
        array1: The first array to compare.
        array2: The second array to compare.
        config: The equality configuration.

    Returns:
        A boolean array with the shape of the inputs.

    Example:
        ```pycon
        >>> import jax.numpy as jnp
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality._helpers.jax import array_close
        >>> array_close(jnp.array([1.0, 2.0]), jnp.array([1.0, 3.0]), EqualityConfig())
        Array([ True, False], dtype=bool)

        ```
    """
    if jnp.bool_ not in (array1.dtype, array2.dtype) and (config.atol > 0 or config.rtol > 0):
        return elementwise_close(
            array1,
            array2,
            atol=config.atol,
            rtol=config.rtol,
            equal_nan=config.equal_nan,
            xp=jnp,
        )
    close = array1 == array2
    if config.equal_nan and jnp.bool_ not in (array1.dtype, array2.dtype):
        close = close | (jnp.isnan(array1) & jnp.isnan(array2))
    return close


def _arrays_close(array1: jnp.ndarray, array2: jnp.ndarray, config: EqualityConfig) -> bool:
    r"""Indicate if two JAX arrays are equal within the tolerance.

    Args:
        array1: The first array.
        array2: The second array.
        config: The configuration of the comparison.

    Returns:
        ``True`` if the arrays have the same shape and dtype and all the
            elements are close, otherwise ``False``.
    """
    if array1.shape != array2.shape or array1.dtype != array2.dtype:
        return False
    close = elementwise_close(
        array1, array2, atol=config.atol, rtol=config.rtol, equal_nan=config.equal_nan, xp=jnp
    )
    return bool(close.all().item())
