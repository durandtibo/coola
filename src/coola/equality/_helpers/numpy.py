r"""Implement handlers for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["array_close", "array_equal", "is_numeric_array", "time_scalar_equal"]

from typing import TYPE_CHECKING

from coola.equality._helpers.elementwise import elementwise_close
from coola.equality._helpers.scalar import int_isclose
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


def array_equal(array1: np.ndarray, array2: np.ndarray, config: EqualityConfig) -> bool:
    r"""Indicate if the two arrays are equal within a tolerance.

    The two arrays must not have the ``object`` dtype: the elements of
    an ``object`` array are compared by the engine (see
    ``NumpyArrayEqualityTester``).

    Args:
        array1: The first array to compare.
        array2: The second array to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two arrays are equal within a tolerance,
            otherwise ``False``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality._helpers.numpy import array_equal
        >>> config = EqualityConfig()
        >>> array_equal(np.ones((2, 3)), np.ones((2, 3)), config)
        True
        >>> array_equal(np.ones((2, 3)), np.zeros((2, 3)), config)
        False

        ```
    """
    if _uses_tolerance(array1, array2, config):
        return array1.shape == array2.shape and bool(array_close(array1, array2, config).all())
    # ``np.isnan`` is not defined for string/object/void dtypes, so
    # ``equal_nan`` is only forwarded when both arrays support it.
    equal_nan = config.equal_nan and _supports_nan(array1) and _supports_nan(array2)
    return np.array_equal(array1, array2, equal_nan=equal_nan)


def array_close(array1: np.ndarray, array2: np.ndarray, config: EqualityConfig) -> np.ndarray:
    r"""Indicate for each element if two arrays are equal within a
    tolerance.

    It is the rule of ``array_equal`` for each pair of elements, so it
    gives the location of the first difference. The two arrays must
    have the same shape.

    Args:
        array1: The first array to compare.
        array2: The second array to compare.
        config: The equality configuration.

    Returns:
        A boolean array with the shape of the inputs.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality._helpers.numpy import array_close
        >>> array_close(np.array([1.0, 2.0]), np.array([1.0, 3.0]), EqualityConfig())
        array([ True, False])

        ```
    """
    if _uses_tolerance(array1, array2, config):
        with np.errstate(invalid="ignore", over="ignore"):  # ``inf - inf``
            if array1.dtype.kind == "m":
                return _durations_close(array1, array2, config)
            return elementwise_close(
                array1,
                array2,
                atol=config.atol,
                rtol=config.rtol,
                equal_nan=config.equal_nan,
                xp=np,
            )
    close = array1 == array2
    if config.equal_nan and _supports_nan(array1) and _supports_nan(array2):
        close = close | (_is_nan_like(array1) & _is_nan_like(array2))
    return np.asarray(close)


def _uses_tolerance(array1: np.ndarray, array2: np.ndarray, config: EqualityConfig) -> bool:
    r"""Indicate if the elements are compared with the numeric rule.

    Args:
        array1: The first array.
        array2: The second array.
        config: The configuration of the comparison.

    Returns:
        ``True`` if the tolerance applies to the arrays, otherwise
            ``False``.
    """
    if config.atol <= 0 and config.rtol <= 0:
        return False
    if array1.dtype.kind == "m" and array2.dtype.kind == "m":
        # A duration is a quantity (unlike an instant): the tolerance applies, ``atol`` being a
        # number of seconds.
        return True
    # A boolean is a label, not a quantity: no tolerance.
    return (
        is_numeric_array(array1)
        and is_numeric_array(array2)
        and array1.dtype.kind != "b"
        and array2.dtype.kind != "b"
    )


def _is_nan_like(array: np.ndarray) -> np.ndarray:
    r"""Indicate the missing values of an array.

    Args:
        array: The array.

    Returns:
        A boolean array that is ``True`` at the ``NaT`` (``datetime64``
            and ``timedelta64``) or ``NaN`` elements.
    """
    return np.isnat(array) if array.dtype.kind in {"m", "M"} else np.isnan(array)


def _durations_close(array1: np.ndarray, array2: np.ndarray, config: EqualityConfig) -> np.ndarray:
    r"""Indicate for each element if two arrays of ``timedelta64`` are
    equal within a tolerance.

    The durations are compared as integer nanoseconds with the rule
    ``|a - b| <= atol + rtol * max(|a|, |b|)`` (``atol`` is a number of
    seconds). ``NaT`` is equal to ``NaT`` only if ``config.equal_nan``
    is ``True``.

    Args:
        array1: The first array of ``timedelta64``.
        array2: The second array of ``timedelta64``.
        config: The configuration of the comparison.

    Returns:
        A boolean array that is ``True`` where the durations are equal
            within the tolerance.
    """
    nat1, nat2 = np.isnat(array1), np.isnat(array2)
    nanoseconds1 = array1.astype("timedelta64[ns]").astype(np.int64)
    nanoseconds2 = array2.astype("timedelta64[ns]").astype(np.int64)
    # The difference is computed with floats: the integers are exact below 2^53 ns (104 days)
    # and the relative error is below 1e-16 above.
    difference = np.abs(nanoseconds1.astype(float) - nanoseconds2.astype(float))
    scale = np.maximum(np.abs(nanoseconds1.astype(float)), np.abs(nanoseconds2.astype(float)))
    close = (nanoseconds1 == nanoseconds2) | (difference <= config.atol * 1e9 + config.rtol * scale)
    return np.where(nat1 | nat2, config.equal_nan & nat1 & nat2, close)


def _supports_nan(array: np.ndarray) -> bool:
    r"""Indicate if an array can hold missing values.

    Args:
        array: The array.

    Returns:
        ``True`` for a numeric, ``datetime64`` or ``timedelta64`` array,
            otherwise ``False``.
    """
    return is_numeric_array(array) or array.dtype.kind in {"m", "M"}


def is_numeric_array(array: np.ndarray) -> bool:
    r"""Indicate if the input array is a numeric array or not.

    Args:
        array: The input array.

    Returns:
        ``True`` if the input array is a numeric array, otherwise ``False``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality._helpers.numpy import is_numeric_array
        >>> is_numeric_array(np.ones((2, 3)))
        True
        >>> is_numeric_array(np.array(["polar", "bear", "meow"]))
        False

        ```
    """
    return array.dtype.kind in {"?", "b", "B", "i", "u", "f", "c"}


def time_scalar_equal(
    actual: np.timedelta64 | np.datetime64,
    expected: np.timedelta64 | np.datetime64,
    config: EqualityConfig,
) -> bool:
    r"""Indicate if two ``numpy.timedelta64`` or ``numpy.datetime64``
    scalars are equal.

    ``NaT`` is equal to ``NaT`` only if ``config.equal_nan`` is
    ``True``. A ``numpy.timedelta64`` is a duration: the tolerances
    apply (``atol`` is a number of seconds, the difference is computed
    in nanoseconds). A ``numpy.datetime64`` is an instant and is
    compared exactly.

    Args:
        actual: The actual scalar.
        expected: The expected scalar, with the same type.
        config: The equality configuration, which gives ``atol``,
            ``rtol`` and ``equal_nan``.

    Returns:
        ``True`` if the scalars are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.equality import Tolerance
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality._helpers.numpy import time_scalar_equal
        >>> config = EqualityConfig(tolerance=Tolerance(atol=1.0))
        >>> time_scalar_equal(np.timedelta64(1, "s"), np.timedelta64(2000, "ms"), config)
        True
        >>> time_scalar_equal(np.datetime64("2020-01-01"), np.datetime64("2020-01-02"), config)
        False

        ```
    """
    actual_nat, expected_nat = bool(np.isnat(actual)), bool(np.isnat(expected))
    if actual_nat or expected_nat:
        return config.equal_nan and actual_nat and expected_nat
    if bool(actual == expected):
        return True
    if not isinstance(actual, np.timedelta64) or (config.atol <= 0.0 and config.rtol <= 0.0):
        return False
    nanoseconds = (int(x.astype("timedelta64[ns]").astype(np.int64)) for x in (actual, expected))
    return int_isclose(*nanoseconds, rtol=config.rtol, atol=config.atol * 1e9)
