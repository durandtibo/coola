r"""Implement handlers for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["NumpyArrayEqualHandler", "NumpyTimeScalarEqualHandler"]

from typing import TYPE_CHECKING

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.elementwise import elementwise_close
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.scalar import int_isclose
from coola.equality.handler.utils import report_difference
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class NumpyArrayEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two NumPy arrays are equal.

    This handler returns ``True`` if the two arrays are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import NumpyArrayEqualHandler
        >>> config = EqualityConfig()
        >>> handler = NumpyArrayEqualHandler()
        >>> handler.handle(np.ones((2, 3)), np.ones((2, 3)), config)
        True
        >>> handler.handle(np.ones((2, 3)), np.zeros((2, 3)), config)
        False

        ```
    """

    def handle(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        config: EqualityConfig,
    ) -> bool:
        object_equal = array_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            report_difference(
                config, format_value_difference(actual, expected, name="numpy.ndarrays")
            )
        return object_equal


def array_equal(array1: np.ndarray, array2: np.ndarray, config: EqualityConfig) -> bool:
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
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.numpy import array_equal
        >>> config = EqualityConfig()
        >>> array_equal(np.ones((2, 3)), np.ones((2, 3)), config)
        True
        >>> array_equal(np.ones((2, 3)), np.zeros((2, 3)), config)
        False

        ```
    """
    if array1.dtype.kind == "O" and array2.dtype.kind == "O":
        # Compare the elements through the registry so ``equal_nan``, the
        # tolerances and the nested testers apply to the stored objects.
        return array1.shape == array2.shape and all(
            config.registry.objects_are_equal(x, y, config)
            for x, y in zip(array1.ravel().tolist(), array2.ravel().tolist(), strict=True)
        )
    if (
        (config.atol > 0 or config.rtol > 0)
        and array1.dtype.kind == "m"
        and array2.dtype.kind == "m"
    ):
        # A duration is a quantity (unlike an instant): the tolerance applies, ``atol`` being a
        # number of seconds.
        return _durations_close(array1, array2, config)
    if (
        (config.atol > 0 or config.rtol > 0)
        and is_numeric_array(array1)
        and is_numeric_array(array2)
        # A boolean is a label, not a quantity: no tolerance.
        and array1.dtype.kind != "b"
        and array2.dtype.kind != "b"
    ):
        return _arrays_close(array1, array2, config)
    # ``np.isnan`` is not defined for string/object/void dtypes, so
    # ``equal_nan`` is only forwarded when both arrays support it.
    equal_nan = config.equal_nan and _supports_nan(array1) and _supports_nan(array2)
    return np.array_equal(array1, array2, equal_nan=equal_nan)


def _durations_close(array1: np.ndarray, array2: np.ndarray, config: EqualityConfig) -> bool:
    r"""Indicate if two arrays of ``timedelta64`` are equal within a
    tolerance.

    The durations are compared as integer nanoseconds with the rule
    ``|a - b| <= atol + rtol * max(|a|, |b|)`` (``atol`` is a number of
    seconds). ``NaT`` is equal to ``NaT`` only if ``config.equal_nan``
    is ``True``.
    """
    if array1.shape != array2.shape:
        return False
    nat1, nat2 = np.isnat(array1), np.isnat(array2)
    nanoseconds1 = array1.astype("timedelta64[ns]").astype(np.int64)
    nanoseconds2 = array2.astype("timedelta64[ns]").astype(np.int64)
    # The difference is computed with floats: the integers are exact below 2^53 ns (104 days)
    # and the relative error is below 1e-16 above.
    difference = np.abs(nanoseconds1.astype(float) - nanoseconds2.astype(float))
    scale = np.maximum(np.abs(nanoseconds1.astype(float)), np.abs(nanoseconds2.astype(float)))
    close = (nanoseconds1 == nanoseconds2) | (difference <= config.atol * 1e9 + config.rtol * scale)
    close = np.where(nat1 | nat2, config.equal_nan & nat1 & nat2, close)
    return bool(close.all())


def _arrays_close(array1: np.ndarray, array2: np.ndarray, config: EqualityConfig) -> bool:
    if array1.shape != array2.shape:
        return False
    with np.errstate(invalid="ignore", over="ignore"):  # ``inf - inf``
        close = elementwise_close(
            array1, array2, atol=config.atol, rtol=config.rtol, equal_nan=config.equal_nan, xp=np
        )
    return bool(close.all())


def _supports_nan(array: np.ndarray) -> bool:
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
        >>> from coola.equality.handler.numpy import is_numeric_array
        >>> is_numeric_array(np.ones((2, 3)))
        True
        >>> is_numeric_array(np.array(["polar", "bear", "meow"]))
        False

        ```
    """
    return array.dtype.kind in {"?", "b", "B", "i", "u", "f", "c"}


class NumpyTimeScalarEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two ``numpy.timedelta64`` or ``numpy.datetime64``
    scalars are equal.

    This handler returns ``True`` if the two scalars are equal,
    otherwise ``False``. See ``time_scalar_equal`` for the rule.
    This handler is designed to be used at the end of the chain of
    responsibility. This handler does not call the next handler.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import NumpyTimeScalarEqualHandler
        >>> config = EqualityConfig()
        >>> handler = NumpyTimeScalarEqualHandler()
        >>> handler.handle(np.timedelta64(1, "s"), np.timedelta64(1, "s"), config)
        True
        >>> handler.handle(np.timedelta64(1, "s"), np.timedelta64(2, "s"), config)
        False

        ```
    """

    def handle(
        self,
        actual: np.timedelta64 | np.datetime64,
        expected: np.timedelta64 | np.datetime64,
        config: EqualityConfig,
    ) -> bool:
        object_equal = time_scalar_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            report_difference(
                config, format_value_difference(actual, expected, name="numpy time scalars")
            )
        return object_equal


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
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.numpy import time_scalar_equal
        >>> config = EqualityConfig(atol=1.0)
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
