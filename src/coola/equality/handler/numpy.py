r"""Implement handlers for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["NumpyArrayEqualHandler"]

from typing import TYPE_CHECKING

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.elementwise import elementwise_close
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
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
        and is_numeric_array(array1)
        and is_numeric_array(array2)
    ):
        return _arrays_close(array1, array2, config)
    # ``np.isnan`` is not defined for string/object/void dtypes, so
    # ``equal_nan`` is only forwarded when both arrays support it.
    equal_nan = config.equal_nan and _supports_nan(array1) and _supports_nan(array2)
    return np.array_equal(array1, array2, equal_nan=equal_nan)


def _arrays_close(array1: np.ndarray, array2: np.ndarray, config: EqualityConfig) -> bool:
    if array1.shape != array2.shape:
        return False
    # The integers and the booleans are converted: an unsigned subtraction wraps around.
    if array1.dtype.kind not in {"f", "c"}:
        array1 = array1.astype(np.float64)
    if array2.dtype.kind not in {"f", "c"}:
        array2 = array2.astype(np.float64)
    with np.errstate(invalid="ignore"):  # ``inf - inf``
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
