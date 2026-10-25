r"""Implement handlers for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["NumpyArrayEqualHandler", "NumpyMaskedArrayDataHandler"]

from typing import TYPE_CHECKING

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import AttributeName, report_difference
from coola.utils.imports import is_numpy_available
from coola.utils.safe import safe_str

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
        # ``np.allclose`` scales the tolerance with the second array only, so the two
        # directions are tried to make the comparison symmetric.
        return np.allclose(
            array1, array2, rtol=config.rtol, atol=config.atol, equal_nan=config.equal_nan
        ) or np.allclose(
            array2, array1, rtol=config.rtol, atol=config.atol, equal_nan=config.equal_nan
        )
    # ``np.isnan`` is not defined for string/object/void dtypes, so
    # ``equal_nan`` is only forwarded when both arrays support it.
    equal_nan = config.equal_nan and _supports_nan(array1) and _supports_nan(array2)
    return np.array_equal(array1, array2, equal_nan=equal_nan)


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


class NumpyMaskedArrayDataHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two NumPy masked arrays have the same mask and the
    same unmasked values.

    This handler returns ``False`` if the masks are different or if
    the values that are not masked are different, otherwise it passes
    the inputs to the next handler. The data under the mask is
    ignored: it is unspecified (it is what the array held before
    masking), so two arrays that print identically are equal. The
    arrays must have the same shape, which another handler checks.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import NumpyMaskedArrayDataHandler, TrueHandler
        >>> config = EqualityConfig()
        >>> handler = NumpyMaskedArrayDataHandler(next_handler=TrueHandler())
        >>> handler.handle(
        ...     np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 3], mask=[0, 1]), config
        ... )
        True
        >>> handler.handle(
        ...     np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[1, 0]), config
        ... )
        False

        ```
    """

    def handle(
        self,
        actual: np.ma.MaskedArray,
        expected: np.ma.MaskedArray,
        config: EqualityConfig,
    ) -> bool:
        mask1, mask2 = np.ma.getmaskarray(actual), np.ma.getmaskarray(expected)
        if not np.array_equal(mask1, mask2):
            if config.show_difference:
                report_difference(
                    config,
                    "objects have different mask: %s vs %s",
                    safe_str(mask1),
                    safe_str(mask2),
                    path=(AttributeName("mask"),),
                )
            return False
        values1, values2 = actual.compressed(), expected.compressed()
        if not config.registry.objects_are_equal(values1, values2, config):
            if config.show_difference:
                report_difference(
                    config,
                    "objects have different data: %s vs %s",
                    safe_str(values1),
                    safe_str(values2),
                    path=(AttributeName("data"),),
                )
            return False
        return self._handle_next(actual, expected, config=config)
