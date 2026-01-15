r"""Implement handlers for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["NumpyArrayEqualHandler"]

import logging
from typing import TYPE_CHECKING

from coola.equality.handler.base import BaseEqualityHandler
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class NumpyArrayEqualHandler(BaseEqualityHandler):
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
            logger.info(
                f"numpy.ndarrays have different elements:\nactual:\n{actual}\nexpected:\n{expected}"
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
    if (config.atol > 0 or config.rtol > 0) and is_numeric_array(array1):
        return np.allclose(
            array1, array2, rtol=config.rtol, atol=config.atol, equal_nan=config.equal_nan
        )
    return np.array_equal(array1, array2, equal_nan=config.equal_nan)


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
