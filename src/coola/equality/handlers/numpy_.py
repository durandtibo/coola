r"""Implement some handlers for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["NumpyArrayEqualHandler"]

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.equality.handlers.base import BaseEqualityHandler
from coola.utils import is_numpy_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class NumpyArrayEqualHandler(BaseEqualityHandler):
    r"""Check if the two NumPy arrays are equal.

    This handler returns ``True`` if the two arrays are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import NumpyArrayEqualHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = NumpyArrayEqualHandler()
    >>> handler.handle(np.ones((2, 3)), np.ones((2, 3)), config)
    True
    >>> handler.handle(np.ones((2, 3)), np.zeros((2, 3)), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        config: EqualityConfig,
    ) -> bool:
        object_equal = array_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            logger.info(
                f"numpy.ndarrays have different elements:\n"
                f"actual:\n{actual}\nexpected:\n{expected}"
            )
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


def array_equal(array1: np.ndarray, array2: np.ndarray, config: EqualityConfig) -> bool:
    r"""Indicate if the two arrays are equal within a tolerance.

    Args:
        array1: The first array to compare.
        array2: The second array to compare.
        config: The equality configuration.

    Returns:
        ``True` `if the two arrays are equal within a tolerance,
            otherwise ``False``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers.numpy_ import array_equal
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
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

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from coola.equality.handlers.numpy_ import is_numeric_array
    >>> is_numeric_array(np.ones((2, 3)))
    True
    >>> is_numeric_array(np.array(["polar", "bear", "meow"]))
    False

    ```
    """
    return array.dtype.kind in {"?", "b", "B", "i", "u", "f", "c"}
