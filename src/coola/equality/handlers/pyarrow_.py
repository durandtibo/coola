r"""Implement some handlers for ``pyarrow`` objects."""

from __future__ import annotations

__all__ = ["PyarrowArrayEqualHandler"]

import logging
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.equality.handlers.base import BaseEqualityHandler
from coola.utils.imports import is_pyarrow_available

if is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    pa = Mock()

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class PyarrowArrayEqualHandler(BaseEqualityHandler):
    r"""Check if the two pyarrow arrays are equal.

    This handler returns ``True`` if the two arrays are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example usage:

    ```pycon

    >>> import pyarrow
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import PyarrowArrayEqualHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = PyarrowArrayEqualHandler()
    >>> handler.handle(pyarrow.array([1, 2, 3]), pyarrow.array([1, 2, 3]), config)
    True
    >>> handler.handle(pyarrow.array([1, 2, 3]), pyarrow.array([1, 2, 4]), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        actual: pa.Array,
        expected: pa.Array,
        config: EqualityConfig,
    ) -> bool:
        object_equal = array_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            logger.info(
                f"pyarrow.Arrays have different elements:\n"
                f"actual:\n{actual}\nexpected:\n{expected}"
            )
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


def array_equal(array1: pa.Array, array2: pa.Array, config: EqualityConfig) -> bool:
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

    >>> import pyarrow
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers.pyarrow_ import array_equal
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> array_equal(pyarrow.array([1, 2, 3]), pyarrow.array([1, 2, 3]), config)
    True
    >>> array_equal(pyarrow.array([1, 2, 3]), pyarrow.array([1, 2, 4]), config)
    False

    ```
    """
    if config.equal_nan:
        warnings.warn(
            f"equal_nan is ignored because it is not supported for {type(array1)}",
            RuntimeWarning,
            stacklevel=3,
        )
    if config.atol > 0:
        warnings.warn(
            f"atol is ignored because it is not supported for {type(array1)}",
            RuntimeWarning,
            stacklevel=3,
        )
    if config.rtol > 0:
        warnings.warn(
            f"rtol is ignored because it is not supported for {type(array1)}",
            RuntimeWarning,
            stacklevel=3,
        )
    with suppress(TypeError):
        return array1.equals(array2)
    return False
