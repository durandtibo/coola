r"""Implement some handlers for ``pyarrow`` objects."""

from __future__ import annotations

__all__ = ["PyarrowEqualHandler"]

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


class PyarrowEqualHandler(BaseEqualityHandler):
    r"""Check if the two pyarrow arrays or tables are equal.

    This handler returns ``True`` if the two arrays or tables are
    equal, otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Note that ``config.equal_nan``, ``config.atol`` and ``config.rtol``
    arguments are ignored.

    Example usage:

    ```pycon

    >>> import pyarrow
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import PyarrowEqualHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = PyarrowEqualHandler()
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
        actual: pa.Array | pa.Table,
        expected: pa.Array | pa.Table,
        config: EqualityConfig,
    ) -> bool:
        equal = object_equal(actual, expected, config)
        if config.show_difference and not equal:
            logger.info(f"objects are different:\nactual:\n{actual}\nexpected:\n{expected}")
        return equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


def object_equal(
    obj1: pa.Array | pa.Table, obj2: pa.Array | pa.Table, config: EqualityConfig
) -> bool:
    r"""Indicate if the two arrays or tables are equal within a
    tolerance.

    ``config.equal_nan``, ``config.atol`` and ``config.rtol``
    arguments are ignored.

    Args:
        obj1: The first array or table to compare.
        obj2: The second array or table to compare.
        config: The equality configuration.

    Returns:
        ``True` `if the two arrays or tables are equal within a
            tolerance, otherwise ``False``.

    Example usage:

    ```pycon

    >>> import pyarrow as pa
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers.pyarrow_ import object_equal
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> object_equal(pa.array([1, 2, 3]), pa.array([1, 2, 3]), config)
    True
    >>> object_equal(pa.array([1, 2, 3]), pa.array([1, 2, 4]), config)
    False

    ```
    """
    if config.equal_nan:
        warnings.warn(
            f"equal_nan is ignored because it is not supported for {type(obj1)}",
            RuntimeWarning,
            stacklevel=3,
        )
    if config.atol > 0:
        warnings.warn(
            f"atol is ignored because it is not supported for {type(obj1)}",
            RuntimeWarning,
            stacklevel=3,
        )
    if config.rtol > 0:
        warnings.warn(
            f"rtol is ignored because it is not supported for {type(obj1)}",
            RuntimeWarning,
            stacklevel=3,
        )
    with suppress(TypeError):
        return obj1.equals(obj2)
    return False
