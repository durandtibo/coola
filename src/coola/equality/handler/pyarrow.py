r"""Implement handlers for ``pyarrow`` objects."""

from __future__ import annotations

__all__ = ["PyarrowEqualHandler"]

import logging
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.utils import handlers_are_equal

if TYPE_CHECKING:
    import pyarrow as pa

    from coola.equality.config import EqualityConfig


logger: logging.Logger = logging.getLogger(__name__)


class PyarrowEqualHandler(BaseEqualityHandler):
    r"""Check if the two pyarrow arrays or tables are equal.

    This handler returns ``True`` if the two arrays or tables are
    equal, otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Note that ``config.equal_nan``, ``config.atol`` and ``config.rtol``
    arguments are ignored.

    Example:
        ```pycon
        >>> import pyarrow
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PyarrowEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PyarrowEqualHandler()
        >>> handler.handle(pyarrow.array([1, 2, 3]), pyarrow.array([1, 2, 3]), config)
        True
        >>> handler.handle(pyarrow.array([1, 2, 3]), pyarrow.array([1, 2, 4]), config)
        False

        ```
    """

    def equal(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return handlers_are_equal(self.next_handler, other.next_handler)

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
        ``True`` if the two arrays or tables are equal within a
            tolerance, otherwise ``False``.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.pyarrow import object_equal
        >>> config = EqualityConfig()
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
