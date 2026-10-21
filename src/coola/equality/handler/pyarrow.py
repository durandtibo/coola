r"""Implement handlers for ``pyarrow`` objects."""

from __future__ import annotations

__all__ = ["PyarrowEqualHandler"]

import logging
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import report_difference

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


logger: logging.Logger = logging.getLogger(__name__)


class PyarrowEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
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

    def handle(
        self,
        actual: Any,
        expected: Any,
        config: EqualityConfig,
    ) -> bool:
        equal = object_equal(actual, expected, config)
        if config.show_difference and not equal:
            report_difference(
                logger, config, format_value_difference(actual=actual, expected=expected)
            )
        return equal


def object_equal(obj1: Any, obj2: Any, config: EqualityConfig) -> bool:
    r"""Indicate if the two arrays or tables are equal within a
    tolerance.

    ``config.equal_nan``, ``config.atol`` and ``config.rtol``
    arguments are ignored. A ``RuntimeWarning`` is emitted for each
    ignored argument, once per comparison (not for each object).

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
        _warn_ignored(config, "equal_nan", obj1)
    if config.atol > 0:
        _warn_ignored(config, "atol", obj1)
    if config.rtol > 0:
        _warn_ignored(config, "rtol", obj1)
    with suppress(TypeError):
        return obj1.equals(obj2)
    return False


def _warn_ignored(config: EqualityConfig, name: str, obj: Any) -> None:
    r"""Warn that a setting is ignored, once per setting for a
    comparison.

    A nested structure can contain many Arrow objects, and the same
    warning for each of them would flood the output.
    """
    if name in config._warned:
        return
    config._warned.add(name)
    warnings.warn(
        f"{name} is ignored because it is not supported for {type(obj)}",
        RuntimeWarning,
        stacklevel=4,
    )
