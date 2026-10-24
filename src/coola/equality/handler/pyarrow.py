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
from coola.utils.imports import is_numpy_available, is_pyarrow_available

if TYPE_CHECKING or is_pyarrow_available():
    import pyarrow as pa
    import pyarrow.compute as pc
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa

    pc = pa

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


logger: logging.Logger = logging.getLogger(__name__)


class PyarrowEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two pyarrow arrays or tables are equal.

    This handler returns ``True`` if the two arrays or tables are
    equal, otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    The integer and floating-point columns honor ``config.equal_nan``,
    ``config.atol`` and ``config.rtol`` (this requires NumPy). The
    other types are compared exactly.

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

    The objects are first compared with ``equals``. If they are
    different and ``config.equal_nan``, ``config.atol`` or
    ``config.rtol`` is set, the integer and floating-point columns
    (or arrays) are compared with the NumPy tolerance rules, and the
    other types (strings, nested types, etc.) are compared exactly.
    ``null`` values must be at the same positions. If NumPy is not
    installed, these arguments are ignored and a ``RuntimeWarning`` is
    emitted for each ignored argument, once per comparison (not for
    each object).

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
        >>> object_equal(pa.array([1.0, 2.0]), pa.array([1.0, 2.1]), EqualityConfig(atol=0.2))
        True

        ```
    """
    with suppress(TypeError):
        if obj1.equals(obj2):
            return True
    if not (config.equal_nan or config.atol > 0 or config.rtol > 0):
        return False
    if not is_numpy_available():  # pragma: no cover
        if config.equal_nan:
            _warn_ignored(config, "equal_nan", obj1)
        if config.atol > 0:
            _warn_ignored(config, "atol", obj1)
        if config.rtol > 0:
            _warn_ignored(config, "rtol", obj1)
        return False
    return _close(obj1, obj2, config)


def _close(obj1: Any, obj2: Any, config: EqualityConfig) -> bool:
    r"""Compare two arrays or tables with a tolerance."""
    if isinstance(obj1, (pa.Table, pa.RecordBatch)):
        return _close_table(obj1, obj2, config)
    if isinstance(obj1, (pa.Array, pa.ChunkedArray)):
        return _close_array(obj1, obj2, config)
    return False


def _close_table(obj1: Any, obj2: Any, config: EqualityConfig) -> bool:
    if not isinstance(obj2, type(obj1)) or not obj1.schema.equals(obj2.schema):
        return False
    if obj1.num_rows != obj2.num_rows:
        return False
    return all(
        _close_array(c1, c2, config) for c1, c2 in zip(obj1.columns, obj2.columns, strict=True)
    )


def _close_array(obj1: Any, obj2: Any, config: EqualityConfig) -> bool:
    if not isinstance(obj2, type(obj1)) or obj1.type != obj2.type or len(obj1) != len(obj2):
        return False
    if not (pa.types.is_floating(obj1.type) or pa.types.is_integer(obj1.type)):
        return bool(obj1.equals(obj2))
    if not obj1.is_null().equals(obj2.is_null()):
        return False
    from coola.equality.handler.numpy import array_equal  # noqa: PLC0415

    return array_equal(_to_numpy(obj1), _to_numpy(obj2), config)


def _to_numpy(array: Any) -> Any:
    r"""Convert a numeric array to NumPy, with the ``null`` values
    replaced by zero (the positions of the nulls are compared apart)."""
    return pc.fill_null(array, 0).to_numpy(zero_copy_only=False)


def _warn_ignored(config: EqualityConfig, name: str, obj: Any) -> None:
    r"""Warn that a setting is ignored, once per setting for a
    comparison.

    A nested structure can contain many Arrow objects, and the same
    warning for each of them would flood the output.
    """
    if not config.mark_warned(name):
        return
    warnings.warn(
        f"{name} is ignored because it is not supported for {type(obj)}",
        RuntimeWarning,
        stacklevel=4,
    )
