r"""Implement handlers for ``pyarrow`` objects."""

from __future__ import annotations

__all__ = ["PyarrowEqualHandler"]

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
        if config.explaining and not equal:
            report_difference(config, format_value_difference(actual=actual, expected=expected))
        return equal


def object_equal(obj1: Any, obj2: Any, config: EqualityConfig) -> bool:
    r"""Indicate if the two arrays or tables are equal within a
    tolerance.

    The structure is compared first (the type and the length of an
    array; the schema, without its metadata, and the number of rows of
    a table). The columns (or the arrays) are then compared one by one,
    by the kind of their type, and the comparison stops at the first
    column that differs:

    - integers and floats: the ``null`` values must be at the same
      positions, then the values are compared with the NumPy tolerance
      rule (``atol``, ``rtol``, ``equal_nan``);
    - durations: same, on the durations (``atol`` is a number of
      seconds);
    - the other types (booleans, strings, instants, decimals, nested
      types, ...): the ``equals`` of Arrow. It never sees the tolerance
      or ``equal_nan``, a ``null`` is equal to a ``null`` and a ``NaN``
      nested in a value is not equal to a ``NaN``.

    If NumPy is not installed, ``config.equal_nan``, ``config.atol``
    and ``config.rtol`` are ignored and a ``RuntimeWarning`` is emitted
    for each ignored argument, once per comparison (not for each
    object).

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
        >>> object_equal(pa.array([1.0, 2.0]), pa.array([1.0, 2.05]), EqualityConfig(atol=0.1))
        True

        ```
    """
    if not is_numpy_available():  # pragma: no cover
        if config.equal_nan:
            _warn_ignored(config, "equal_nan", obj1)
        if config.atol > 0:
            _warn_ignored(config, "atol", obj1)
        if config.rtol > 0:
            _warn_ignored(config, "rtol", obj1)
        with suppress(TypeError):
            return bool(obj1.equals(obj2))
        return False
    if isinstance(obj1, (pa.Table, pa.RecordBatch)):
        return _table_equal(obj1, obj2, config)
    if isinstance(obj1, (pa.Array, pa.ChunkedArray)):
        return _array_equal(obj1, obj2, config)
    with suppress(TypeError):
        return bool(obj1.equals(obj2))
    return False


def _table_equal(obj1: Any, obj2: Any, config: EqualityConfig) -> bool:
    if not isinstance(obj2, type(obj1)) or not obj1.schema.equals(obj2.schema):
        return False
    if obj1.num_rows != obj2.num_rows:
        return False
    return all(
        _array_equal(c1, c2, config) for c1, c2 in zip(obj1.columns, obj2.columns, strict=True)
    )


def _array_equal(obj1: Any, obj2: Any, config: EqualityConfig) -> bool:
    if not isinstance(obj2, type(obj1)) or obj1.type != obj2.type or len(obj1) != len(obj2):
        return False
    kind = obj1.type
    if pa.types.is_floating(kind) or pa.types.is_integer(kind) or pa.types.is_duration(kind):
        if not obj1.is_null().equals(obj2.is_null()):
            return False
        from coola.equality.handler.numpy import array_equal  # noqa: PLC0415

        return array_equal(_to_numpy(obj1), _to_numpy(obj2), config)
    return bool(obj1.equals(obj2))


def _to_numpy(array: Any) -> Any:
    r"""Convert a numeric or duration array to NumPy, with the ``null``
    values replaced by zero (the positions of the nulls are compared
    apart)."""
    return pc.fill_null(array, pa.scalar(0, type=array.type)).to_numpy(zero_copy_only=False)


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
