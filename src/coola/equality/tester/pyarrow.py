r"""Implement equality testers for PyArrow arrays and tables.

This module provides equality testers for pyarrow.Array and pyarrow.Table
using PyArrow's built-in equals() method.
"""

from __future__ import annotations

__all__ = ["PyarrowEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality.handler import (
    PyarrowEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_pyarrow, is_pyarrow_available

if TYPE_CHECKING or is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class PyarrowEqualityTester(BaseEqualityTester[pa.Array]):
    r"""Implement an equality tester for ``pyarrow.Array``s and
    ``pyarrow.Table``s.

    This tester uses PyArrow's equals() method for comparison. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are pyarrow objects
    3. PyarrowEqualHandler: Use PyArrow's equals() method

    Note:
        The ``config.equal_nan``, ``config.atol``, and ``config.rtol`` arguments
        are ignored as PyArrow's equals() method does not support these parameters.
        PyArrow performs its own equality checking logic.

    Example:
        Basic array comparison:

        ```pycon
        >>> import pyarrow as pa
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import PyarrowEqualityTester
        >>> config = EqualityConfig()
        >>> tester = PyarrowEqualityTester()
        >>> tester.objects_are_equal(pa.array([1, 2, 3]), pa.array([1, 2, 3]), config)
        True
        >>> tester.objects_are_equal(pa.array([1, 2, 3]), pa.array([1, 2, 4]), config)
        False

        ```

        Table comparison:

        ```pycon
        >>> import pyarrow as pa
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import PyarrowEqualityTester
        >>> config = EqualityConfig()
        >>> tester = PyarrowEqualityTester()
        >>> table1 = pa.table({"col": [1, 2, 3]})
        >>> table2 = pa.table({"col": [1, 2, 3]})
        >>> tester.objects_are_equal(table1, table2, config)
        True

        ```
    """

    def __init__(self) -> None:
        check_pyarrow()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(PyarrowEqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: pa.Array,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
