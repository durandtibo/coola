r"""Implement equality testers for PyArrow arrays and tables.

This module provides equality testers for pyarrow.Array and
pyarrow.Table using PyArrow's built-in equals() method.
"""

from __future__ import annotations

__all__ = ["PyarrowEqualityTester"]

from typing import TYPE_CHECKING, Any, cast

from coola.equality.handler import (
    PyarrowEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
    create_chain,
)
from coola.equality.tester.chain import BaseChainEqualityTester
from coola.utils.imports import check_pyarrow, is_pyarrow_available

if TYPE_CHECKING or is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa

if TYPE_CHECKING:
    from coola.equality.handler import BaseEqualityHandler
    from coola.equality.tester.base import BaseEqualityTester


class PyarrowEqualityTester(BaseChainEqualityTester[Any]):
    r"""Implement an equality tester for ``pyarrow.Array``s and
    ``pyarrow.Table``s.

    This tester uses PyArrow's equals() method for comparison. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are pyarrow objects
    3. PyarrowEqualHandler: Use PyArrow's equals() method

    Note:
        If the objects are different, the integer and floating-point columns
        honor ``config.equal_nan``, ``config.atol`` and ``config.rtol`` (NumPy
        is required). The other types are compared exactly.

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

    def _check_dependencies(self) -> None:
        check_pyarrow()

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(SameObjectHandler(), SameTypeHandler(), PyarrowEqualHandler())


def get_default_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the default equality testers of the pyarrow objects.

    Returns:
        The equality testers of the pyarrow objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the equality testers of the pyarrow objects.
    """
    if not is_pyarrow_available():
        return {}

    tester = PyarrowEqualityTester()
    pyarrow = cast("Any", pa)
    return {
        pyarrow.Array: tester,
        pyarrow.ChunkedArray: tester,
        pyarrow.RecordBatch: tester,
        pyarrow.Table: tester,
    }
