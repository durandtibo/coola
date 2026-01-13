r"""Implement an equality tester for ``pyarrow.Array``s and
``pyarrow.Table``s."""

from __future__ import annotations

__all__ = ["PyarrowEqualityTester"]

import logging
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
    from coola.equality.config import EqualityConfig2

logger: logging.Logger = logging.getLogger(__name__)


class PyarrowEqualityTester(BaseEqualityTester[pa.Array]):
    r"""Implement an equality tester for ```pyarrow.Array``s and
    ``pyarrow.Table``s.

    Note that ``config.equal_nan``, ``config.atol`` and ``config.rtol``
    arguments are ignored.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.tester import PyarrowEqualityTester
        >>> config = EqualityConfig2()
        >>> tester = PyarrowEqualityTester()
        >>> tester.equal(pa.array([1, 2, 3]), pa.array([1, 2, 3]), config)
        True
        >>> tester.equal(pa.array([1, 2, 3]), pa.array([1, 2, 4]), config)
        False

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
        actual: object,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
