r"""Implement an equality comparator for ``pyarrow.Array``s and
``pyarrow.Table``s."""

from __future__ import annotations

__all__ = ["PyarrowEqualityComparator", "get_type_comparator_mapping"]

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    PyarrowEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.utils.imports import check_pyarrow, is_pyarrow_available

if is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    pa = Mock()

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class PyarrowEqualityComparator(BaseEqualityComparator[pa.Array]):
    r"""Implement an equality comparator for ```pyarrow.Array``s and
    ``pyarrow.Table``s.

    Note that ``config.equal_nan``, ``config.atol`` and ``config.rtol``
    arguments are ignored.

    Example usage:

    ```pycon

    >>> import pyarrow as pa
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import PyarrowEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = PyarrowEqualityComparator()
    >>> comparator.equal(pa.array([1, 2, 3]), pa.array([1, 2, 3]), config)
    True
    >>> comparator.equal(pa.array([1, 2, 3]), pa.array([1, 2, 4]), config)
    False

    ```
    """

    def __init__(self) -> None:
        check_pyarrow()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(PyarrowEqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> PyarrowEqualityComparator:
        return self.__class__()

    def equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a default mapping between the types and the equality
    comparators.

    This function returns an empty dictionary if ``pyarrow`` is not
    installed.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon

    >>> from coola.equality.comparators.pyarrow_ import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'pyarrow.lib.Array'>: PyarrowEqualityComparator(),
     <class 'pyarrow.lib.Table'>: PyarrowEqualityComparator()}

    ```
    """
    if not is_pyarrow_available():
        return {}
    return {
        pa.Array: PyarrowEqualityComparator(),
        pa.Table: PyarrowEqualityComparator(),
    }
