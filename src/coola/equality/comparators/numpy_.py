r"""Implement an equality comparator for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = [
    "NumpyArrayEqualityComparator",
    "NumpyMaskedArrayEqualityComparator",
    "get_type_comparator_mapping",
]

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    NumpyArrayEqualHandler,
    SameAttributeHandler,
    SameDataHandler,
    SameDTypeHandler,
    SameObjectHandler,
    SameShapeHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.utils import check_numpy, is_numpy_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class NumpyArrayEqualityComparator(BaseEqualityComparator[np.ndarray]):
    r"""Implement an equality comparator for ``numpy.ndarray``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import NumpyArrayEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = NumpyArrayEqualityComparator()
    >>> comparator.equal(np.ones((2, 3)), np.ones((2, 3)), config)
    True
    >>> comparator.equal(np.ones((2, 3)), np.zeros((2, 3)), config)
    False

    ```
    """

    def __init__(self) -> None:
        check_numpy()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameDTypeHandler()).chain(
            SameShapeHandler()
        ).chain(NumpyArrayEqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> NumpyArrayEqualityComparator:
        return self.__class__()

    def equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


class NumpyMaskedArrayEqualityComparator(BaseEqualityComparator[np.ma.MaskedArray]):
    r"""Implement an equality comparator for ``numpy.ndarray``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import NumpyMaskedArrayEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = NumpyMaskedArrayEqualityComparator()
    >>> comparator.equal(
    ...     np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
    ...     np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
    ...     config,
    ... )
    True
    >>> comparator.equal(
    ...     np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
    ...     np.ma.array(data=[0.0, 1.0, 2.0], mask=[0, 1, 0]),
    ...     config,
    ... )
    False

    ```
    """

    def __init__(self) -> None:
        check_numpy()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameDTypeHandler()).chain(
            SameShapeHandler()
        ).chain(SameDataHandler()).chain(SameAttributeHandler("mask")).chain(
            SameAttributeHandler("fill_value")
        ).chain(
            TrueHandler()
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> NumpyMaskedArrayEqualityComparator:
        return self.__class__()

    def equal(self, actual: np.ma.MaskedArray, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a default mapping between the types and the equality
    comparators.

    This function returns an empty dictionary if ``numpy`` is not
    installed.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon

    >>> from coola.equality.comparators.numpy_ import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'numpy.ndarray'>: NumpyArrayEqualityComparator(),
     <class 'numpy.ma...MaskedArray'>: NumpyMaskedArrayEqualityComparator()}

    ```
    """
    if not is_numpy_available():
        return {}
    return {
        np.ndarray: NumpyArrayEqualityComparator(),
        np.ma.MaskedArray: NumpyMaskedArrayEqualityComparator(),
    }
