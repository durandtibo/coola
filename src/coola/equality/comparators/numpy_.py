r"""Implement an equality comparator for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["ArrayEqualityComparator", "get_type_comparator_mapping"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    ArraySameDTypeHandler,
    ArraySameShapeHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.equality.handlers.numpy_ import ArrayEqualHandler
from coola.utils import check_numpy, is_numpy_available

if is_numpy_available():
    import numpy as np

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class ArrayEqualityComparator(BaseEqualityComparator[Any]):
    r"""Implement an equality comparator for ``numpy.ndarray``.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import ArrayEqualityComparator
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = ArrayEqualityComparator()
    >>> comparator.equal(np.ones((2, 3)), np.ones((2, 3)), config)
    True
    >>> comparator.equal(np.ones((2, 3)), np.zeros((2, 3)), config)
    False

    ```
    """

    def __init__(self) -> None:
        check_numpy()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(ArraySameDTypeHandler()).chain(
            ArraySameShapeHandler()
        ).chain(ArrayEqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> ArrayEqualityComparator:
        return self.__class__()

    def equal(self, object1: Any, object2: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(object1=object1, object2=object2, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a default mapping between the types and the equality
    comparators.

    This function returns an empty dictionary if numpy is not
    installed.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon
    >>> from coola.equality.comparators.numpy_ import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'numpy.ndarray'>: ArrayEqualityComparator()}

    ```
    """
    if not is_numpy_available():
        return {}
    return {np.ndarray: ArrayEqualityComparator()}
