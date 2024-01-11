r"""Implement scalar equality comparators."""

from __future__ import annotations

__all__ = ["FloatEqualityComparator", "get_type_comparator_mapping"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    SameObjectHandler,
    SameTypeHandler,
    ScalarEqualHandler,
)

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class FloatEqualityComparator(BaseEqualityComparator[Any]):
    r"""Implement a default equality comparator.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import FloatEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = FloatEqualityComparator()
    >>> comparator.equal(42.0, 42.0, config)
    True
    >>> comparator.equal(42.0, 1.0, config)
    False

    ```
    """

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(ScalarEqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> FloatEqualityComparator:
        return self.__class__()

    def equal(self, object1: Any, object2: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(object1=object1, object2=object2, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a mapping between the types and the equality comparators.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon
    >>> from coola.equality.comparators.scalar import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'float'>: FloatEqualityComparator()}

    ```
    """
    return {float: FloatEqualityComparator()}
