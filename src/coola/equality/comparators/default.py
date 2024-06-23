r"""Implement the default equality comparator."""

from __future__ import annotations

__all__ = ["DefaultEqualityComparator", "get_type_comparator_mapping"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    ObjectEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
)

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class DefaultEqualityComparator(BaseEqualityComparator[Any]):
    r"""Implement a default equality comparator.

    The ``==`` operator is used to test the equality between the
    objects.

    Example usage:

    ```pycon

    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import DefaultEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = DefaultEqualityComparator()
    >>> comparator.equal(42, 42, config)
    True
    >>> comparator.equal("meow", "meov", config)
    False

    ```
    """

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(ObjectEqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> DefaultEqualityComparator:
        return self.__class__()

    def equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a default mapping between the types and the equality
    comparators.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon

    >>> from coola.equality.comparators.default import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'object'>: DefaultEqualityComparator()}

    ```
    """
    return {object: DefaultEqualityComparator()}
