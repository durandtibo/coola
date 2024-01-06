r"""Implement equality comparators for ``collections`` objects like
``Sequence`` and ``Mapping``."""

from __future__ import annotations

__all__ = ["SequenceEqualityComparator", "get_type_comparator_mapping"]

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    SequenceSameValueHandler,
    TrueHandler,
)

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class SequenceEqualityComparator(BaseEqualityComparator[Any]):
    r"""Implement a sequence equality comparator.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import SequenceEqualityComparator
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = SequenceEqualityComparator()
    >>> comparator.equal([1, 2, 3], [1, 2, 3], config)
    True
    >>> comparator.equal([1, 2, 3], [1, 2, 4], config)
    False

    ```
    """

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameLengthHandler()).chain(
            SequenceSameValueHandler()
        ).chain(TrueHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> SequenceEqualityComparator:
        return self.__class__()

    def equal(self, object1: Any, object2: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(object1=object1, object2=object2, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a mapping between the types and the equality comparators.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon
    >>> from coola.equality.comparators.collection import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'collections.abc.Sequence'>: SequenceEqualityComparator(),
     <class 'list'>: SequenceEqualityComparator(),
     <class 'tuple'>: SequenceEqualityComparator()}

    ```
    """
    return {
        Sequence: SequenceEqualityComparator(),
        list: SequenceEqualityComparator(),
        tuple: SequenceEqualityComparator(),
    }
