r"""Implement equality comparators for ``collections`` objects like
``Sequence`` and ``Mapping``."""

from __future__ import annotations

__all__ = ["MappingEqualityComparator", "SequenceEqualityComparator", "get_type_comparator_mapping"]

import logging
from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    MappingSameKeysHandler,
    MappingSameValuesHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    SequenceSameValuesHandler,
    TrueHandler,
)

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class MappingEqualityComparator(BaseEqualityComparator[Mapping]):
    r"""Implement a sequence equality comparator.

    Example usage:

    ```pycon

    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import MappingEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = MappingEqualityComparator()
    >>> comparator.equal({"a": 1, "b": 2}, {"a": 1, "b": 2}, config)
    True
    >>> comparator.equal({"a": 1, "b": 2}, {"a": 1, "b": 4}, config)
    False

    ```
    """

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameLengthHandler()).chain(
            MappingSameKeysHandler()
        ).chain(MappingSameValuesHandler()).chain(TrueHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> MappingEqualityComparator:
        return self.__class__()

    def equal(self, actual: Mapping, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


class SequenceEqualityComparator(BaseEqualityComparator[Sequence]):
    r"""Implement a sequence equality comparator.

    Example usage:

    ```pycon

    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import SequenceEqualityComparator
    >>> from coola.equality.testers import EqualityTester
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
            SequenceSameValuesHandler()
        ).chain(TrueHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> SequenceEqualityComparator:
        return self.__class__()

    def equal(self, actual: Sequence, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a mapping between the types and the equality comparators.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon

    >>> from coola.equality.comparators.collection import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'collections.abc.Mapping'>: MappingEqualityComparator(),
     <class 'collections.abc.Sequence'>: SequenceEqualityComparator(),
     <class 'collections.deque'>: SequenceEqualityComparator(),
     <class 'dict'>: MappingEqualityComparator(),
     <class 'list'>: SequenceEqualityComparator(),
     <class 'tuple'>: SequenceEqualityComparator()}

    ```
    """
    map_cmp = MappingEqualityComparator()
    seq_cmp = SequenceEqualityComparator()
    return {
        Mapping: map_cmp,
        Sequence: seq_cmp,
        deque: seq_cmp,
        dict: map_cmp,
        list: seq_cmp,
        tuple: seq_cmp,
    }
