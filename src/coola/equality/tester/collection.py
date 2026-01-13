r"""Implement equality testers for ``collections`` objects like
``Sequence`` and ``Mapping``."""

from __future__ import annotations

__all__ = ["MappingEqualityTester", "SequenceEqualityTester"]

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from coola.equality.handler import (
    MappingSameKeysHandler,
    MappingSameValuesHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    SequenceSameValuesHandler,
    TrueHandler,
)
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class MappingEqualityTester(BaseEqualityTester[Mapping[Any, Any]]):
    r"""Implement a sequence equality tester.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import MappingEqualityTester
        >>> config = EqualityConfig()
        >>> tester = MappingEqualityTester()
        >>> tester.objects_are_equal({"a": 1, "b": 2}, {"a": 1, "b": 2}, config)
        True
        >>> tester.objects_are_equal({"a": 1, "b": 2}, {"a": 1, "b": 4}, config)
        False

        ```
    """

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameLengthHandler()).chain(
            MappingSameKeysHandler()
        ).chain(MappingSameValuesHandler()).chain(TrueHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class SequenceEqualityTester(BaseEqualityTester[Sequence[Any]]):
    r"""Implement a sequence equality tester.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import SequenceEqualityTester
        >>> config = EqualityConfig()
        >>> tester = SequenceEqualityTester()
        >>> tester.objects_are_equal([1, 2, 3], [1, 2, 3], config)
        True
        >>> tester.objects_are_equal([1, 2, 3], [1, 2, 4], config)
        False

        ```
    """

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameLengthHandler()).chain(
            SequenceSameValuesHandler()
        ).chain(TrueHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
