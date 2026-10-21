r"""Implement an equality tester for dataclass instances."""

from __future__ import annotations

__all__ = ["DataclassEqualityTester"]

from typing import TYPE_CHECKING, Any

from coola.equality.handler import (
    DataclassSameFieldsHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
    create_chain,
)
from coola.equality.tester.chain import BaseChainEqualityTester

if TYPE_CHECKING:
    from coola.equality.handler import BaseEqualityHandler


class DataclassEqualityTester(BaseChainEqualityTester[Any]):
    r"""Implement an equality tester for dataclass instances.

    The two instances must have the same type and equal fields. The
    fields are compared with the registry, so they support the
    tolerance and ``equal_nan``.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import DataclassEqualityTester
        >>> @dataclass
        ... class Point:
        ...     x: float
        ...
        >>> config = EqualityConfig()
        >>> tester = DataclassEqualityTester()
        >>> tester.objects_are_equal(Point(1.0), Point(1.0), config)
        True

        ```
    """

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            DataclassSameFieldsHandler(),
            TrueHandler(),
        )
