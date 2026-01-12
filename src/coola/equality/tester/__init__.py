r"""Contain the equality testers to check if two objects are equal or
not."""

from __future__ import annotations

__all__ = [
    "BaseEqualityTester",
    "DefaultEqualityTester",
    "EqualityTesterRegistry",
    "HandlerEqualityTester",
    "PandasDataFrameEqualityTester",
    "PandasSeriesEqualityTester",
    "PolarsDataFrameEqualityTester",
    "PolarsLazyFrameEqualityTester",
    "PolarsSeriesEqualityTester",
    "ScalarEqualityTester",
    "get_default_registry",
    "register_equality_testers",
]

from coola.equality.tester.base import BaseEqualityTester
from coola.equality.tester.default import DefaultEqualityTester
from coola.equality.tester.handler import HandlerEqualityTester
from coola.equality.tester.interface import (
    get_default_registry,
    register_equality_testers,
)
from coola.equality.tester.pandas import (
    PandasDataFrameEqualityTester,
    PandasSeriesEqualityTester,
)
from coola.equality.tester.polars import (
    PolarsDataFrameEqualityTester,
    PolarsLazyFrameEqualityTester,
    PolarsSeriesEqualityTester,
)
from coola.equality.tester.registry import EqualityTesterRegistry
from coola.equality.tester.scalar import ScalarEqualityTester
