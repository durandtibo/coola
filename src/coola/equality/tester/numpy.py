r"""Implement an equality tester for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["NumpyArrayEqualityTester", "NumpyMaskedArrayEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality.handler import (
    NumpyArrayEqualHandler,
    SameAttributeHandler,
    SameDataHandler,
    SameDTypeHandler,
    SameObjectHandler,
    SameShapeHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_numpy, is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class NumpyArrayEqualityTester(BaseEqualityTester[np.ndarray]):
    r"""Implement an equality tester for ``numpy.ndarray``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import NumpyArrayEqualityTester
        >>> config = EqualityConfig()
        >>> tester = NumpyArrayEqualityTester()
        >>> tester.objects_are_equal(np.ones((2, 3)), np.ones((2, 3)), config)
        True
        >>> tester.objects_are_equal(np.ones((2, 3)), np.zeros((2, 3)), config)
        False

        ```
    """

    def __init__(self) -> None:
        check_numpy()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameDTypeHandler()).chain(
            SameShapeHandler()
        ).chain(NumpyArrayEqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: np.ndarray,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class NumpyMaskedArrayEqualityTester(BaseEqualityTester[np.ma.MaskedArray]):
    r"""Implement an equality tester for ``numpy.ma.MaskedArray``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import NumpyMaskedArrayEqualityTester
        >>> config = EqualityConfig()
        >>> tester = NumpyMaskedArrayEqualityTester()
        >>> tester.objects_are_equal(
        ...     np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
        ...     np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
        ...     config,
        ... )
        True
        >>> tester.objects_are_equal(
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
        ).chain(TrueHandler())  # fmt: skip

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: np.ma.MaskedArray,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
