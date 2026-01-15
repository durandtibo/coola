r"""Implement equality testers for NumPy arrays.

This module provides equality testers for numpy.ndarray and
numpy.ma.MaskedArray with support for NaN equality, tolerance-based
comparisons, and dtype/shape checking.
"""

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
    create_chain,
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

    This tester compares NumPy arrays element-wise with support for NaN equality
    and tolerance-based comparisons. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are numpy arrays
    3. SameDTypeHandler: Check arrays have the same dtype
    4. SameShapeHandler: Verify arrays have the same shape
    5. NumpyArrayEqualHandler: Element-wise comparison with tolerance support

    The tester respects config.equal_nan for NaN comparisons and config.atol/rtol
    for floating-point tolerance.

    Example:
        Basic array comparison:

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

        NaN comparison with equal_nan:

        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import NumpyArrayEqualityTester
        >>> config = EqualityConfig(equal_nan=True)
        >>> tester = NumpyArrayEqualityTester()
        >>> tester.objects_are_equal(
        ...     np.array([1.0, float("nan")]),
        ...     np.array([1.0, float("nan")]),
        ...     config,
        ... )
        True

        ```
    """

    def __init__(self) -> None:
        check_numpy()
        self._handler = create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameDTypeHandler(),
            SameShapeHandler(),
            NumpyArrayEqualHandler(),
        )

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

    This tester compares NumPy masked arrays by checking data, mask, and fill_value.
    The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are masked arrays
    3. SameDTypeHandler: Check arrays have the same dtype
    4. SameShapeHandler: Verify arrays have the same shape
    5. SameDataHandler: Compare the underlying data arrays
    6. SameAttributeHandler("mask"): Compare the mask arrays
    7. SameAttributeHandler("fill_value"): Compare fill values
    8. TrueHandler: Return True if all checks pass

    Example:
        Basic masked array comparison:

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

        Different masks are not equal:

        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import NumpyMaskedArrayEqualityTester
        >>> config = EqualityConfig()
        >>> tester = NumpyMaskedArrayEqualityTester()
        >>> tester.objects_are_equal(
        ...     np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
        ...     np.ma.array(data=[0.0, 1.0, 1.2], mask=[1, 1, 0]),
        ...     config,
        ... )
        False

        ```
    """

    def __init__(self) -> None:
        check_numpy()
        self._handler = create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameDTypeHandler(),
            SameShapeHandler(),
            SameDataHandler(),
            SameAttributeHandler("mask"),
            SameAttributeHandler("fill_value"),
            TrueHandler(),
        )

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
