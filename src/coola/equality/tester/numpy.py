r"""Implement equality testers for NumPy arrays.

This module provides equality testers for numpy.ndarray and
numpy.ma.MaskedArray with support for NaN equality, tolerance-based
comparisons, and dtype/shape checking.
"""

from __future__ import annotations

__all__ = [
    "NumpyArrayEqualityTester",
    "NumpyMaskedArrayEqualityTester",
    "NumpyTimeScalarEqualityTester",
]

from itertools import count
from typing import TYPE_CHECKING, Any

from coola.equality._helpers.format import format_value_difference
from coola.equality._helpers.numpy import array_close, array_equal, time_scalar_equal
from coola.equality._helpers.utils import AttributeName
from coola.equality.tester._array import check_dtype_and_shape
from coola.equality.tester.base import BaseEqualityTester
from coola.equality.tester.scalar import ScalarEqualityTester
from coola.utils.imports import check_numpy, is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.equality.engine import Engine


class NumpyArrayEqualityTester(BaseEqualityTester[np.ndarray]):
    r"""Implement an equality tester for ``numpy.ndarray``.

    Two arrays are equal if they have the same data type, the same
    shape and equal elements. The numbers are compared with the
    tolerance and ``equal_nan`` of the comparison (a boolean, a string
    and an instant are exact; ``atol`` of a duration is a number of
    seconds). The elements of an ``object`` array are compared by the
    engine, one by one. The location of a difference is the index of
    the first element that differs (the position in the flattened
    array for an ``object`` array).

    Raises:
        MissingPackageError: If ``numpy`` is not installed.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality import compare
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import NumpyArrayEqualityTester, get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> tester = NumpyArrayEqualityTester()
        >>> tester.objects_are_equal(np.ones((2, 3)), np.ones((2, 3)), engine)
        True
        >>> tester.objects_are_equal(np.ones((2, 3)), np.zeros((2, 3)), engine)
        False
        >>> compare(np.ones((2, 3)), np.zeros((2, 3))).path
        (0, 0)

        ```
    """

    def __init__(self) -> None:
        check_numpy()

    def objects_are_equal(
        self, actual: np.ndarray, expected: np.ndarray, ctx: Engine
    ) -> bool | Iterator[tuple[object, object, object]]:
        r"""Test two NumPy arrays.

        The arrays must have the same dtype and shape.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``False`` if the arrays differ, ``True`` if they are equal,
                or, for an array of objects, the pairs of elements that
                must all be equal.
        """
        failure = check_dtype_and_shape(actual, expected, ctx)
        if failure is not None:
            return failure
        if actual.dtype.kind == "O":
            # The elements are objects: they are children of the engine.
            return zip(actual.ravel().tolist(), expected.ravel().tolist(), count(), strict=False)
        if array_equal(actual, expected, ctx.config):
            return True
        if not ctx.explaining:
            return False
        # The index is looked up only when the difference is explained.
        index = tuple(int(i) for i in np.argwhere(~array_close(actual, expected, ctx.config))[0])
        return ctx.fail(
            lambda: format_value_difference(actual[index], expected[index], name="values"), *index
        )


class NumpyMaskedArrayEqualityTester(BaseEqualityTester[np.ma.MaskedArray]):
    r"""Implement an equality tester for ``numpy.ma.MaskedArray``.

    Two masked arrays are equal if they have the same data type, the
    same shape, the same data (the masked values are compared too), the
    same mask and the same ``fill_value``. The location of a difference
    is ``.data``, ``.mask`` or ``.fill_value``, and then the index in
    the array.

    Raises:
        MissingPackageError: If ``numpy`` is not installed.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> engine.run(
        ...     np.ma.array([1, 2, 3], mask=[0, 0, 1]), np.ma.array([1, 2, 3], mask=[0, 0, 1])
        ... )
        True
        >>> engine.run(
        ...     np.ma.array([1, 2, 3], mask=[0, 0, 1]), np.ma.array([1, 2, 3], mask=[0, 1, 1])
        ... )
        False

        ```
    """

    def __init__(self) -> None:
        check_numpy()

    def objects_are_equal(
        self, actual: np.ma.MaskedArray, expected: np.ma.MaskedArray, ctx: Engine
    ) -> bool | Iterator[tuple[object, object, object]]:
        r"""Test two NumPy masked arrays.

        The arrays must have the same dtype and shape.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``False`` if the dtypes or shapes differ, otherwise the
                pairs of ``data``, ``mask`` and ``fill_value`` that must
                all be equal.
        """
        failure = check_dtype_and_shape(actual, expected, ctx)
        if failure is not None:
            return failure
        return iter(
            (
                (actual.data, expected.data, AttributeName("data")),
                (actual.mask, expected.mask, AttributeName("mask")),
                (actual.fill_value, expected.fill_value, AttributeName("fill_value")),
            )
        )


class NumpyTimeScalarEqualityTester(BaseEqualityTester[object]):
    r"""Implement an equality tester for ``numpy.timedelta64`` and
    ``numpy.datetime64`` scalars.

    ``NaT`` is equal to ``NaT`` only if ``equal_nan`` is ``True``. A
    ``numpy.timedelta64`` is a duration: the tolerances apply (``atol``
    is a number of seconds). A ``numpy.datetime64`` is an instant and
    is compared exactly.

    Raises:
        MissingPackageError: If ``numpy`` is not installed.

    Example:
        ```pycon
        >>> from coola.equality import Tolerance
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import NumpyTimeScalarEqualityTester, get_default_registry
        >>> tester = NumpyTimeScalarEqualityTester()
        >>> engine = Engine(EqualityConfig(tolerance=Tolerance(atol=1.0)))
        >>> tester.objects_are_equal(np.timedelta64(1, "s"), np.timedelta64(2000, "ms"), engine)
        True
        >>> tester.objects_are_equal(
        ...     np.datetime64("2020-01-01"), np.datetime64("2020-01-02"), engine
        ... )
        False

        ```
    """

    def __init__(self) -> None:
        check_numpy()

    def objects_are_equal(self, actual: Any, expected: Any, ctx: Engine) -> bool:
        r"""Test two NumPy ``datetime64`` or ``timedelta64`` scalars.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the objects are equal, otherwise ``False`` (the
                reason is recorded in the engine).
        """
        return time_scalar_equal(actual, expected, ctx.config) or ctx.fail(
            lambda: format_value_difference(actual, expected, name="numpy time scalars")
        )


def get_default_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the default equality testers of the NumPy arrays and scalars.

    Returns:
        The equality testers of the NumPy arrays and scalars and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the equality testers of the NumPy arrays and scalars.
    """
    if not is_numpy_available():
        return {}

    scalar = ScalarEqualityTester()
    time_scalar = NumpyTimeScalarEqualityTester()
    return {
        # NumPy scalars use the scalar tester to get the tolerance and ``equal_nan``;
        # without it, ``==`` is used and both are ignored. ``np.float64`` is also a
        # ``float``, the other types (``np.float32``, ``np.int64``...) are not.
        # ``np.bool_`` is not registered: a boolean has no tolerance.
        np.floating: scalar,
        np.integer: scalar,
        np.complexfloating: scalar,
        # ``np.timedelta64`` is a subclass of ``np.signedinteger``: it needs its own tester,
        # which must be more specific than ``np.integer`` (the MRO decides).
        np.timedelta64: time_scalar,
        np.datetime64: time_scalar,
        np.ndarray: NumpyArrayEqualityTester(),
        np.ma.MaskedArray: NumpyMaskedArrayEqualityTester(),
    }
