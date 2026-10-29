r"""Implement an equality tester for SciPy sparse matrices and arrays.

This module provides an equality tester for the ``scipy.sparse``
matrices (``csr_matrix``...) and arrays (``csr_array``...).
"""

from __future__ import annotations

__all__ = ["ScipySparseEqualityTester", "get_default_testers"]

from typing import TYPE_CHECKING, Any

from coola.equality._helpers.format import format_value_difference
from coola.equality._helpers.numpy import array_close, array_equal
from coola.equality.tester._array import check_dtype_and_shape
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_numpy, check_scipy, is_numpy_available, is_scipy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING or is_scipy_available():
    import scipy.sparse as sp
else:  # pragma: no cover
    from coola.utils.fallback.scipy import scipy

    sp = scipy.sparse

if TYPE_CHECKING:
    from coola.equality.engine import Engine


class ScipySparseEqualityTester(BaseEqualityTester[Any]):
    r"""Implement an equality tester for ``scipy.sparse`` matrices and
    arrays.

    Two sparse objects are equal if they have the same class (a matrix
    is not equal to an array, and the format is part of the class:
    ``csr_matrix``, ``csc_matrix``...; the engine checks it), the same
    shape, the same data type, the same non-zero positions and
    equal values. The values are compared with the tolerance and
    ``equal_nan`` of the comparison. The duplicate entries are summed
    and the explicitly stored zeros are ignored, so two objects that
    represent the same matrix are equal. The tolerance applies to the
    stored values: a stored value is never compared with an absent one.
    The location of a difference is the coordinates of the first value
    that differs.

    Example:
        ```pycon
        >>> from scipy import sparse
        >>> from coola.equality import compare
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import ScipySparseEqualityTester
        >>> engine = Engine(EqualityConfig())
        >>> tester = ScipySparseEqualityTester()
        >>> tester.objects_are_equal(sparse.eye(3, format="csr"), sparse.eye(3, format="csr"), engine)
        True
        >>> tester.objects_are_equal(
        ...     sparse.eye(3, format="csr"), 2 * sparse.eye(3, format="csr"), engine
        ... )
        False
        >>> compare(sparse.eye(3, format="csr"), 2 * sparse.eye(3, format="csr")).path
        (0, 0)

        ```
    """

    def __init__(self) -> None:
        check_numpy()
        check_scipy()

    def objects_are_equal(self, actual: Any, expected: Any, ctx: Engine) -> bool | Any:
        failure = check_dtype_and_shape(actual, expected, ctx)
        if failure is not None:
            return failure
        coo1, coo2 = _canonical_coo(actual), _canonical_coo(expected)
        if len(coo1.data) != len(coo2.data) or not all(
            np.array_equal(c1, c2) for c1, c2 in zip(coo1.coords, coo2.coords, strict=True)
        ):
            return ctx.fail("objects have different non-zero positions")
        if array_equal(coo1.data, coo2.data, ctx.config):
            return True
        if not ctx.explaining:
            return False
        # The index is looked up only when the difference is explained.
        i = int(np.argmin(array_close(coo1.data, coo2.data, ctx.config)))
        coords = tuple(int(c[i]) for c in coo1.coords)
        return ctx.fail(
            lambda: format_value_difference(coo1.data[i], coo2.data[i], name="values"), *coords
        )


def _canonical_coo(matrix: Any) -> Any:
    r"""Convert a sparse matrix or array to a canonical COO object.

    The result is a copy: the duplicate entries are summed, the
    coordinates are sorted and the stored zeros are removed.

    Args:
        matrix: The sparse matrix or array to convert.

    Returns:
        The canonical COO object. The input is not modified.

    Example:
        ```pycon
        >>> from scipy import sparse
        >>> from coola.equality.tester.scipy import _canonical_coo
        >>> _canonical_coo(sparse.eye(2, format="csr")).format
        'coo'

        ```
    """
    coo = matrix.tocoo(copy=True)
    coo.sum_duplicates()
    coo.eliminate_zeros()
    return coo


def get_default_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the default equality testers of the SciPy sparse matrices
    and arrays.

    Returns:
        The equality tester of the SciPy sparse matrices and arrays and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the equality testers of SciPy.
    """
    if not (is_numpy_available() and is_scipy_available()):
        return {}

    tester = ScipySparseEqualityTester()
    return {sp.sparray: tester, sp.spmatrix: tester}
