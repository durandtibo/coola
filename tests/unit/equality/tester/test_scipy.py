from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.equality import compare, objects_are_allclose, objects_are_equal
from coola.equality.config import EqualityConfig
from coola.equality.engine import Engine
from coola.equality.tester import ScipySparseEqualityTester
from coola.equality.tester.scipy import (
    _canonical_coo,
    get_default_testers,
)
from coola.equality.tolerance import Tolerance
from coola.testing.marks import numpy_available, scipy_available
from coola.utils.imports import MissingPackageError, is_numpy_available, is_scipy_available
from tests.unit.equality.utils import explaining, messages, run_tester

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_scipy_available():
    from scipy import sparse
else:
    sparse = Mock()


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


def make_csr(data: list[float], dtype: str = "float64") -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (np.array(data, dtype=dtype), ([0, 1, 2][: len(data)], [0, 1, 2][: len(data)])),
        shape=(3, 3),
    )


######################################################
#     Tests for ScipySparseEqualityTester     #
######################################################


@scipy_available
def test_scipy_sparse_equality_tester_repr() -> None:
    assert repr(ScipySparseEqualityTester()).startswith("ScipySparseEqualityTester(")


@scipy_available
def test_scipy_sparse_equality_tester_equal_true() -> None:
    assert ScipySparseEqualityTester() == ScipySparseEqualityTester()


@scipy_available
def test_scipy_sparse_equality_tester_equal_false_different_type() -> None:
    assert ScipySparseEqualityTester() != 42


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_true_same_object(config: EqualityConfig) -> None:
    x = make_csr([1.0, 2.0])
    assert run_tester(ScipySparseEqualityTester(), x, x, config)


@scipy_available
@numpy_available
@pytest.mark.parametrize("fmt", ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"])
def test_scipy_sparse_equality_tester_true_format(fmt: str, config: EqualityConfig) -> None:
    x = sparse.eye(3, format=fmt)
    assert run_tester(ScipySparseEqualityTester(), x, sparse.eye(3, format=fmt), config)


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_true_sparse_array(config: EqualityConfig) -> None:
    assert run_tester(
        ScipySparseEqualityTester(),
        sparse.csr_array(sparse.eye(3)),
        sparse.csr_array(sparse.eye(3)),
        config,
    )


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_true_1d_coo_array(config: EqualityConfig) -> None:
    assert run_tester(
        ScipySparseEqualityTester(),
        sparse.coo_array(np.array([0.0, 1.0, 0.0])),
        sparse.coo_array(np.array([0.0, 1.0, 0.0])),
        config,
    )


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_true_empty(config: EqualityConfig) -> None:
    assert run_tester(
        ScipySparseEqualityTester(),
        sparse.csr_matrix((3, 3)),
        sparse.csr_matrix((3, 3)),
        config,
    )


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_true_duplicate_entries(config: EqualityConfig) -> None:
    x = sparse.coo_matrix(([1.0, 1.0], ([0, 0], [0, 0])), shape=(2, 2))
    y = sparse.coo_matrix(([2.0], ([0], [0])), shape=(2, 2))
    assert run_tester(ScipySparseEqualityTester(), x, y, config)


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_true_explicit_zero(config: EqualityConfig) -> None:
    x = sparse.coo_matrix(([1.0, 0.0], ([0, 1], [0, 1])), shape=(2, 2))
    y = sparse.coo_matrix(([1.0], ([0], [0])), shape=(2, 2))
    assert run_tester(ScipySparseEqualityTester(), x, y, config)


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_true_unsorted(config: EqualityConfig) -> None:
    x = sparse.coo_matrix(([1.0, 2.0], ([0, 1], [0, 1])), shape=(2, 2))
    y = sparse.coo_matrix(([2.0, 1.0], ([1, 0], [1, 0])), shape=(2, 2))
    assert run_tester(ScipySparseEqualityTester(), x, y, config)


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_false_different_values(config: EqualityConfig) -> None:
    assert not run_tester(
        ScipySparseEqualityTester(), make_csr([1.0, 2.0]), make_csr([1.0, 3.0]), config
    )


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_false_different_positions(config: EqualityConfig) -> None:
    x = sparse.coo_matrix(([1.0], ([0], [0])), shape=(2, 2))
    y = sparse.coo_matrix(([1.0], ([0], [1])), shape=(2, 2))
    assert not run_tester(ScipySparseEqualityTester(), x, y, config)


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_false_different_number_of_values(
    config: EqualityConfig,
) -> None:
    assert not run_tester(
        ScipySparseEqualityTester(), make_csr([1.0, 2.0]), make_csr([1.0]), config
    )


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_false_different_format(config: EqualityConfig) -> None:
    assert not run_tester(
        ScipySparseEqualityTester(),
        sparse.eye(3, format="csr"),
        sparse.eye(3, format="csc"),
        config,
    )


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_false_matrix_vs_array(config: EqualityConfig) -> None:
    assert not run_tester(
        ScipySparseEqualityTester(),
        sparse.csr_matrix(sparse.eye(3)),
        sparse.csr_array(sparse.eye(3)),
        config,
    )


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_false_different_shape(config: EqualityConfig) -> None:
    assert not run_tester(
        ScipySparseEqualityTester(),
        sparse.eye(3, format="csr"),
        sparse.eye(4, format="csr"),
        config,
    )


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_false_different_dtype(config: EqualityConfig) -> None:
    assert not run_tester(
        ScipySparseEqualityTester(),
        make_csr([1.0, 2.0]),
        make_csr([1.0, 2.0], dtype="float32"),
        config,
    )


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_false_does_not_log(config: EqualityConfig) -> None:
    run_tester(ScipySparseEqualityTester(), make_csr([1.0]), make_csr([2.0]), config)
    assert not messages(config)


@scipy_available
@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected", "message"),
    [
        pytest.param(
            lambda: sparse.csr_matrix(sparse.eye(2)),
            lambda: sparse.csr_array(sparse.eye(2)),
            "objects have different types",
            id="class",
        ),
        pytest.param(
            lambda: sparse.eye(2, format="csr"),
            lambda: sparse.eye(2, format="csc"),
            "objects have different types",
            id="format",
        ),
        pytest.param(
            lambda: sparse.eye(2, format="csr"),
            lambda: sparse.eye(3, format="csr"),
            "objects have different shapes",
            id="shape",
        ),
        pytest.param(
            lambda: make_csr([1.0]),
            lambda: make_csr([1.0], dtype="float32"),
            "objects have different data types: float64 vs float32",
            id="dtype",
        ),
        pytest.param(
            lambda: sparse.coo_matrix(([1.0], ([0], [0])), shape=(2, 2)),
            lambda: sparse.coo_matrix(([1.0], ([0], [1])), shape=(2, 2)),
            "objects have different non-zero positions",
            id="positions",
        ),
        pytest.param(
            lambda: make_csr([1.0, 2.0]),
            lambda: make_csr([1.0, 3.0]),
            "values are different",
            id="values",
        ),
    ],
)
def test_scipy_sparse_equality_tester_false_explaining_reports(
    actual: Callable[[], object],
    expected: Callable[[], object],
    message: str,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    run_tester(ScipySparseEqualityTester(), actual(), expected(), config)
    assert messages(config)[0].startswith(message)


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_equal_nan_false(config: EqualityConfig) -> None:
    x = make_csr([float("nan")])
    assert not run_tester(ScipySparseEqualityTester(), x, make_csr([float("nan")]), config)


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_equal_nan_true(config: EqualityConfig) -> None:
    config = replace(config, equal_nan=True)
    assert run_tester(
        ScipySparseEqualityTester(), make_csr([float("nan")]), make_csr([float("nan")]), config
    )


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_tolerance(config: EqualityConfig) -> None:
    config = replace(config, tolerance=Tolerance(atol=0.1))
    assert run_tester(ScipySparseEqualityTester(), make_csr([1.0]), make_csr([1.05]), config)


@scipy_available
@numpy_available
def test_scipy_sparse_equality_tester_tolerance_exceeded(config: EqualityConfig) -> None:
    config = replace(config, tolerance=Tolerance(atol=0.01))
    assert not run_tester(ScipySparseEqualityTester(), make_csr([1.0]), make_csr([1.05]), config)


@scipy_available
def test_scipy_sparse_equality_tester_no_scipy() -> None:
    with (
        patch("coola.utils.imports.optional.is_scipy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'scipy' package is required but not installed."),
    ):
        ScipySparseEqualityTester()


@numpy_available
def test_scipy_sparse_equality_tester_no_numpy() -> None:
    with (
        patch("coola.utils.imports.optional.is_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'numpy' package is required but not installed."),
    ):
        ScipySparseEqualityTester()


#####################################
#     Tests for compare / API       #
#####################################


@scipy_available
@numpy_available
def test_objects_are_equal_copy() -> None:
    x = sparse.csr_matrix(sparse.eye(3))
    assert objects_are_equal(x, x.copy())


@scipy_available
@numpy_available
def test_objects_are_equal_different() -> None:
    assert not objects_are_equal(sparse.csr_matrix(sparse.eye(3)), sparse.csr_matrix((3, 3)))


@scipy_available
@numpy_available
def test_objects_are_equal_in_a_list() -> None:
    assert objects_are_equal([sparse.eye(2, format="csr")], [sparse.eye(2, format="csr")])


@scipy_available
@numpy_available
def test_objects_are_allclose_atol() -> None:
    assert objects_are_allclose(make_csr([1.0]), make_csr([1.05]), atol=0.1)


@scipy_available
@numpy_available
def test_compare_path_is_the_coordinates() -> None:
    assert compare(make_csr([1.0, 2.0]), make_csr([1.0, 3.0])).path == (1, 1)


@scipy_available
@numpy_available
def test_compare_reason() -> None:
    result = compare(make_csr([1.0, 2.0]), make_csr([1.0, 3.0]))
    assert result.reason is not None
    assert result.reason.startswith("values are different:")


@scipy_available
@numpy_available
def test_engine_resolves_sparse_matrix_and_array() -> None:
    assert Engine(EqualityConfig()).run(sparse.eye(2, format="csr"), sparse.eye(2, format="csr"))


#####################################
#     Tests for _canonical_coo      #
#####################################


@scipy_available
@numpy_available
def test_canonical_coo_format() -> None:
    assert _canonical_coo(sparse.eye(2, format="csr")).format == "coo"


@scipy_available
@numpy_available
def test_canonical_coo_sums_duplicates() -> None:
    x = sparse.coo_matrix(([1.0, 1.0], ([0, 0], [0, 0])), shape=(2, 2))
    assert _canonical_coo(x).data.tolist() == [2.0]


@scipy_available
@numpy_available
def test_canonical_coo_removes_stored_zeros() -> None:
    x = sparse.coo_matrix(([1.0, 0.0], ([0, 1], [0, 1])), shape=(2, 2))
    assert _canonical_coo(x).data.tolist() == [1.0]


@scipy_available
@numpy_available
def test_canonical_coo_sorts_coordinates() -> None:
    x = sparse.coo_matrix(([2.0, 1.0], ([1, 0], [1, 0])), shape=(2, 2))
    assert _canonical_coo(x).data.tolist() == [1.0, 2.0]


@scipy_available
@numpy_available
def test_canonical_coo_does_not_modify_input() -> None:
    x = sparse.coo_matrix(([1.0, 1.0], ([0, 0], [0, 0])), shape=(2, 2))
    _canonical_coo(x)
    assert x.data.tolist() == [1.0, 1.0]


############################################
#     Tests for get_default_testers        #
############################################


@scipy_available
@numpy_available
def test_get_default_testers_types() -> None:
    assert set(get_default_testers()) == {sparse.sparray, sparse.spmatrix}


@scipy_available
@numpy_available
def test_get_default_testers_values() -> None:
    assert all(isinstance(t, ScipySparseEqualityTester) for t in get_default_testers().values())


def test_get_default_testers_no_scipy() -> None:
    with patch("coola.equality.tester.scipy.is_scipy_available", lambda: False):
        assert get_default_testers() == {}


def test_get_default_testers_no_numpy() -> None:
    with patch("coola.equality.tester.scipy.is_numpy_available", lambda: False):
        assert get_default_testers() == {}
