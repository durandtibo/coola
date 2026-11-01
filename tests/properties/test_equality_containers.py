r"""Property-based tests that the same numbers give the same answer in a
list, a NumPy array, a torch tensor, a pandas series, a polars series, a
pyarrow array and an xarray data array.

The list is the reference: its answer is the one of the numeric rule of
the user guide.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import xarray as xr

from coola.equality import objects_are_allclose, objects_are_equal

pytest.importorskip("hypothesis")

from hypothesis import assume, given, strategies as st

torch = pytest.importorskip("torch")

floats = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
# A new NaN object per draw: a shared one is the same object, which the list treats as equal.
nan_floats = st.one_of(floats, st.builds(float, st.just("nan")))
ints = st.integers(min_value=-(2**31), max_value=2**31)
big_ints = st.integers(min_value=2**53, max_value=2**62)
tolerances = st.sampled_from([0.0, 1e-8, 0.5, 1.0, 10.0])
relative_tolerances = st.sampled_from([0.0, 1e-5, 0.1])

FLOAT_BUILDERS = {
    "list": list,
    "numpy": lambda v: np.array(v, dtype=np.float64),
    "torch": lambda v: torch.tensor(v, dtype=torch.float64),
    "pandas": lambda v: pd.Series(v, dtype="float64"),
    "polars": lambda v: pl.Series(v, dtype=pl.Float64),
    "pyarrow": lambda v: pa.array(v, type=pa.float64()),
    "xarray": lambda v: xr.DataArray(np.array(v, dtype=np.float64)),
}
INT_BUILDERS = {
    "list": list,
    "numpy": lambda v: np.array(v, dtype=np.int64),
    "torch": lambda v: torch.tensor(v, dtype=torch.int64),
    "pandas": lambda v: pd.Series(v, dtype="int64"),
    "polars": lambda v: pl.Series(v, dtype=pl.Int64),
    "pyarrow": lambda v: pa.array(v, type=pa.int64()),
    "xarray": lambda v: xr.DataArray(np.array(v, dtype=np.int64)),
}
NAMES = sorted(FLOAT_BUILDERS)
OTHERS = [name for name in NAMES if name != "list"]


###################################
#     Copies and symmetry         #
###################################


@pytest.mark.parametrize("name", NAMES)
@given(values=st.lists(floats, max_size=6))
def test_float_container_equals_its_copy(name: str, values: list[float]) -> None:
    obj = FLOAT_BUILDERS[name](values)
    assert objects_are_equal(obj, copy.deepcopy(obj)) is True


@pytest.mark.parametrize("name", NAMES)
@given(values=st.lists(nan_floats, max_size=6))
def test_float_container_with_nan_equals_its_copy_with_equal_nan(
    name: str, values: list[float]
) -> None:
    obj = FLOAT_BUILDERS[name](values)
    assert objects_are_equal(obj, copy.deepcopy(obj), equal_nan=True) is True


@pytest.mark.parametrize("name", NAMES)
@given(values=st.lists(ints, max_size=6))
def test_int_container_equals_its_copy(name: str, values: list[int]) -> None:
    obj = INT_BUILDERS[name](values)
    assert objects_are_equal(obj, copy.deepcopy(obj)) is True


@pytest.mark.parametrize("name", NAMES)
@given(
    pairs=st.lists(st.tuples(nan_floats, nan_floats), min_size=1, max_size=6),
    rtol=relative_tolerances,
    atol=tolerances,
    equal_nan=st.booleans(),
)
def test_float_container_is_symmetric(
    name: str, pairs: list[tuple[float, float]], rtol: float, atol: float, equal_nan: bool
) -> None:
    a = FLOAT_BUILDERS[name]([p[0] for p in pairs])
    b = FLOAT_BUILDERS[name]([p[1] for p in pairs])
    assert objects_are_allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan) == (
        objects_are_allclose(b, a, rtol=rtol, atol=atol, equal_nan=equal_nan)
    )


@pytest.mark.parametrize("name", NAMES)
@given(
    pairs=st.lists(st.tuples(ints, ints), min_size=1, max_size=6),
    rtol=relative_tolerances,
    atol=tolerances,
)
def test_int_container_is_symmetric(
    name: str, pairs: list[tuple[int, int]], rtol: float, atol: float
) -> None:
    a = INT_BUILDERS[name]([p[0] for p in pairs])
    b = INT_BUILDERS[name]([p[1] for p in pairs])
    assert objects_are_allclose(a, b, rtol=rtol, atol=atol) == objects_are_allclose(
        b, a, rtol=rtol, atol=atol
    )


#######################################
#     Agreement with the list         #
#######################################


@pytest.mark.parametrize("name", OTHERS)
@given(
    pairs=st.lists(st.tuples(nan_floats, nan_floats), min_size=1, max_size=6),
    rtol=relative_tolerances,
    atol=tolerances,
    equal_nan=st.booleans(),
)
def test_float_container_agrees_with_list(
    name: str, pairs: list[tuple[float, float]], rtol: float, atol: float, equal_nan: bool
) -> None:
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    expected = objects_are_allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    build = FLOAT_BUILDERS[name]
    assert (
        objects_are_allclose(build(a), build(b), rtol=rtol, atol=atol, equal_nan=equal_nan)
        == expected
    )


@pytest.mark.parametrize("name", OTHERS)
@given(
    pairs=st.lists(st.tuples(ints, ints), min_size=1, max_size=6),
    rtol=relative_tolerances,
    atol=tolerances,
)
def test_int_container_agrees_with_list(
    name: str, pairs: list[tuple[int, int]], rtol: float, atol: float
) -> None:
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    expected = objects_are_allclose(a, b, rtol=rtol, atol=atol)
    build = INT_BUILDERS[name]
    assert objects_are_allclose(build(a), build(b), rtol=rtol, atol=atol) == expected


@pytest.mark.parametrize("name", OTHERS)
@given(pairs=st.lists(st.tuples(nan_floats, nan_floats), min_size=1, max_size=6))
def test_exact_float_container_agrees_with_list(
    name: str, pairs: list[tuple[float, float]]
) -> None:
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    build = FLOAT_BUILDERS[name]
    assert objects_are_equal(build(a), build(b)) == objects_are_equal(a, b)


#######################################
#     Fixed defects (regressions)     #
#######################################


@pytest.mark.parametrize("name", OTHERS)
@given(value=big_ints)
def test_large_int_difference_is_exact(name: str, value: int) -> None:
    assume(value + 1 <= 2**62)
    build = INT_BUILDERS[name]
    assert not objects_are_allclose(build([value]), build([value + 1]), rtol=0.0, atol=0.5)


def test_pandas_booleans_are_exact() -> None:
    assert not objects_are_allclose(pd.Series([True]), pd.Series([False]), atol=1.0)
