r"""Check that the tolerance and ``equal_nan`` apply to the common
numeric types."""

from __future__ import annotations

from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from coola.equality import compare, objects_are_allclose, objects_are_equal
from coola.testing.marks import (
    numpy_available,
    pandas_available,
    polars_available,
    pyarrow_available,
    torch_available,
)
from coola.utils.imports import (
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_torch_available,
)

if is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    pd = Mock()

if is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    pa = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if is_polars_available():
    import polars as pl
else:  # pragma: no cover
    pl = Mock()

NAN = float("nan")


@dataclass
class Point:
    x: float
    y: float = 0.0


@dataclass
class WithArray:
    data: np.ndarray


@dataclass
class Ignored:
    x: float
    note: str = field(default="", compare=False)


@dataclass(eq=False)
class CustomEq:
    x: float

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = None  # type: ignore[assignment]


@numpy_available
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numpy_float_scalar_tolerance(dtype: type) -> None:
    assert objects_are_allclose(dtype(1.0), dtype(1.01), atol=0.1)


@numpy_available
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numpy_float_scalar_tolerance_different(dtype: type) -> None:
    assert not objects_are_allclose(dtype(1.0), dtype(2.0), atol=0.1)


@numpy_available
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numpy_float_scalar_equal_is_exact(dtype: type) -> None:
    assert not objects_are_equal(dtype(1.0), dtype(1.01))


@numpy_available
@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint32])
def test_numpy_int_scalar_tolerance(dtype: type) -> None:
    assert objects_are_allclose(dtype(10), dtype(11), atol=2)


@numpy_available
@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint32])
def test_numpy_int_scalar_tolerance_different(dtype: type) -> None:
    assert not objects_are_allclose(dtype(10), dtype(20), atol=2)


@numpy_available
@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint32])
def test_numpy_int_scalar_equal(dtype: type) -> None:
    assert objects_are_equal(dtype(10), dtype(10))


@numpy_available
def test_numpy_scalar_returns_true() -> None:
    assert objects_are_equal(np.float32(1.0), np.float32(1.0)) is True


@numpy_available
def test_numpy_scalar_returns_false() -> None:
    assert objects_are_equal(np.float32(1.0), np.float32(2.0)) is False


@numpy_available
def test_numpy_scalar_nan_is_not_equal() -> None:
    assert not objects_are_equal(np.float32("nan"), np.float32("nan"))


@numpy_available
def test_numpy_scalar_equal_nan() -> None:
    assert objects_are_equal(np.float32("nan"), np.float32("nan"), equal_nan=True)


@numpy_available
def test_numpy_scalar_different_types() -> None:
    assert not objects_are_equal(np.float32(1.0), np.float64(1.0))


@numpy_available
def test_numpy_complex_scalar_tolerance() -> None:
    assert objects_are_allclose(np.complex64(1 + 1j), np.complex64(1.01 + 1j), atol=0.1)


@numpy_available
def test_numpy_complex_scalar_tolerance_different() -> None:
    assert not objects_are_allclose(np.complex64(1 + 1j), np.complex64(1 + 2j), atol=0.1)


def test_complex_exact() -> None:
    assert objects_are_equal(1 + 1j, 1 + 1j)


def test_complex_exact_different() -> None:
    assert not objects_are_equal(1 + 1j, 1 + 1.01j)


def test_complex_tolerance() -> None:
    assert objects_are_allclose(1 + 1j, 1 + 1.01j, atol=0.1)


def test_complex_tolerance_different() -> None:
    assert not objects_are_allclose(1 + 1j, 1 + 2j, atol=0.1)


def test_complex_imaginary_part_is_compared() -> None:
    assert not objects_are_allclose(1 + 0j, 1 + 5j, atol=0.1)


def test_complex_nan_is_not_equal() -> None:
    assert not objects_are_equal(complex(NAN, 0), complex(NAN, 0))


def test_complex_equal_nan() -> None:
    assert objects_are_equal(complex(NAN, 0), complex(NAN, 0), equal_nan=True)


def test_complex_equal_nan_real_and_imaginary_parts() -> None:
    assert objects_are_equal(complex(NAN, NAN), complex(NAN, NAN), equal_nan=True)


def test_complex_not_equal_nan_vs_number() -> None:
    assert not objects_are_equal(complex(NAN, 0), 1 + 0j, equal_nan=True)


def test_dataclass_equal() -> None:
    assert objects_are_equal(Point(1.0, 2.0), Point(1.0, 2.0)) is True


def test_dataclass_not_equal() -> None:
    assert objects_are_equal(Point(1.0, 2.0), Point(1.0, 3.0)) is False


def test_dataclass_different_types() -> None:
    assert not objects_are_equal(Point(1.0), Ignored(1.0))


def test_dataclass_tolerance() -> None:
    assert objects_are_allclose(Point(1.0), Point(1.0000001), atol=1e-3)


def test_dataclass_tolerance_different() -> None:
    assert not objects_are_allclose(Point(1.0), Point(1.1), atol=1e-3)


def test_dataclass_nan_is_not_equal() -> None:
    assert not objects_are_equal(Point(float("nan")), Point(float("nan")))


def test_dataclass_equal_nan() -> None:
    assert objects_are_equal(Point(float("nan")), Point(float("nan")), equal_nan=True)


def test_dataclass_nested() -> None:
    assert objects_are_allclose({"p": [Point(1.0)]}, {"p": [Point(1.0000001)]}, atol=1e-3)


def test_dataclass_ignores_compare_false_fields() -> None:
    assert objects_are_equal(Ignored(1.0, "a"), Ignored(1.0, "b"))


@numpy_available
def test_dataclass_with_array() -> None:
    assert objects_are_equal(WithArray(np.ones(3)), WithArray(np.ones(3))) is True


@numpy_available
def test_dataclass_with_array_different() -> None:
    assert objects_are_equal(WithArray(np.ones(3)), WithArray(np.zeros(3))) is False


@numpy_available
def test_dataclass_with_array_tolerance() -> None:
    assert objects_are_allclose(WithArray(np.ones(3)), WithArray(np.ones(3) + 1e-9))


def test_dataclass_with_custom_eq_keeps_its_eq() -> None:
    assert objects_are_equal(CustomEq(1.0), CustomEq(2.0))


def test_dataclass_compare_path() -> None:
    assert compare(Point(1.0, 2.0), Point(1.0, 3.0)).format_path("actual") == "actual.y"


def test_dataclass_class_is_not_an_instance() -> None:
    assert objects_are_equal(Point, Point)


#############################################################
#     Tests for the types compared with their difference    #
#############################################################


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(Decimal("1.0"), Decimal("1.00001"), id="decimal"),
        pytest.param(Fraction(1, 3), Fraction(333333, 1000000), id="fraction"),
        pytest.param(timedelta(seconds=1), timedelta(seconds=1.0001), id="timedelta"),
    ],
)
def test_objects_are_allclose_difference_types_atol(actual: object, expected: object) -> None:
    assert objects_are_allclose(actual, expected, atol=1e-3)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(Decimal("1.0"), Decimal("1.00001"), id="decimal"),
        pytest.param(Fraction(1, 3), Fraction(333333, 1000000), id="fraction"),
        pytest.param(timedelta(seconds=1), timedelta(seconds=1.0001), id="timedelta"),
    ],
)
def test_objects_are_equal_difference_types_ignore_the_default_tolerance(
    actual: object, expected: object
) -> None:
    assert not objects_are_equal(actual, expected)


def test_objects_are_allclose_difference_types_in_a_list() -> None:
    assert objects_are_allclose([Decimal("1.0")], [Decimal("1.00001")], atol=1e-3)


def test_objects_are_equal_decimal_nan_equal_nan() -> None:
    assert objects_are_equal(Decimal("NaN"), Decimal("NaN"), equal_nan=True)


########################################
#     Tests for the tolerance rule     #
########################################

# |1 - 3| = 2 <= 0.5 + 0.5 * 3: the absolute and the relative tolerances add up.
PAIRS_BY_TYPE = {
    "float": (1.0, 3.0),
    "int": (1, 3),
    "complex": (complex(1, 0), complex(3, 0)),
    "Decimal": (Decimal(1), Decimal(3)),
    "Fraction": (Fraction(1), Fraction(3)),
    "timedelta": (timedelta(seconds=1), timedelta(seconds=3)),
    "numpy float64": (np.float64(1), np.float64(3)),
    "numpy array": (np.array([1.0]), np.array([3.0])),
    "list": ([1.0], [3.0]),
}


@pytest.mark.parametrize("pair", PAIRS_BY_TYPE.values(), ids=PAIRS_BY_TYPE.keys())
def test_allclose_atol_and_rtol_add_up(pair: tuple) -> None:
    assert objects_are_allclose(*pair, atol=0.5, rtol=0.5)


@pytest.mark.parametrize("pair", PAIRS_BY_TYPE.values(), ids=PAIRS_BY_TYPE.keys())
def test_allclose_atol_and_rtol_add_up_is_symmetric(pair: tuple) -> None:
    assert objects_are_allclose(*reversed(pair), atol=0.5, rtol=0.5)


################################################
#     Tests for the tolerance of data frames   #
################################################


@polars_available
def test_objects_are_allclose_polars_integers_follow_rtol() -> None:
    assert objects_are_allclose(pl.Series([100]), pl.Series([101]), rtol=0.01, atol=0.0)


@polars_available
def test_objects_are_allclose_polars_integers_follow_atol() -> None:
    assert objects_are_allclose(pl.Series([1]), pl.Series([2]), atol=2.0)


@pandas_available
def test_objects_are_allclose_pandas_and_numpy_agree_on_atol_and_rtol() -> None:
    kwargs: dict[str, Any] = {"atol": 0.5, "rtol": 0.5}
    assert objects_are_allclose(
        pd.Series([0.0]), pd.Series([0.9]), **kwargs
    ) == objects_are_allclose(np.array([0.0]), np.array([0.9]), **kwargs)


@polars_available
def test_objects_are_allclose_polars_and_numpy_agree_on_atol_and_rtol() -> None:
    kwargs: dict[str, Any] = {"atol": 0.5, "rtol": 0.5}
    assert objects_are_allclose(
        pl.Series([0.0]), pl.Series([0.9]), **kwargs
    ) == objects_are_allclose(np.array([0.0]), np.array([0.9]), **kwargs)


@polars_available
def test_objects_are_allclose_polars_lazy_frame_integers() -> None:
    assert objects_are_allclose(
        pl.LazyFrame({"a": [100]}), pl.LazyFrame({"a": [101]}), rtol=0.01, atol=0.0
    )


@pandas_available
def test_objects_are_allclose_pandas_frame_in_a_dict() -> None:
    assert objects_are_allclose(
        {"df": pd.DataFrame({"a": [100, 200]})}, {"df": pd.DataFrame({"a": [101, 201]})}, rtol=0.01
    )


########################################
#     Tests for pandas nulls           #
########################################


@pandas_available
def test_objects_are_equal_pandas_frame_with_a_null_is_equal_to_its_copy() -> None:
    frame = pd.DataFrame({"a": ["x", None], "b": pd.array([1, None], dtype="Int64")})
    assert objects_are_equal(frame, frame.copy())


@pandas_available
def test_objects_are_equal_pandas_frame_with_a_nan_is_not_equal_to_its_copy() -> None:
    frame = pd.DataFrame({"a": [1.0, float("nan")]})
    assert not objects_are_equal(frame, frame.copy())


@pandas_available
def test_objects_are_equal_pandas_frame_with_a_nan_is_equal_to_its_copy_with_equal_nan() -> None:
    frame = pd.DataFrame({"a": [1.0, float("nan")]})
    assert objects_are_equal(frame, frame.copy(), equal_nan=True)


@pandas_available
def test_objects_are_equal_pandas_frame_in_a_dict_with_a_null() -> None:
    frame = pd.DataFrame({"a": ["x", None]})
    assert objects_are_equal({"df": frame}, {"df": frame.copy()})


########################################
#     Tests for NaN dictionary keys    #
########################################


def test_objects_are_equal_nan_keys_equal_nan() -> None:
    assert objects_are_equal({float("nan"): 1}, {float("nan"): 1}, equal_nan=True)


def test_objects_are_equal_nan_keys_not_equal_nan() -> None:
    assert not objects_are_equal({float("nan"): 1}, {float("nan"): 1})


def test_objects_are_equal_nan_keys_different_values_equal_nan() -> None:
    assert not objects_are_equal({float("nan"): 1}, {float("nan"): 2}, equal_nan=True)


def test_objects_are_equal_nan_keys_nested_equal_nan() -> None:
    assert objects_are_equal({"a": {float("nan"): [1]}}, {"a": {float("nan"): [1]}}, equal_nan=True)


def test_objects_are_equal_nan_keys_ordered_dict_equal_nan() -> None:
    assert objects_are_equal(
        OrderedDict([(float("nan"), 1), ("a", 2)]),
        OrderedDict([(float("nan"), 1), ("a", 2)]),
        equal_nan=True,
    )


############################################
#     Tests for NumPy time scalars         #
############################################


@numpy_available
def test_objects_are_allclose_timedelta64_scalar() -> None:
    assert objects_are_allclose(np.timedelta64(1, "s"), np.timedelta64(1, "s"))


@numpy_available
def test_objects_are_allclose_timedelta64_scalar_in_list() -> None:
    assert objects_are_allclose([np.timedelta64(1, "s")], [np.timedelta64(1, "s")])


@numpy_available
def test_objects_are_equal_timedelta64_nat_equal_nan() -> None:
    assert objects_are_equal(np.timedelta64("NaT", "s"), np.timedelta64("NaT", "s"), equal_nan=True)


@numpy_available
def test_objects_are_equal_timedelta64_scalar() -> None:
    assert objects_are_equal(np.timedelta64(1, "s"), np.timedelta64(1, "s"))


@numpy_available
def test_objects_are_equal_timedelta64_scalar_different() -> None:
    assert not objects_are_equal(np.timedelta64(1, "s"), np.timedelta64(2, "s"))


########################################
#     Tests for the option types       #
########################################


@pytest.mark.parametrize("value", [True, 1.5])
def test_objects_are_equal_max_depth_wrong_type(value: object) -> None:
    with pytest.raises(TypeError, match="max_depth must be an int"):
        objects_are_equal([1], [1], max_depth=value)  # ty: ignore[invalid-argument-type]


def test_objects_are_allclose_atol_bool() -> None:
    with pytest.raises(TypeError, match="atol must be a real number"):
        objects_are_allclose(1.0, 1.0, atol=True)


########################################
#     Tests for durations              #
########################################


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (timedelta(seconds=1), timedelta(seconds=1.5)),
        (np.array([1000], dtype="m8[ms]"), np.array([1500], dtype="m8[ms]")),
    ],
)
@numpy_available
def test_objects_are_allclose_durations_follow_atol(actual: object, expected: object) -> None:
    assert objects_are_allclose(actual, expected, atol=1.0)


@numpy_available
def test_objects_are_allclose_instants_are_exact() -> None:
    assert not objects_are_allclose(
        np.array(["2020-01-01"], dtype="M8[D]"), np.array(["2020-01-02"], dtype="M8[D]"), atol=1e9
    )


@numpy_available
def test_objects_are_allclose_duration_array_in_a_dict() -> None:
    assert objects_are_allclose(
        {"x": np.array([1000], dtype="m8[ms]")}, {"x": np.array([1500], dtype="m8[ms]")}, atol=1.0
    )


@pytest.mark.parametrize("kwargs", [{"atol": 1.0}, {"rtol": 1.0}])
def test_objects_are_allclose_bool_has_no_tolerance(kwargs: dict) -> None:
    assert not objects_are_allclose(True, False, **kwargs)


def test_objects_are_allclose_bool_in_a_list_has_no_tolerance() -> None:
    assert not objects_are_allclose([True], [False], atol=1.0)


@numpy_available
def test_objects_are_allclose_numpy_bool_array_has_no_tolerance() -> None:
    assert not objects_are_allclose(np.array([True]), np.array([False]), atol=1.0)


@torch_available
def test_objects_are_allclose_torch_bool_tensor_has_no_tolerance() -> None:
    assert not objects_are_allclose(torch.tensor([True]), torch.tensor([False]), atol=1.0)


def test_objects_are_allclose_mappingproxy_has_tolerance() -> None:
    assert objects_are_allclose(
        MappingProxyType({"a": 1.0}), MappingProxyType({"a": 1.05}), atol=0.1
    )


@numpy_available
def test_objects_are_equal_mappingproxy_with_arrays() -> None:
    assert objects_are_equal(
        MappingProxyType({"a": np.array([1, 2])}), MappingProxyType({"a": np.array([1, 2])})
    )


@numpy_available
def test_objects_are_equal_mappingproxy_with_different_arrays() -> None:
    assert not objects_are_equal(
        MappingProxyType({"a": np.array([1, 2])}), MappingProxyType({"a": np.array([1, 3])})
    )


def test_objects_are_equal_mappingproxy_and_dict_differ() -> None:
    assert not objects_are_equal(MappingProxyType({"a": 1}), {"a": 1})


def test_objects_are_allclose_array_has_tolerance() -> None:
    assert objects_are_allclose(array("d", [1.0, 2.0]), array("d", [1.0, 2.05]), atol=0.1)


def test_objects_are_allclose_array_outside_tolerance() -> None:
    assert not objects_are_allclose(array("d", [1.0, 2.0]), array("d", [1.0, 2.5]), atol=0.1)


def test_objects_are_equal_array_and_list_differ() -> None:
    assert not objects_are_equal(array("d", [1.0]), [1.0])


BIG = 2**62


def test_objects_are_allclose_large_ints_in_a_list_differ() -> None:
    assert not objects_are_allclose([BIG], [BIG + 1], atol=0.5, rtol=0.0)


@numpy_available
@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_objects_are_allclose_large_ints_in_a_numpy_array_differ(dtype: str) -> None:
    assert not objects_are_allclose(
        np.array([BIG], dtype=dtype), np.array([BIG + 1], dtype=dtype), atol=0.5, rtol=0.0
    )


@torch_available
def test_objects_are_allclose_large_ints_in_a_torch_tensor_differ() -> None:
    assert not objects_are_allclose(
        torch.tensor([BIG]), torch.tensor([BIG + 1]), atol=0.5, rtol=0.0
    )


@pandas_available
@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_objects_are_allclose_large_ints_in_a_pandas_series_differ(dtype: str) -> None:
    assert not objects_are_allclose(
        pd.Series([BIG], dtype=dtype), pd.Series([BIG + 1], dtype=dtype), atol=0.5, rtol=0.0
    )


@pandas_available
def test_objects_are_allclose_large_ints_with_a_null_in_a_pandas_series_differ() -> None:
    assert not objects_are_allclose(
        pd.Series([BIG, None], dtype="Int64"),
        pd.Series([BIG + 1, None], dtype="Int64"),
        atol=0.5,
        rtol=0.0,
    )


@pandas_available
def test_objects_are_allclose_large_ints_in_a_pandas_frame_differ() -> None:
    assert not objects_are_allclose(
        pd.DataFrame({"a": [BIG]}), pd.DataFrame({"a": [BIG + 1]}), atol=0.5, rtol=0.0
    )


@polars_available
def test_objects_are_allclose_large_ints_in_a_polars_series_differ() -> None:
    assert not objects_are_allclose(pl.Series([BIG]), pl.Series([BIG + 1]), atol=0.5, rtol=0.0)


@numpy_available
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_objects_are_allclose_small_floats_at_the_edge_of_the_range(dtype: type) -> None:
    assert not objects_are_allclose(
        np.array([60000], dtype=dtype), np.array([-60000], dtype=dtype), rtol=1.5, atol=0.0
    )


@torch_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_objects_are_allclose_small_torch_floats_at_the_edge_of_the_range(dtype: Any) -> None:
    assert not objects_are_allclose(
        torch.tensor([60000.0], dtype=dtype),
        torch.tensor([-60000.0], dtype=dtype),
        rtol=1.5,
        atol=0.0,
    )


@polars_available
def test_objects_are_allclose_polars_durations_have_tolerance() -> None:
    assert objects_are_allclose(
        pl.Series([timedelta(seconds=1)]), pl.Series([timedelta(seconds=1.5)]), atol=1.0
    )


@pyarrow_available
def test_objects_are_allclose_pyarrow_durations_have_tolerance() -> None:
    assert objects_are_allclose(
        pa.array([1_000_000], type=pa.duration("us")),
        pa.array([1_500_000], type=pa.duration("us")),
        atol=1.0,
    )


@polars_available
def test_objects_are_allclose_polars_nested_floats_have_no_tolerance() -> None:
    assert not objects_are_allclose(pl.Series([[1.0]]), pl.Series([[1.0000001]]), atol=1.0)


@pyarrow_available
def test_objects_are_allclose_pyarrow_nested_floats_have_no_tolerance() -> None:
    assert not objects_are_allclose(pa.array([[1.0]]), pa.array([[1.0000001]]), atol=1.0)


@pandas_available
def test_objects_are_allclose_pandas_durations_have_tolerance() -> None:
    assert objects_are_allclose(
        pd.Series(pd.to_timedelta([1], unit="s").as_unit("ns")),
        pd.Series(pd.to_timedelta([1.5], unit="s").as_unit("ns")),
        atol=1.0,
    )


@pandas_available
def test_objects_are_allclose_pandas_nested_floats_have_no_tolerance() -> None:
    assert not objects_are_allclose(pd.Series([[1.0]]), pd.Series([[1.0000001]]), atol=1.0)
