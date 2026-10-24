r"""Property-based tests for the equality of numbers, NumPy scalars,
complex numbers and dataclasses."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given, strategies as st

from coola.equality import objects_are_allclose, objects_are_equal

finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
# Within the range of ``float16``, so that casting to it does not overflow.
np_finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-6e4, max_value=6e4)
tolerances = st.floats(min_value=0, max_value=10)
float_types = st.sampled_from([np.float16, np.float32, np.float64])
int_types = st.sampled_from([np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16])
complex_numbers = st.complex_numbers(allow_nan=False, allow_infinity=False, max_magnitude=1e6)


def _is_close(a: Any, b: Any, atol: float, rtol: float) -> bool:
    return a == b or abs(a - b) <= atol + rtol * max(abs(a), abs(b))


###############################
#     Result is always a bool #
###############################


@dataclass
class Plain:
    x: Any
    y: Any = None


@dataclass
class WithArray:
    data: np.ndarray


@given(x=finite, y=finite)
def test_result_is_true_for_equal_dataclasses_with_arrays(x: float, y: float) -> None:
    a, b = WithArray(np.array([x, y])), WithArray(np.array([x, y]))
    assert objects_are_equal(a, b) is True


@given(x=finite, y=finite)
def test_result_is_true_for_close_dataclasses_with_arrays(x: float, y: float) -> None:
    a, b = WithArray(np.array([x, y])), WithArray(np.array([x, y]))
    assert objects_are_allclose(a, b) is True


@given(x=finite, y=finite)
def test_result_is_false_for_different_dataclasses_with_arrays(x: float, y: float) -> None:
    a = WithArray(np.array([x, y]))
    assert objects_are_equal(a, WithArray(np.array([x, y + 1.0]))) is False


@pytest.mark.parametrize("function", [objects_are_equal, objects_are_allclose])
@given(
    x=np_finite, y=np_finite, dtype=st.sampled_from([np.float32, np.float64, np.int32, np.bool_])
)
def test_result_is_a_bool_for_numpy_scalars(function: Any, x: float, y: float, dtype: type) -> None:
    assert type(function(dtype(x), dtype(y))) is bool


@pytest.mark.parametrize("function", [objects_are_equal, objects_are_allclose])
@given(
    value=st.one_of(finite, st.integers(), complex_numbers, st.text()),
    other=st.one_of(finite, st.integers(), complex_numbers, st.text()),
)
def test_result_is_a_bool_for_scalars(function: Any, value: Any, other: Any) -> None:
    assert type(function(value, other)) is bool


########################
#     NumPy scalars    #
########################


@given(dtype=float_types, x=np_finite, y=np_finite, atol=tolerances, rtol=st.floats(0, 1))
def test_numpy_float_follows_the_numpy_rule(
    dtype: type, x: float, y: float, atol: float, rtol: float
) -> None:
    a, b = dtype(x), dtype(y)
    assume(np.isfinite(a) and np.isfinite(b))
    expected = _is_close(float(a), float(b), atol, rtol)
    assert objects_are_allclose(a, b, atol=atol, rtol=rtol) == expected


@given(dtype=float_types, x=np_finite)
def test_numpy_float_is_reflexive(dtype: type, x: float) -> None:
    assert objects_are_equal(dtype(x), dtype(x))


@given(dtype=float_types, x=np_finite)
def test_numpy_float_is_reflexive_for_allclose(dtype: type, x: float) -> None:
    assert objects_are_allclose(dtype(x), dtype(x))


@given(dtype=int_types, x=st.integers(0, 100), y=st.integers(0, 100), atol=st.integers(0, 50))
def test_numpy_int_follows_absolute_tolerance(dtype: type, x: int, y: int, atol: int) -> None:
    assert objects_are_allclose(dtype(x), dtype(y), atol=float(atol), rtol=0.0) == (
        abs(x - y) <= atol
    )


@given(dtype=int_types, x=st.integers(0, 100), y=st.integers(0, 100))
def test_numpy_int_equality_follows_eq(dtype: type, x: int, y: int) -> None:
    assert objects_are_equal(dtype(x), dtype(y)) == (x == y)


@given(dtype=float_types, x=np_finite)
def test_numpy_scalar_of_another_type_is_different(dtype: type, x: float) -> None:
    other = np.float64 if dtype is not np.float64 else np.float32
    assert not objects_are_equal(dtype(x), other(x))


@given(dtype=float_types, x=np_finite)
def test_numpy_scalar_of_another_type_is_not_close(dtype: type, x: float) -> None:
    other = np.float64 if dtype is not np.float64 else np.float32
    assert not objects_are_allclose(dtype(x), other(x), atol=1.0)


@given(dtype=float_types)
def test_numpy_float_nan_requires_equal_nan(dtype: type) -> None:
    assert not objects_are_equal(dtype("nan"), dtype("nan"))


@given(dtype=float_types)
def test_numpy_float_nan_is_equal_with_equal_nan(dtype: type) -> None:
    assert objects_are_equal(dtype("nan"), dtype("nan"), equal_nan=True)


@given(dtype=float_types)
def test_numpy_float_nan_is_close_with_equal_nan(dtype: type) -> None:
    assert objects_are_allclose(dtype("nan"), dtype("nan"), equal_nan=True)


@given(dtype=float_types, x=np_finite)
def test_numpy_float_nan_is_not_equal_to_a_number(dtype: type, x: float) -> None:
    assert not objects_are_equal(dtype("nan"), dtype(x), equal_nan=True)


@given(dtype=float_types, x=np_finite)
def test_numpy_float_number_is_not_equal_to_nan(dtype: type, x: float) -> None:
    assert not objects_are_equal(dtype(x), dtype("nan"), equal_nan=True)


@given(x=np_finite, y=np_finite, atol=tolerances)
def test_numpy_float_inside_containers(x: float, y: float, atol: float) -> None:
    actual = {"a": [np.float32(x)], "b": (np.float64(y),)}
    expected = {"a": [np.float32(x)], "b": (np.float64(y),)}
    assert objects_are_allclose(actual, expected, atol=atol)


########################
#   Complex numbers    #
########################


@given(a=complex_numbers, b=complex_numbers, atol=tolerances, rtol=st.floats(0, 1))
def test_complex_follows_the_numpy_rule(a: complex, b: complex, atol: float, rtol: float) -> None:
    assert objects_are_allclose(a, b, atol=atol, rtol=rtol) == _is_close(a, b, atol, rtol)


@given(a=complex_numbers, b=complex_numbers)
def test_complex_exact_follows_eq(a: complex, b: complex) -> None:
    assert objects_are_equal(a, b) == (a == b)


@given(a=complex_numbers, b=complex_numbers)
def test_complex_exact_is_symmetric(a: complex, b: complex) -> None:
    assert objects_are_equal(a, b) == objects_are_equal(b, a)


@given(a=complex_numbers)
def test_complex_is_reflexive(a: complex) -> None:
    assert objects_are_equal(a, complex(a))


@given(a=complex_numbers)
def test_complex_is_reflexive_for_allclose(a: complex) -> None:
    assert objects_are_allclose(a, complex(a))


@given(a=complex_numbers, delta=st.floats(1.0, 100.0))
def test_complex_imaginary_part_is_not_ignored(a: complex, delta: float) -> None:
    other = complex(a.real, a.imag + delta)
    assert not objects_are_allclose(a, other, atol=delta / 2, rtol=0.0)


def make_complex_with_nan(nan_part: str, real: float, imag: float) -> complex:
    r"""Create a complex number whose real or imaginary part is NaN."""
    if nan_part == "real":
        return complex(float("nan"), imag)
    return complex(real, float("nan"))


@pytest.mark.parametrize("nan_part", ["real", "imag"])
@given(real=finite, imag=finite)
def test_complex_nan_requires_equal_nan(nan_part: str, real: float, imag: float) -> None:
    value = make_complex_with_nan(nan_part, real, imag)
    assert not objects_are_equal(value, complex(value.real, value.imag))


@pytest.mark.parametrize("nan_part", ["real", "imag"])
@given(real=finite, imag=finite)
def test_complex_nan_is_equal_with_equal_nan(nan_part: str, real: float, imag: float) -> None:
    value = make_complex_with_nan(nan_part, real, imag)
    assert objects_are_equal(value, complex(value.real, value.imag), equal_nan=True)


@pytest.mark.parametrize("nan_part", ["real", "imag"])
@given(real=finite, imag=finite)
def test_complex_nan_is_not_equal_to_a_number(nan_part: str, real: float, imag: float) -> None:
    value = make_complex_with_nan(nan_part, real, imag)
    assert not objects_are_equal(value, complex(real, imag), equal_nan=True)


@given(dtype=st.sampled_from([np.complex64, np.complex128]), a=complex_numbers, atol=tolerances)
def test_numpy_complex_follows_the_numpy_rule(dtype: type, a: complex, atol: float) -> None:
    x = dtype(a)
    y = dtype(a + 1.0)
    assume(np.isfinite(x) and np.isfinite(y))
    assert objects_are_allclose(x, y, atol=atol, rtol=0.0) == _is_close(
        complex(x), complex(y), atol, 0.0
    )


###################
#    Dataclasses  #
###################


@dataclass
class Other:
    x: Any
    y: Any = None


@dataclass
class Ignored:
    x: float
    note: str = field(default="", compare=False)


@dataclass
class Nested:
    inner: Plain
    items: list = field(default_factory=list)


dataclass_values = st.one_of(st.integers(-5, 5), finite, st.text(max_size=3), st.none())


@given(x=dataclass_values, y=dataclass_values)
def test_dataclass_is_reflexive(x: Any, y: Any) -> None:
    value = Plain(x, y)
    assert objects_are_equal(value, copy.deepcopy(value))


@given(x=dataclass_values, y=dataclass_values)
def test_dataclass_is_reflexive_for_allclose(x: Any, y: Any) -> None:
    value = Plain(x, y)
    assert objects_are_allclose(value, copy.deepcopy(value))


@given(x=dataclass_values, y=dataclass_values, u=dataclass_values, v=dataclass_values)
def test_dataclass_equality_matches_the_equality_of_the_fields(
    x: Any, y: Any, u: Any, v: Any
) -> None:
    expected = objects_are_equal([x, y], [u, v])
    assert objects_are_equal(Plain(x, y), Plain(u, v)) == expected


@given(x=dataclass_values, y=dataclass_values, u=dataclass_values, v=dataclass_values)
def test_dataclass_equality_is_symmetric(x: Any, y: Any, u: Any, v: Any) -> None:
    expected = objects_are_equal([x, y], [u, v])
    assert objects_are_equal(Plain(u, v), Plain(x, y)) == expected


@given(x=dataclass_values, y=dataclass_values)
def test_dataclass_of_another_type_is_different(x: Any, y: Any) -> None:
    assert not objects_are_equal(Plain(x, y), Other(x, y))


@given(x=dataclass_values, y=dataclass_values)
def test_dataclass_of_another_type_is_not_close(x: Any, y: Any) -> None:
    assert not objects_are_allclose(Plain(x, y), Other(x, y), atol=1.0)


@given(x=finite, atol=st.floats(0.001, 10), ratio=st.floats(0.0, 0.9))
def test_dataclass_tolerance_is_applied_to_the_fields(x: float, atol: float, ratio: float) -> None:
    close = Plain(x, [x])
    other = Plain(x + atol * ratio, [x + atol * ratio])
    assert objects_are_allclose(close, other, atol=atol, rtol=0.0)


@given(x=finite, atol=st.floats(0.001, 10))
def test_dataclass_tolerance_does_not_hide_a_difference(x: float, atol: float) -> None:
    assert not objects_are_allclose(
        Plain(x, [x]), Plain(x + 2 * atol + 1, [x]), atol=atol, rtol=0.0
    )


@given(x=finite, note=st.text(max_size=3), other_note=st.text(max_size=3))
def test_dataclass_ignores_the_fields_with_compare_false(
    x: float, note: str, other_note: str
) -> None:
    assert objects_are_equal(Ignored(x, note), Ignored(x, other_note))


@given(x=finite, note=st.text(max_size=3))
def test_dataclass_does_not_ignore_the_fields_with_compare_true(x: float, note: str) -> None:
    assert not objects_are_equal(Ignored(x, note), Ignored(x + 1.0, note))


@given(equal_nan=st.booleans())
def test_dataclass_nan_requires_equal_nan(equal_nan: bool) -> None:
    result = objects_are_equal(Plain(float("nan")), Plain(float("nan")), equal_nan=equal_nan)
    assert result == equal_nan


def make_nested(x: float, n: int) -> dict[str, Any]:
    r"""Create a dictionary of containers of nested dataclasses."""
    return {"a": [Nested(Plain(x), [Plain(x)] * n)], "b": (Plain(np.float32(x)),)}


@given(x=finite, n=st.integers(0, 4))
def test_dataclass_nested_in_containers(x: float, n: int) -> None:
    assert objects_are_equal(make_nested(x, n), make_nested(x, n))


@given(x=finite, n=st.integers(0, 4))
def test_dataclass_nested_in_containers_different(x: float, n: int) -> None:
    assert not objects_are_equal(
        make_nested(x, n), {**make_nested(x, n), "b": (Plain(np.float32(x) + 1),)}
    )


@given(data=st.lists(finite, min_size=1, max_size=5))
def test_dataclass_with_arrays_matches_array_equality(data: list[float]) -> None:
    array = np.array(data)
    assert objects_are_equal(WithArray(array), WithArray(array.copy()))


@given(data=st.lists(finite, min_size=1, max_size=5))
def test_dataclass_with_arrays_of_different_values_differ(data: list[float]) -> None:
    array = np.array(data)
    assert not objects_are_equal(WithArray(array), WithArray(array + 1.0))


@given(data=st.lists(finite, min_size=1, max_size=5))
def test_dataclass_with_arrays_of_different_shapes_differ(data: list[float]) -> None:
    array = np.array(data)
    assert not objects_are_equal(WithArray(array), WithArray(np.append(array, 0.0)))


@given(x=finite)
def test_dataclass_with_custom_eq_keeps_its_eq(x: float) -> None:
    @dataclass(eq=False)
    class AlwaysEqual:
        x: float

        def __eq__(self, other: object) -> bool:
            return True

        __hash__ = None  # type: ignore[assignment]

    assert objects_are_equal(AlwaysEqual(x), AlwaysEqual(x + 1.0))


@pytest.mark.parametrize("function", [objects_are_equal, objects_are_allclose])
def test_dataclass_class_is_equal_to_itself(function: Any) -> None:
    assert function(Plain, Plain)


@pytest.mark.parametrize("function", [objects_are_equal, objects_are_allclose])
def test_dataclass_class_is_different_from_another_class(function: Any) -> None:
    assert not function(Plain, Other)


@pytest.mark.parametrize("function", [objects_are_equal, objects_are_allclose])
@given(x=dataclass_values)
def test_dataclass_class_is_different_from_an_instance(function: Any, x: Any) -> None:
    assert not function(Plain, Plain(x))
