from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.reducer import EmptySequenceError, NativeReducer, TorchReducer
from coola.reducer.torch import _to_float_tensor
from coola.testing.marks import numpy_available, torch_available
from coola.utils.imports import MissingPackageError, is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()  # pragma: no cover

if is_torch_available():
    import torch
else:
    torch = Mock()  # pragma: no cover


if TYPE_CHECKING:
    from collections.abc import Sequence


EMPTY_SEQUENCES = [[], (), torch.tensor([])]


##################################
#     Tests for TorchReducer     #
##################################


@torch_available
def test_torch_reducer_str() -> None:
    assert str(TorchReducer()).startswith("TorchReducer(")


@torch_available
@pytest.mark.parametrize(
    "values",
    [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [2], [2, -1, -2, -3], torch.tensor([-2, -1, 0, 1, 2])],
)
def test_torch_reducer_max_int_type(values: Sequence[int | float]) -> None:
    val = TorchReducer().max(values)
    assert isinstance(val, int)


@torch_available
@pytest.mark.parametrize(
    "values",
    [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [2], [2, -1, -2, -3], torch.tensor([-2, -1, 0, 1, 2])],
)
def test_torch_reducer_max_int_value(values: Sequence[int | float]) -> None:
    val = TorchReducer().max(values)
    assert val == 2


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-2.5, -1.5, 0.5, 1.5, 2.5],
        (-2.5, -1.5, 0.5, 1.5, 2.5),
        [2.5],
        [2.5, -1.5, -2, -3],
        torch.tensor([-2.5, -1.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_max_float_type(values: Sequence[int | float]) -> None:
    val = TorchReducer().max(values)
    assert isinstance(val, float)


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-2.5, -1.5, 0.5, 1.5, 2.5],
        (-2.5, -1.5, 0.5, 1.5, 2.5),
        [2.5],
        [2.5, -1.5, -2, -3],
        torch.tensor([-2.5, -1.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_max_float_value(values: Sequence[int | float]) -> None:
    val = TorchReducer().max(values)
    assert val == 2.5


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_max_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the maximum because the sequence is empty"
    ):
        TorchReducer().max(values)


@torch_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0], torch.tensor([-2, -1, 0, 1, 2])]
)
def test_torch_reducer_mean_int_type(values: Sequence[int | float]) -> None:
    val = TorchReducer().mean(values)
    assert isinstance(val, float)


@torch_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0], torch.tensor([-2, -1, 0, 1, 2])]
)
def test_torch_reducer_mean_int_value(values: Sequence[int | float]) -> None:
    val = TorchReducer().mean(values)
    assert val == 0.0


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        [0.5],
        torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_mean_float_type(values: Sequence[int | float]) -> None:
    val = TorchReducer().mean(values)
    assert isinstance(val, float)


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        [0.5],
        torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_mean_float_value(values: Sequence[int | float]) -> None:
    val = TorchReducer().mean(values)
    assert val == 0.5


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_mean_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the mean because the sequence is empty"
    ):
        TorchReducer().mean(values)


@torch_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0], torch.tensor([-2, -1, 0, 1, 2])]
)
def test_torch_reducer_median_int_type(values: Sequence[int | float]) -> None:
    val = TorchReducer().median(values)
    assert isinstance(val, int)


@torch_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0], torch.tensor([-2, -1, 0, 1, 2])]
)
def test_torch_reducer_median_int_value(values: Sequence[int | float]) -> None:
    val = TorchReducer().median(values)
    assert val == 0


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        [0.5],
        torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_median_float_type(values: Sequence[int | float]) -> None:
    val = TorchReducer().median(values)
    assert isinstance(val, float)


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        [0.5],
        torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_median_float_value(values: Sequence[int | float]) -> None:
    val = TorchReducer().median(values)
    assert val == 0.5


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_median_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the median because the sequence is empty"
    ):
        TorchReducer().median(values)


@torch_available
@pytest.mark.parametrize(
    "values",
    [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [-2], [-2, 1, 2, 3], torch.tensor([-2, -1, 0, 1, 2])],
)
def test_torch_reducer_min_int_type(values: Sequence[int | float]) -> None:
    val = TorchReducer().min(values)
    assert isinstance(val, int)


@torch_available
@pytest.mark.parametrize(
    "values",
    [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [-2], [-2, 1, 2, 3], torch.tensor([-2, -1, 0, 1, 2])],
)
def test_torch_reducer_min_int_value(values: Sequence[int | float]) -> None:
    val = TorchReducer().min(values)
    assert val == -2


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-2.5, -1.5, 0.5, 1.5, 2.5],
        (-2.5, -1.5, 0.5, 1.5, 2.5),
        [-2.5],
        [-2.5, 1.5, 2, 3],
        torch.tensor([-2.5, -1.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_min_float_type(values: Sequence[int | float]) -> None:
    val = TorchReducer().min(values)
    assert isinstance(val, float)


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-2.5, -1.5, 0.5, 1.5, 2.5],
        (-2.5, -1.5, 0.5, 1.5, 2.5),
        [-2.5],
        [-2.5, 1.5, 2, 3],
        torch.tensor([-2.5, -1.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_min_float_value(values: Sequence[int | float]) -> None:
    val = TorchReducer().min(values)
    assert val == -2.5


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_min_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the minimum because the sequence is empty"
    ):
        TorchReducer().min(values)


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ],
)
def test_torch_reducer_quantile_int(values: Sequence[int | float]) -> None:
    assert TorchReducer().quantile(values, (0.2, 0.5, 0.9)) == [2, 5, 9]


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5),
        torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]),
    ],
)
def test_torch_reducer_quantile_float(values: Sequence[int | float]) -> None:
    assert TorchReducer().quantile(values, (0.0, 0.1, 0.4, 0.9)) == [0.5, 1.5, 4.5, 9.5]


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_quantile_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the quantiles because the sequence is empty"
    ):
        TorchReducer().quantile(values, [0.5])


@torch_available
@pytest.mark.parametrize(
    "values", [[2, 1, -2, 3, 0], (2, 1, -2, 3, 0), torch.tensor([2, 1, -2, 3, 0])]
)
def test_torch_reducer_sort_int(values: Sequence[int | float]) -> None:
    assert TorchReducer().sort(values) == [-2, 0, 1, 2, 3]


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [2.5, 1.5, -2.5, 3.5, 0.5],
        (2.5, 1.5, -2.5, 3.5, 0.5),
        torch.tensor([2.5, 1.5, -2.5, 3.5, 0.5]),
    ],
)
def test_torch_reducer_sort_float(values: Sequence[int | float]) -> None:
    assert TorchReducer().sort(values) == [-2.5, 0.5, 1.5, 2.5, 3.5]


@torch_available
@pytest.mark.parametrize(
    "values", [[2, 1, -2, 3, 0], (2, 1, -2, 3, 0), torch.tensor([2, 1, -2, 3, 0])]
)
def test_torch_reducer_sort_descending(values: Sequence[int | float]) -> None:
    assert TorchReducer().sort(values, descending=True) == [3, 2, 1, 0, -2]


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_sort_empty(values: Sequence[int | float]) -> None:
    assert TorchReducer().sort(values) == []


@torch_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), torch.tensor([-2, -1, 0, 1, 2])]
)
def test_torch_reducer_std_int(values: Sequence[int | float]) -> None:
    assert math.isclose(TorchReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_std_float(values: Sequence[int | float]) -> None:
    assert math.isclose(TorchReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@torch_available
@pytest.mark.parametrize("values", [[1], [1.0], torch.tensor([1]), torch.tensor([1.0])])
def test_torch_reducer_std_one(values: Sequence[int | float]) -> None:
    assert math.isnan(TorchReducer().std(values))


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_std_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError,
        match=r"Cannot compute the standard deviation because the sequence is empty",
    ):
        TorchReducer().std(values)


@torch_available
def test_torch_reducer_no_torch() -> None:
    with (
        patch("coola.utils.imports.optional.is_torch_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'torch' package is required but not installed."),
    ):
        TorchReducer()


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        pytest.param([1, 2, 3, 4], id="list-even"),
        pytest.param([4, 1, 3, 2], id="list-unsorted"),
        pytest.param(torch.tensor([1.0, 2.0, 3.0, 4.0]), id="tensor-even"),
    ],
)
def test_torch_reducer_median_even_averages_middle_values(values: Sequence[float]) -> None:
    assert TorchReducer().median(values) == 2.5


@torch_available
def test_torch_reducer_median_odd() -> None:
    assert TorchReducer().median([5, 1, 3]) == 3


@torch_available
def test_torch_reducer_matches_native_reducer_median() -> None:
    values = [0.1, 0.7, 0.3, 2.5, 1.1, 9.9]
    torch_reducer, native = TorchReducer(), NativeReducer()
    assert torch_reducer.median(values) == pytest.approx(native.median(values))


@torch_available
def test_torch_reducer_matches_native_reducer_mean() -> None:
    values = [0.1, 0.7, 0.3, 2.5, 1.1, 9.9]
    torch_reducer, native = TorchReducer(), NativeReducer()
    assert torch_reducer.mean(values) == pytest.approx(native.mean(values), rel=1e-12)


@torch_available
def test_torch_reducer_matches_native_reducer_std() -> None:
    values = [0.1, 0.7, 0.3, 2.5, 1.1, 9.9]
    torch_reducer, native = TorchReducer(), NativeReducer()
    assert torch_reducer.std(values) == pytest.approx(native.std(values), rel=1e-12)


@torch_available
def test_torch_reducer_matches_native_reducer_quantile() -> None:
    values = [0.1, 0.7, 0.3, 2.5, 1.1, 9.9]
    torch_reducer, native = TorchReducer(), NativeReducer()
    assert torch_reducer.quantile(values, [0.1, 0.5]) == pytest.approx(
        native.quantile(values, [0.1, 0.5]), rel=1e-12
    )


@torch_available
@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        pytest.param(torch.tensor([0.1, 0.2], dtype=torch.float32), "float32", id="float32"),
        pytest.param(torch.tensor([0.1, 0.2], dtype=torch.float64), "float64", id="float64"),
        pytest.param(torch.tensor([0.1, 0.2], dtype=torch.float16), "float16", id="float16"),
        pytest.param(torch.tensor([1, 2]), "float64", id="int64"),
        pytest.param(torch.tensor([True, False]), "float64", id="bool"),
    ],
)
def test_to_float_tensor_dtype(values: object, dtype: str) -> None:
    assert _to_float_tensor(values).dtype == getattr(torch, dtype)


@torch_available
def test_torch_reducer_float32_input_keeps_float32_precision_mean() -> None:
    values = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    assert TorchReducer().mean(values) == values.mean().item()


@torch_available
def test_torch_reducer_float32_input_keeps_float32_precision_std() -> None:
    values = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    assert TorchReducer().std(values) == values.std().item()


@torch_available
def test_torch_reducer_float32_input_keeps_float32_precision_median() -> None:
    values = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    assert TorchReducer().median(values) == pytest.approx(0.25)


@torch_available
def test_torch_reducer_float16_quantile() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    assert TorchReducer().quantile(values, [0.5]) == [2.0]


@torch_available
def test_torch_reducer_python_floats_keep_double_precision() -> None:
    assert TorchReducer().mean([0.1, 0.2]) == pytest.approx(0.15, rel=1e-15)


@torch_available
def test_torch_reducer_median_nan_tensor() -> None:
    assert math.isnan(TorchReducer().median(torch.tensor([1.0, float("nan"), 2.0, 3.0])))


@torch_available
def test_torch_reducer_median_int_tensor_is_not_nan() -> None:
    assert TorchReducer().median(torch.tensor([1, 2, 3, 4])) == 2.5


@torch_available
def test_torch_reducer_median_does_not_overflow() -> None:
    assert TorchReducer().median([1e308, 1e308]) == 1e308


@torch_available
def test_torch_reducer_median_infinite_values() -> None:
    assert TorchReducer().median([float("inf"), float("inf")]) == float("inf")


@torch_available
def test_torch_reducer_max_generator_raises() -> None:
    with pytest.raises(TypeError, match="iterator"):
        TorchReducer().max(i for i in range(3))


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        pytest.param([2**70, 1], id="big-int"),
        pytest.param([None, 1], id="none"),
        pytest.param(["a", "b"], id="str"),
        pytest.param([Decimal(1)], id="decimal"),
        pytest.param([[1, 2], [3]], id="ragged"),
    ],
)
def test_torch_reducer_unsupported_values_raise(values: list) -> None:
    with pytest.raises(TypeError, match=r"TorchReducer does not support .*use NativeReducer"):
        TorchReducer().max(values)


@torch_available
@pytest.mark.parametrize("values", [[1 + 2j, 3], torch.tensor([1 + 2j, 3])])
def test_torch_reducer_complex_raises(values: object) -> None:
    with pytest.raises(TypeError, match=r"TorchReducer does not support complex numbers"):
        TorchReducer().max(values)


@torch_available
@numpy_available
def test_torch_reducer_complex_array_raises() -> None:
    with pytest.raises(TypeError, match=r"TorchReducer does not support an array of dtype complex"):
        TorchReducer().max(np.array([1 + 2j, 3]))


@torch_available
@numpy_available
def test_torch_reducer_array_max() -> None:
    assert TorchReducer().max(np.array([1, 3, 2])) == 3


@torch_available
@pytest.mark.parametrize("values", [[[1, 2], [3, 4]], torch.tensor(1), torch.ones(2, 0)])
def test_torch_reducer_not_1d_raises(values: object) -> None:
    with pytest.raises(ValueError, match=r"TorchReducer takes a 1-D input"):
        TorchReducer().max(values)


@torch_available
def test_torch_reducer_sort_2d_raises() -> None:
    with pytest.raises(ValueError, match=r"TorchReducer takes a 1-D input, got 2 dimension"):
        TorchReducer().sort([[2, 1], [4, 3]])


@torch_available
def test_torch_reducer_tensor_does_not_copy() -> None:
    tensor = torch.tensor([3, 1, 2])
    assert TorchReducer()._prepare(tensor) is tensor


@torch_available
@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        pytest.param([0.1, 0.2], "float64", id="list-of-floats"),
        pytest.param([1, 2], "int64", id="list-of-ints"),
        pytest.param((0.1, 0.2), "float64", id="tuple"),
        pytest.param([1, 2.5], "float64", id="mixed"),
    ],
)
def test_torch_reducer_prepare_dtype(values: object, dtype: str) -> None:
    assert TorchReducer()._prepare(values).dtype == getattr(torch, dtype)


@torch_available
def test_torch_reducer_result_is_python_scalar() -> None:
    assert type(TorchReducer().max(torch.tensor([1, 2]))) is int
