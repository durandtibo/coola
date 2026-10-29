from __future__ import annotations

import math

import pytest

from coola.rounding.torch import round_torch
from coola.testing.marks import torch_available
from coola.utils.imports import is_torch_available

if is_torch_available():
    import torch

####################################
#     Tests for round_torch        #
####################################


@torch_available
def test_round_torch_round_half_to_even() -> None:
    result = round_torch(torch.tensor([0.5, 1.5, 2.5, -1.5], dtype=torch.float64), "round", 0)
    assert result.tolist() == [0.0, 2.0, 2.0, -2.0]


@torch_available
def test_round_torch_floor() -> None:
    result = round_torch(torch.tensor([2.567, -2.567, 2.0], dtype=torch.float64), "floor", 0)
    assert result.tolist() == [2.0, -3.0, 2.0]


@torch_available
def test_round_torch_ceil() -> None:
    result = round_torch(torch.tensor([2.567, -2.567, 2.0], dtype=torch.float64), "ceil", 0)
    assert result.tolist() == [3.0, -2.0, 2.0]


@torch_available
def test_round_torch_floor_decimals() -> None:
    result = round_torch(torch.tensor([2.567, 0.29, -2.567], dtype=torch.float64), "floor", 2)
    assert result.tolist() == [2.56, 0.29, -2.57]


@torch_available
def test_round_torch_ceil_decimals() -> None:
    result = round_torch(torch.tensor([2.561, 0.07, -2.561], dtype=torch.float64), "ceil", 2)
    assert result.tolist() == [2.57, 0.07, -2.56]


@torch_available
def test_round_torch_round_decimals() -> None:
    result = round_torch(torch.tensor([1.234, 5.678], dtype=torch.float64), "round", 1)
    assert result.tolist() == [1.2, 5.7]


@torch_available
def test_round_torch_special_values_kept() -> None:
    result = round_torch(
        torch.tensor([math.inf, -math.inf, math.nan, 1e308], dtype=torch.float64), "floor", 2
    )
    assert result.tolist()[:2] == [math.inf, -math.inf]
    assert math.isnan(result[2].item())
    assert result[3].item() == 1e308


@torch_available
@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int64, torch.uint8, torch.bool])
def test_round_torch_integer_and_bool_unchanged(mode: str, dtype: torch.dtype) -> None:
    data = torch.tensor([0, 1, 1], dtype=dtype)
    assert torch.equal(round_torch(data, mode, 2), data)  # ty: ignore[invalid-argument-type]


@torch_available
def test_round_torch_integer_does_not_share_memory() -> None:
    data = torch.tensor([1, 2])
    assert round_torch(data, "floor", 1).data_ptr() != data.data_ptr()


@torch_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_round_torch_float_dtype_kept(dtype: torch.dtype) -> None:
    assert round_torch(torch.tensor([1.25], dtype=dtype), "floor", 1).dtype == dtype


@torch_available
def test_round_torch_float32_floor_value_on_the_grid() -> None:
    data = torch.tensor([0.29, 0.07])
    assert torch.equal(round_torch(data, "floor", 2), data)


@torch_available
def test_round_torch_float32_ceil_value_on_the_grid() -> None:
    data = torch.tensor([0.29, 0.07])
    assert torch.equal(round_torch(data, "ceil", 2), data)


@torch_available
def test_round_torch_does_not_modify_input() -> None:
    data = torch.tensor([1.26, 2.5], dtype=torch.float64)
    round_torch(data, "floor", 1)
    assert data.tolist() == [1.26, 2.5]


@torch_available
def test_round_torch_requires_grad_is_detached() -> None:
    data = torch.tensor([1.26], requires_grad=True)
    assert not round_torch(data, "floor", 1).requires_grad


@torch_available
def test_round_torch_multidimensional() -> None:
    result = round_torch(torch.full((2, 3, 4), 1.26, dtype=torch.float64), "floor", 1)
    assert result.shape == (2, 3, 4)


@torch_available
def test_round_torch_empty() -> None:
    assert round_torch(torch.tensor([]), "floor", 1).shape == (0,)


@torch_available
def test_round_torch_zero_dim() -> None:
    assert round_torch(torch.tensor(2.567, dtype=torch.float64), "floor", 2).item() == 2.56


@torch_available
def test_round_torch_too_many_decimals_unchanged() -> None:
    data = torch.tensor([0.1], dtype=torch.float64)
    assert round_torch(data, "floor", 400).tolist() == [0.1]


@torch_available
def test_round_torch_complex_unsupported() -> None:
    with pytest.raises(TypeError, match=r"Unsupported dtype"):
        round_torch(torch.tensor([1 + 2j]), "round", 0)


@torch_available
def test_round_torch_unsupported_type() -> None:
    with pytest.raises(TypeError, match=r"Unsupported type"):
        round_torch([1.5], "round", 0)


##########################################
#     Tests for the one decimal rule     #
##########################################


@torch_available
def test_round_torch_ceil_one_ulp_above_grid_point() -> None:
    result = round_torch(torch.tensor([5.5200000000000005], dtype=torch.float64), "ceil", 2)
    assert result.tolist() == [5.53]


@torch_available
def test_round_torch_round_halfway_decimal_goes_to_even() -> None:
    result = round_torch(torch.tensor([2.675, 0.125, 8.995], dtype=torch.float64), "round", 2)
    assert result.tolist() == [2.68, 0.12, 9.0]


@torch_available
def test_round_torch_negative_zero_result_keeps_sign() -> None:
    result = round_torch(torch.tensor([-0.4], dtype=torch.float64), "ceil", 0)
    assert math.copysign(1.0, result.item()) == -1.0
