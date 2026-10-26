from __future__ import annotations

from unittest.mock import Mock

import pytest

from coola.equality._helpers.torch import (
    tensor_equal,
)
from coola.equality.config import EqualityConfig
from coola.equality.tolerance import Tolerance
from coola.testing.fixtures import torch_available
from coola.utils.imports import is_torch_available

if is_torch_available():
    import torch
else:
    torch = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


##################################################
#     Tests for TorchTensorSameDeviceHandler     #
##################################################


@torch_available
def test_tensor_equal_sparse(config: EqualityConfig) -> None:
    eye = torch.eye(2)
    assert tensor_equal(eye.to_sparse(), eye.to_sparse(), config)


@torch_available
def test_tensor_equal_sparse_different(config: EqualityConfig) -> None:
    eye = torch.eye(2)
    assert not tensor_equal(eye.to_sparse(), (2 * eye).to_sparse(), config)


@torch_available
def test_tensor_equal_different_layout(config: EqualityConfig) -> None:
    eye = torch.eye(2)
    assert not tensor_equal(eye.to_sparse(), eye, config)


@torch_available
def test_tensor_equal_meta(config: EqualityConfig) -> None:
    assert tensor_equal(torch.empty(2, device="meta"), torch.empty(2, device="meta"), config)


@torch_available
def test_tensor_equal_meta_different(config: EqualityConfig) -> None:
    assert not tensor_equal(torch.empty(2, device="meta"), torch.empty(3, device="meta"), config)


@torch_available
def test_tensor_equal_rtol_scales_with_the_second_tensor() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert tensor_equal(torch.tensor([100.0]), torch.tensor([101.005]), config)


@torch_available
def test_tensor_equal_rtol_scales_with_the_first_tensor() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert tensor_equal(torch.tensor([101.005]), torch.tensor([100.0]), config)


@torch_available
def test_tensor_equal_rtol_outside_the_tolerance() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert not tensor_equal(torch.tensor([101.5]), torch.tensor([100.0]), config)


@torch_available
def test_tensor_equal_rtol_outside_the_tolerance_reversed() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert not tensor_equal(torch.tensor([100.0]), torch.tensor([101.5]), config)


@torch_available
def test_tensor_equal_rtol_each_element_in_its_own_direction() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert tensor_equal(
        torch.tensor([100.0, 101.005], dtype=torch.float64),
        torch.tensor([101.005, 100.0], dtype=torch.float64),
        config,
    )


@torch_available
def test_tensor_equal_rtol_each_element_in_its_own_direction_reversed() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert tensor_equal(
        torch.tensor([101.005, 100.0], dtype=torch.float64),
        torch.tensor([100.0, 101.005], dtype=torch.float64),
        config,
    )


@torch_available
def test_tensor_equal_rtol_one_element_outside_the_tolerance() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert not tensor_equal(
        torch.tensor([100.0, 101.005], dtype=torch.float64),
        torch.tensor([101.005, 105.0], dtype=torch.float64),
        config,
    )


@torch_available
def test_tensor_equal_different_shape_with_tolerance() -> None:
    assert not tensor_equal(
        torch.ones(2), torch.ones(3), EqualityConfig(tolerance=Tolerance(atol=1.0))
    )


@torch_available
def test_tensor_equal_different_dtype_with_tolerance() -> None:
    assert not tensor_equal(
        torch.ones(2, dtype=torch.float32),
        torch.ones(2, dtype=torch.int32),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    )


@torch_available
def test_tensor_equal_integers_with_tolerance() -> None:
    assert tensor_equal(
        torch.tensor([100, 101]),
        torch.tensor([101, 100]),
        EqualityConfig(tolerance=Tolerance(rtol=0.02)),
    )


@torch_available
def test_tensor_equal_bool_has_no_atol() -> None:
    assert not tensor_equal(
        torch.tensor([True]), torch.tensor([False]), EqualityConfig(tolerance=Tolerance(atol=1.0))
    )


@torch_available
def test_tensor_equal_bool_equal_with_tolerance() -> None:
    assert tensor_equal(
        torch.tensor([True, False]),
        torch.tensor([True, False]),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    )


@torch_available
def test_tensor_equal_bool_equal_nan() -> None:
    assert tensor_equal(
        torch.tensor([True, False]), torch.tensor([True, False]), EqualityConfig(equal_nan=True)
    )


@torch_available
def test_tensor_equal_uint64_with_tolerance() -> None:
    assert tensor_equal(
        torch.tensor([3], dtype=torch.uint64),
        torch.tensor([5], dtype=torch.uint64),
        EqualityConfig(tolerance=Tolerance(atol=2.0)),
    )


@torch_available
def test_tensor_equal_uint64_outside_the_tolerance() -> None:
    assert not tensor_equal(
        torch.tensor([3], dtype=torch.uint64),
        torch.tensor([5], dtype=torch.uint64),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    )
