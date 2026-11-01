from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.jsonable import to_jsonable
from coola.jsonable.torch import TensorTransformer, get_default_transformers
from coola.testing.marks import torch_available
from coola.utils.imports import is_torch_available
from tests.unit.recursive.helpers import expand_not_none

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


@torch_available
def test_get_default_transformers_keys() -> None:
    assert list(get_default_transformers()) == [torch.Tensor]


def test_get_default_transformers_without_torch() -> None:
    with patch("coola.jsonable.torch.is_torch_available", lambda: False):
        assert get_default_transformers() == {}


@torch_available
def test_to_jsonable_tensor() -> None:
    assert to_jsonable(torch.arange(3)) == [0, 1, 2]


@torch_available
def test_to_jsonable_tensor_0d() -> None:
    assert to_jsonable(torch.tensor(1.5)) == 1.5


@torch_available
def test_to_jsonable_tensor_empty() -> None:
    assert to_jsonable(torch.tensor([])) == []


@torch_available
def test_to_jsonable_tensor_2d() -> None:
    assert to_jsonable(torch.ones(2, 2)) == [[1.0, 1.0], [1.0, 1.0]]


@torch_available
def test_to_jsonable_tensor_requires_grad() -> None:
    assert to_jsonable(torch.ones(2, requires_grad=True)) == [1.0, 1.0]


@torch_available
def test_to_jsonable_tensor_bfloat16() -> None:
    assert to_jsonable(torch.ones(2, dtype=torch.bfloat16)) == [1.0, 1.0]


@torch_available
def test_to_jsonable_tensor_bool() -> None:
    assert to_jsonable(torch.tensor([True, False])) == [True, False]


@torch_available
def test_to_jsonable_tensor_in_a_container() -> None:
    assert to_jsonable({"a": (torch.arange(2),)}) == {"a": [[0, 1]]}


@torch_available
def test_to_jsonable_complex_tensor_raises() -> None:
    with pytest.raises(TypeError, match=r"Cannot convert an object of type complex"):
        to_jsonable(torch.tensor([1 + 2j]))


@torch_available
def test_tensor_transformer_repr() -> None:
    assert repr(TensorTransformer()) == "TensorTransformer()"


@torch_available
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.int8, torch.int64, torch.float16, torch.bfloat16, torch.float64]
)
def test_tensor_transformer_real_tensor_has_no_children(dtype: torch.dtype) -> None:
    children, _ = expand_not_none(TensorTransformer(), torch.zeros(2, dtype=dtype))
    assert children == ()


@torch_available
def test_tensor_transformer_real_tensor_rebuild() -> None:
    _, rebuild = expand_not_none(TensorTransformer(), torch.arange(3))
    assert rebuild([]) == [0, 1, 2]


@torch_available
def test_tensor_transformer_complex_tensor_is_converted_in_turn() -> None:
    children, _ = expand_not_none(TensorTransformer(), torch.tensor([1 + 2j]))
    assert children == [[1 + 2j]]
