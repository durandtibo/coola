from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.jsonable import to_jsonable
from coola.jsonable.torch import get_default_transformers
from coola.testing.marks import torch_available
from coola.utils.imports import is_torch_available

if is_torch_available():
    import torch


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
