from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any
from unittest.mock import Mock

import pytest

from coola import objects_are_allclose, objects_are_equal
from coola.testing import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_torch_available():
    import torch
else:
    torch = Mock()

EQUAL_FUNCTIONS: tuple[Callable[[Any, Any], bool], ...] = (
    objects_are_equal,
    partial(objects_are_allclose, atol=0.0, rtol=0.0),
)


@numpy_available
@torch_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_complex_object(equal_fn: Callable[[Any, Any], bool]) -> None:
    assert equal_fn(
        {
            "list": [1, 2.0, torch.arange(5), np.arange(3), [1, 2, 3]],
            "tuple": ("1", (1, 2, torch.ones(2, 3), np.ones((2, 3)))),
            "dict": {"torch": torch.zeros(2, 3), "numpy": np.zeros((2, 3)), "list": []},
            "str": "abc",
            "int": 1,
            "float": 2.5,
        },
        {
            "list": [1, 2.0, torch.arange(5), np.arange(3), [1, 2, 3]],
            "tuple": ("1", (1, 2, torch.ones(2, 3), np.ones((2, 3)))),
            "dict": {"torch": torch.zeros(2, 3), "numpy": np.zeros((2, 3)), "list": []},
            "str": "abc",
            "int": 1,
            "float": 2.5,
        },
    )


##################
#     Scalar     #
##################


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@pytest.mark.parametrize(("value1", "value2"), [(True, True), (1, 1), (1.0, 1.0)])
def test_consistency_equal_float_true(
    equal_fn: Callable[[Any, Any], bool],
    value1: bool | float,
    value2: bool | float,
) -> None:
    assert equal_fn(value1, value2)


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@pytest.mark.parametrize("value", [True, 1, 2.0])
def test_consistency_equal_float_false(
    equal_fn: Callable[[Any, Any], bool], value: bool | float
) -> None:
    assert not equal_fn(1.0, value)


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@pytest.mark.parametrize("value", [True, 1, 1.0, float("NaN")])
def test_consistency_equal_scalar_false_nan(
    equal_fn: Callable[[Any, Any], bool],
    value: bool | float,
) -> None:
    assert not equal_fn(float("NaN"), value)


###################
#     Mapping     #
###################

MAPPING_TYPES = (dict, OrderedDict)


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@pytest.mark.parametrize("type_fn", MAPPING_TYPES)
def test_consistency_equal_mapping_true(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
) -> None:
    assert equal_fn(type_fn({"int": 1, "str": "abc"}), type_fn({"int": 1, "str": "abc"}))


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_mapping_false_different_types(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn({"int": 1, "str": "abc"}, OrderedDict({"int": 1, "str": "abc"}))


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@pytest.mark.parametrize("type_fn", MAPPING_TYPES)
def test_consistency_equal_mapping_false_different_number_of_elements(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
) -> None:
    assert not equal_fn(
        type_fn({"int": 1, "str": "abc"}), type_fn({"int": 1, "str": "abc", "float": 0.2})
    )


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@pytest.mark.parametrize("type_fn", MAPPING_TYPES)
def test_consistency_equal_mapping_false_different_keys(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
) -> None:
    assert not equal_fn(type_fn({"int": 1, "str": "abc"}), type_fn({"int": 1, "str0": "abc"}))


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@pytest.mark.parametrize("type_fn", MAPPING_TYPES)
def test_consistency_equal_mapping_false_different_values(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
) -> None:
    assert not equal_fn(type_fn({"int": 1, "str": "abc"}), type_fn({"int": 1, "str": "abcd"}))


####################
#     Sequence     #
####################

SEQUENCE_TYPES = (list, tuple)


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@pytest.mark.parametrize("type_fn", SEQUENCE_TYPES)
def test_consistency_equal_sequence_true(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
) -> None:
    assert equal_fn(type_fn([1, 2, "abc"]), type_fn([1, 2, "abc"]))


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_sequence_false_different_types(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn([1, 2, "abc"], (1, 2, "abc"))


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@pytest.mark.parametrize("type_fn", SEQUENCE_TYPES)
def test_consistency_equal_sequence_false_different_lengths(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
) -> None:
    assert not equal_fn(type_fn([1, 2, "abc"]), type_fn([1, 2, "abc", 4]))


@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@pytest.mark.parametrize("type_fn", SEQUENCE_TYPES)
def test_consistency_equal_sequence_false_different_values(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
) -> None:
    assert not equal_fn(type_fn([1, 2, "abc"]), type_fn([1, 2, "abcd"]))


########################
#     torch.Tensor     #
########################


@torch_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_tensor_true(equal_fn: Callable[[Any, Any], bool]) -> None:
    assert equal_fn(torch.ones(2, 3), torch.ones(2, 3))


@torch_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_tensor_false_different_shapes(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn(torch.ones(2, 3), torch.ones(3, 2))


@torch_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_tensor_false_different_dtypes(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.long))


@torch_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_tensor_false_different_values(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn(torch.ones(2, 3), torch.zeros(2, 3))


@torch_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_tensor_false_different_devices(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn(
        Mock(spec=torch.Tensor, dtype=torch.float, device=torch.device("cpu")),
        Mock(spec=torch.Tensor, dtype=torch.float, device=torch.device("cuda:0")),
    )


#############################################
#     torch.nn.utils.rnn.PackedSequence     #
#############################################


@torch_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_packed_sequence_true(equal_fn: Callable[[Any, Any], bool]) -> None:
    assert equal_fn(
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )


@torch_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_packed_sequence_false_different_values(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn(
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10).view(2, 5).add(1).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )


@torch_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_packed_sequence_false_different_lengths(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn(
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 2], dtype=torch.long),
            batch_first=True,
        ),
    )


@torch_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_packed_sequence_false_different_types(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn(
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.arange(10, dtype=torch.float).view(2, 5),
    )


#########################
#     numpy.ndarray     #
#########################


@numpy_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_ndarray_true(equal_fn: Callable[[Any, Any], bool]) -> None:
    assert equal_fn(np.ones((2, 3)), np.ones((2, 3)))


@numpy_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_ndarray_false_different_shapes(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn(np.ones((2, 3)), np.ones((3, 2)))


@numpy_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_ndarray_false_different_dtypes(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn(np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int))


@numpy_available
@pytest.mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_ndarray_false_different_values(
    equal_fn: Callable[[Any, Any], bool]
) -> None:
    assert not equal_fn(np.ones((2, 3)), np.zeros((2, 3)))
