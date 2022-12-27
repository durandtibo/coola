from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any, Union
from unittest.mock import Mock

import numpy as np
import torch
from pytest import mark
from torch import Tensor

from coola import objects_are_allclose, objects_are_equal

EQUAL_FUNCTIONS: tuple[Callable[[Any, Any], bool], ...] = (
    objects_are_equal,
    partial(objects_are_allclose, atol=0.0, rtol=0.0),
)

##################
#     Scalar     #
##################


@mark.parametrize("value1", (True, 1, 1.0))
@mark.parametrize("value2", (True, 1, 1.0))
@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_scalar_true(
    equal_fn: Callable[[Any, Any], bool],
    value1: Union[bool, int, float],
    value2: Union[bool, int, float],
):
    assert equal_fn(value1, value2)  # TODO: add an option for strict types???
    assert value1 == value2


###################
#     Mapping     #
###################

MAPPING_TYPES = (dict, OrderedDict)


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@mark.parametrize("type_fn", MAPPING_TYPES)
def test_consistency_equal_mapping_true(equal_fn: Callable[[Any, Any], bool], type_fn: Callable):
    value1 = type_fn({"int": 1, "str": "abc"})
    value2 = type_fn({"int": 1, "str": "abc"})
    assert equal_fn(value1, value2)
    assert value1 == value2


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_mapping_false_different_types(equal_fn: Callable[[Any, Any], bool]):
    value1 = {"int": 1, "str": "abc"}
    value2 = OrderedDict({"int": 1, "str": "abc"})
    assert not equal_fn(value1, value2)
    assert value1 != value2


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@mark.parametrize("type_fn", MAPPING_TYPES)
def test_consistency_equal_mapping_false_different_number_of_elements(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
):
    value1 = type_fn({"int": 1, "str": "abc"})
    value2 = type_fn({"int": 1, "str": "abc", "float": 0.2})
    assert not equal_fn(value1, value2)
    assert value1 != value2


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@mark.parametrize("type_fn", MAPPING_TYPES)
def test_consistency_equal_mapping_false_different_keys(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
):
    value1 = type_fn({"int": 1, "str": "abc"})
    value2 = type_fn({"int": 1, "str0": "abc"})
    assert not equal_fn(value1, value2)
    assert value1 != value2


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@mark.parametrize("type_fn", MAPPING_TYPES)
def test_consistency_equal_mapping_false_different_values(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
):
    value1 = type_fn({"int": 1, "str": "abc"})
    value2 = type_fn({"int": 1, "str": "abcd"})
    assert not equal_fn(value1, value2)
    assert value1 != value2


####################
#     Sequence     #
####################

SEQUENCE_TYPES = (list, tuple)


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@mark.parametrize("type_fn", SEQUENCE_TYPES)
def test_consistency_equal_sequence_true(equal_fn: Callable[[Any, Any], bool], type_fn: Callable):
    assert equal_fn(type_fn([1, 2, "abc"]), type_fn([1, 2, "abc"]))
    value1 = type_fn([1, 2, "abc"])
    value2 = type_fn([1, 2, "abc"])
    assert equal_fn(value1, value2)
    assert value1 == value2


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_sequence_false_different_types(equal_fn: Callable[[Any, Any], bool]):
    value1 = [1, 2, "abc"]
    value2 = (1, 2, "abc")
    assert not equal_fn(value1, value2)
    assert value1 != value2


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@mark.parametrize("type_fn", SEQUENCE_TYPES)
def test_consistency_equal_sequence_false_different_lengths(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
):
    value1 = type_fn([1, 2, "abc"])
    value2 = type_fn([1, 2, "abc", 4])
    assert not equal_fn(value1, value2)
    assert value1 != value2


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
@mark.parametrize("type_fn", SEQUENCE_TYPES)
def test_consistency_equal_sequence_false_different_values(
    equal_fn: Callable[[Any, Any], bool], type_fn: Callable
):
    value1 = type_fn([1, 2, "abc"])
    value2 = type_fn([1, 2, "abcd"])
    assert not equal_fn(value1, value2)
    assert value1 != value2


########################
#     torch.Tensor     #
########################


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_tensor_true(equal_fn: Callable[[Any, Any], bool]):
    assert equal_fn(torch.ones(2, 3), torch.ones(2, 3))


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_tensor_false_different_shapes(equal_fn: Callable[[Any, Any], bool]):
    assert not equal_fn(torch.ones(2, 3), torch.ones(3, 2))


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_tensor_false_different_dtypes(equal_fn: Callable[[Any, Any], bool]):
    assert not equal_fn(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.long))


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_tensor_false_different_values(equal_fn: Callable[[Any, Any], bool]):
    assert not equal_fn(torch.ones(2, 3), torch.zeros(2, 3))


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_tensor_false_different_devices(equal_fn: Callable[[Any, Any], bool]):
    assert not equal_fn(
        Mock(spec=Tensor, dtype=torch.float, device=torch.device("cpu")),
        Mock(spec=Tensor, dtype=torch.float, device=torch.device("cuda:0")),
    )


#########################
#     numpy.ndarray     #
#########################


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_ndarray_true(equal_fn: Callable[[Any, Any], bool]):
    assert equal_fn(np.ones((2, 3)), np.ones((2, 3)))


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_ndarray_false_different_shapes(equal_fn: Callable[[Any, Any], bool]):
    assert not equal_fn(np.ones((2, 3)), np.ones((3, 2)))


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_ndarray_false_different_dtypes(equal_fn: Callable[[Any, Any], bool]):
    assert not equal_fn(np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int))


@mark.parametrize("equal_fn", EQUAL_FUNCTIONS)
def test_consistency_equal_ndarray_false_different_values(equal_fn: Callable[[Any, Any], bool]):
    assert not equal_fn(np.ones((2, 3)), np.zeros((2, 3)))
