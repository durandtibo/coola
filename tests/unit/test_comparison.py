from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
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


##########################################
#     Tests for objects_are_allclose     #
##########################################


def test_objects_are_allclose_false_different_type() -> None:
    assert not objects_are_allclose([], ())


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (1, 1),
        (0, 0),
        (-1, -1),
        (1.0, 1.0),
        (1.0, 1.0 + 1e-9),
        (0.0, 0.0),
        (0.0, 1e-9),
        (-1.0, -1.0),
        (True, True),
        (False, False),
    ],
)
def test_objects_are_allclose_scalar_true_float(
    object1: bool | float, object2: bool | float
) -> None:
    assert objects_are_allclose(object1, object2)


@pytest.mark.parametrize(("value", "atol"), [(1.5, 1.0), (1.05, 1e-1), (1.005, 1e-2)])
def test_objects_are_allclose_scalar_true_atol(value: float, atol: float) -> None:
    assert objects_are_allclose(value, 1.0, atol=atol, rtol=0.0)


@pytest.mark.parametrize(("value", "rtol"), [(1.5, 1.0), (1.05, 1e-1), (1.005, 1e-2)])
def test_objects_are_allclose_scalar_true_rtol(value: float, rtol: float) -> None:
    assert objects_are_allclose(value, 1.0, rtol=rtol)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (1, 2),
        (1.0, 2.0),
        (1.0, 1.0 + 1e-7),
        (True, False),
        (1, 1.0),
        (1, True),
        (1.0, True),
    ],
)
def test_objects_are_allclose_scalar_false(object1: float, object2: float) -> None:
    assert not objects_are_allclose(object1, object2, rtol=0.0)


@torch_available
@pytest.mark.parametrize(
    "tensor", [torch.ones(2, 3), torch.full((2, 3), 1.0 + 1e-9), torch.full((2, 3), 1.0 - 1e-9)]
)
def test_objects_are_allclose_torch_tensor_true(tensor: torch.Tensor) -> None:
    assert objects_are_allclose(tensor, torch.ones(2, 3))


@torch_available
@pytest.mark.parametrize(
    "tensor",
    [torch.zeros(2, 3), torch.full((2, 3), 1.0 + 1e-7), torch.full((2, 3), 1.0 - 1e-7)],
)
def test_objects_are_allclose_torch_tensor_false(tensor: torch.Tensor) -> None:
    assert not objects_are_allclose(tensor, torch.ones(2, 3), rtol=0.0)


@torch_available
@pytest.mark.parametrize(
    ("tensor", "atol"),
    [
        (torch.full((2, 3), 1.5), 1),
        (torch.full((2, 3), 1.05), 1e-1),
        (torch.full((2, 3), 1.005), 1e-2),
    ],
)
def test_objects_are_allclose_torch_tensor_true_atol(tensor: torch.Tensor, atol: float) -> None:
    assert objects_are_allclose(tensor, torch.ones(2, 3), atol=atol, rtol=0.0)


@torch_available
@pytest.mark.parametrize(
    ("tensor", "rtol"),
    [
        (torch.full((2, 3), 1.5), 1),
        (torch.full((2, 3), 1.05), 1e-1),
        (torch.full((2, 3), 1.005), 1e-2),
    ],
)
def test_objects_are_allclose_torch_tensor_true_rtol(tensor: torch.Tensor, rtol: float) -> None:
    assert objects_are_allclose(tensor, torch.ones(2, 3), rtol=rtol)


@numpy_available
@pytest.mark.parametrize(
    "array", [np.ones((2, 3)), np.full((2, 3), 1.0 + 1e-9), np.full((2, 3), 1.0 - 1e-9)]
)
def test_objects_are_allclose_numpy_array_true(array: np.ndarray) -> None:
    assert objects_are_allclose(array, np.ones((2, 3)))


@numpy_available
@pytest.mark.parametrize(
    "array", [np.zeros((2, 3)), np.full((2, 3), 1.0 + 1e-7), np.full((2, 3), 1.0 - 1e-7)]
)
def test_objects_are_allclose_numpy_array_false(array: np.ndarray) -> None:
    assert not objects_are_allclose(array, np.ones((2, 3)), rtol=0.0)


@numpy_available
@pytest.mark.parametrize(
    ("array", "atol"),
    [
        (np.full((2, 3), 1.5), 1.0),
        (np.full((2, 3), 1.05), 1e-1),
        (np.full((2, 3), 1.005), 1e-2),
    ],
)
def test_objects_are_allclose_numpy_array_true_atol(array: np.ndarray, atol: float) -> None:
    assert objects_are_allclose(array, np.ones((2, 3)), atol=atol, rtol=0.0)


@numpy_available
@pytest.mark.parametrize(
    ("array", "rtol"),
    [
        (np.full((2, 3), 1.5), 1.0),
        (np.full((2, 3), 1.05), 1e-1),
        (np.full((2, 3), 1.005), 1e-2),
    ],
)
def test_objects_are_allclose_numpy_array_true_rtol(array: np.ndarray, rtol: float) -> None:
    assert objects_are_allclose(array, np.ones((2, 3)), rtol=rtol)


@torch_available
@pytest.mark.parametrize(("object1", "object2"), [([], []), ((), ()), ([1, 2, 3], [1, 2, 3])])
def test_objects_are_allclose_sequence_true(object1: Sequence, object2: Sequence) -> None:
    assert objects_are_allclose(object1, object2)


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([torch.ones(2, 3), torch.zeros(2)], [torch.ones(2, 3), torch.zeros(2)]),
        ((torch.ones(2, 3), torch.zeros(2)), (torch.ones(2, 3), torch.zeros(2))),
        (
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
        ),
    ],
)
def test_objects_are_allclose_sequence_true_torch(object1: Sequence, object2: Sequence) -> None:
    assert objects_are_allclose(object1, object2)


def test_objects_are_allclose_sequence_false() -> None:
    assert not objects_are_allclose([1, 2], [1, 3])


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({}, {}),
        (OrderedDict({}), OrderedDict({})),
        (
            {"1": torch.ones(2, 3), "2": torch.zeros(2)},
            {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        ),
        (
            {"1": torch.ones(2, 3), "2": {"3": torch.zeros(2), "4": torch.ones(2)}},
            {"1": torch.ones(2, 3), "2": {"3": torch.zeros(2), "4": torch.ones(2)}},
        ),
        (
            OrderedDict([("1", torch.ones(2, 3)), ("2", torch.zeros(2))]),
            OrderedDict([("1", torch.ones(2, 3)), ("2", torch.zeros(2))]),
        ),
    ],
)
def test_objects_are_allclose_mapping_true(object1: Mapping, object2: Mapping) -> None:
    assert objects_are_allclose(object1, object2)


def test_objects_are_allclose_mapping_false() -> None:
    assert not objects_are_allclose({"abc": 1, "def": 2}, {"abc": 1, "def": 3})


@pytest.mark.parametrize(
    ("object1", "object2"),
    [("abc", "abc"), (set(), set()), ({1, 2, 3}, {1, 2, 3})],
)
def test_objects_are_allclose_other_types_true(object1: Any, object2: Any) -> None:
    assert objects_are_allclose(object1, object2)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ("abc", "abcd"),
        (set(), ()),
        ({1, 2}, {1, 2, 3}),
        ({1, 2, 4}, {1, 2, 3}),
    ],
)
def test_objects_are_allclose_other_types_false(object1: Any, object2: Any) -> None:
    assert not objects_are_allclose(object1, object2)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (float("nan"), float("nan")),
        ([4.2, 2.3, float("nan")], [4.2, 2.3, float("nan")]),
        ([4.2, 2.3, (float("nan"), 2, 3)], [4.2, 2.3, (float("nan"), 2, 3)]),
    ],
)
def test_objects_are_allclose_scalar_equal_nan_true(object1: Any, object2: Any) -> None:
    assert objects_are_allclose(object1, object2, equal_nan=True)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (float("nan"), float("nan")),
        (float("nan"), "abc"),
        ([4.2, 2.3, float("nan")], [4.2, 2.3, float("nan")]),
        ([4.2, 2.3, (float("nan"), 2, 3)], [4.2, 2.3, (float("nan"), 2, 3)]),
    ],
)
def test_objects_are_allclose_scalar_equal_nan_false(object1: Any, object2: Any) -> None:
    assert not objects_are_allclose(object1, object2)


@numpy_available
@torch_available
def test_objects_are_allclose_true_complex_objects() -> None:
    assert objects_are_allclose(
        {
            "list": [
                1,
                2.0,
                torch.arange(5, dtype=torch.float),
                np.arange(3, dtype=float),
                [1, 2, 3],
            ],
            "tuple": ("1", (1, 2, torch.ones(2, 3), np.ones((2, 3)))),
            "dict": {"torch": torch.zeros(2, 3), "numpy": np.zeros((2, 3)), "list": []},
            "str": "abc",
            "int": 1,
            "float": 2.5,
            "torch": torch.ones(5),
            "numpy": np.ones(4),
        },
        {
            "list": [
                1,
                2.0,
                torch.arange(5, dtype=torch.float).add(3e-9),
                np.arange(3, dtype=float) + 3e-9,
                [1, 2, 3],
            ],
            "tuple": ("1", (1, 2, torch.ones(2, 3), np.ones((2, 3)))),
            "dict": {"torch": torch.zeros(2, 3), "numpy": np.zeros((2, 3)), "list": []},
            "str": "abc",
            "int": 1,
            "float": 2.5 + 1e-9,
            "torch": torch.ones(5).add(2e-9),
            "numpy": np.ones(4) + 2e-9,
        },
    )


#######################################
#     Tests for objects_are_equal     #
#######################################


def test_objects_are_equal_false_different_type() -> None:
    assert not objects_are_equal([], ())


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (1, 1),
        (0, 0),
        (-1, -1),
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, -1.0),
        (True, True),
        (False, False),
        (None, None),
    ],
)
def test_objects_are_equal_scalar_true(object1: bool | float, object2: bool | float) -> None:
    assert objects_are_equal(object1, object2)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (1, 2),
        (1.0, 2.0),
        (True, False),
        (1, 1.0),
        (1, True),
        (1.0, True),
        (1.0, None),
    ],
)
def test_objects_are_equal_scalar_false(object1: bool | float, object2: bool | float) -> None:
    assert not objects_are_equal(object1, object2)


@torch_available
def test_objects_are_equal_torch_tensor_true() -> None:
    assert objects_are_equal(torch.ones(2, 3), torch.ones(2, 3))


@torch_available
def test_objects_are_equal_torch_tensor_false() -> None:
    assert not objects_are_equal(torch.ones(2, 3), torch.zeros(2, 3))


@numpy_available
def test_objects_are_equal_numpy_array_true() -> None:
    assert objects_are_equal(np.ones((2, 3)), np.ones((2, 3)))


@numpy_available
def test_objects_are_equal_numpy_array_false() -> None:
    assert not objects_are_equal(np.ones((2, 3)), np.zeros((2, 3)))


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([], []),
        ((), ()),
        ([torch.ones(2, 3), torch.zeros(2)], [torch.ones(2, 3), torch.zeros(2)]),
        ((torch.ones(2, 3), torch.zeros(2)), (torch.ones(2, 3), torch.zeros(2))),
        (
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
        ),
    ],
)
def test_objects_are_equal_sequence_true(object1: Sequence, object2: Sequence) -> None:
    assert objects_are_equal(object1, object2)


def test_objects_are_equal_sequence_false() -> None:
    assert not objects_are_equal([1, 2], [1, 3])


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({}, {}),
        (
            {"1": torch.ones(2, 3), "2": torch.zeros(2)},
            {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        ),
        (
            {"1": torch.ones(2, 3), "2": {"3": torch.zeros(2), "4": torch.ones(2)}},
            {"1": torch.ones(2, 3), "2": {"3": torch.zeros(2), "4": torch.ones(2)}},
        ),
        (
            OrderedDict([("1", torch.ones(2, 3)), ("2", torch.zeros(2))]),
            OrderedDict([("1", torch.ones(2, 3)), ("2", torch.zeros(2))]),
        ),
    ],
)
def test_objects_are_equal_mapping_true(object1: Mapping, object2: Mapping) -> None:
    assert objects_are_equal(object1, object2)


def test_objects_are_equal_mapping_false() -> None:
    assert not objects_are_equal({"abc": 1, "def": 2}, {"abc": 1, "def": 3})


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ("abc", "abc"),
        (set(), set()),
        ({1, 2, 3}, {1, 2, 3}),
    ],
)
def test_objects_are_equal_other_types_true(object1: Any, object2: Any) -> None:
    assert objects_are_equal(object1, object2)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ("abc", "abcd"),
        (set(), ()),
        ({1, 2}, {1, 2, 3}),
        ({1, 2, 4}, {1, 2, 3}),
    ],
)
def test_objects_are_equal_other_types_false(object1: Any, object2: Any) -> None:
    assert not objects_are_equal(object1, object2)


@numpy_available
@torch_available
def test_objects_are_equal_true_complex_objects() -> None:
    assert objects_are_equal(
        {
            "list": [1, 2.0, torch.arange(5), np.arange(3), [1, 2, 3]],
            "tuple": ("1", (1, 2, torch.ones(2, 3), np.ones((2, 3)))),
            "dict": {"torch": torch.zeros(2, 3), "numpy": np.zeros((2, 3)), "list": []},
            "str": "abc",
            "int": 1,
            "float": 2.5,
            "torch": torch.ones(5),
            "numpy": np.ones(4),
        },
        {
            "list": [1, 2.0, torch.arange(5), np.arange(3), [1, 2, 3]],
            "tuple": ("1", (1, 2, torch.ones(2, 3), np.ones((2, 3)))),
            "dict": {"torch": torch.zeros(2, 3), "numpy": np.zeros((2, 3)), "list": []},
            "str": "abc",
            "int": 1,
            "float": 2.5,
            "torch": torch.ones(5),
            "numpy": np.ones(4),
        },
    )
