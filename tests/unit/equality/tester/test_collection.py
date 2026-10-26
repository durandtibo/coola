from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.engine import Engine
from coola.equality.tester import (
    MappingEqualityTester,
    OrderedMappingEqualityTester,
    SequenceEqualityTester,
    SetEqualityTester,
)
from coola.equality.tolerance import Tolerance
from coola.testing.fixtures import numpy_available
from coola.utils.imports import is_numpy_available
from tests.unit.equality.utils import (
    ExamplePair,
    ExplainingConfig,
    explaining,
    messages,
    run_tester,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, Set as AbstractSet

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


MAPPING_EQUAL = [
    pytest.param(ExamplePair(actual={}, expected={}), id="empty dict"),
    pytest.param(ExamplePair(actual={"a": 1, "b": 2}, expected={"a": 1, "b": 2}), id="flat dict"),
    pytest.param(
        ExamplePair(actual={"a": 1, "b": {"k": 1}}, expected={"a": 1, "b": {"k": 1}}),
        id="nested dict",
    ),
    pytest.param(
        ExamplePair(actual=OrderedDict({"a": 1, "b": 2}), expected=OrderedDict({"a": 1, "b": 2})),
        id="OrderedDict",
    ),
]

MAPPING_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual={"a": 1, "b": 2},
            expected={"a": 1, "c": 2},
            expected_message="mappings have different keys:",
        ),
        id="different keys",
    ),
    pytest.param(
        ExamplePair(
            actual={"a": 1, "b": 2},
            expected={"a": 1, "b": 3},
            expected_message="numbers are different:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual={"a": 1, "b": {"k": 1}},
            expected={"a": 1, "b": {"k": 2}},
            expected_message="numbers are different:",
        ),
        id="different values - nested",
    ),
    pytest.param(
        ExamplePair(
            actual={"a": 1, "b": 2},
            expected={"a": 1, "b": float("nan")},
            expected_message="objects have different types:",
        ),
        id="different values - nan",
    ),
    pytest.param(
        ExamplePair(
            actual={"a": 1, "b": 2},
            expected={"a": 1, "b": 2, "c": 3},
            expected_message="mappings have different keys:",
        ),
        id="different number of items",
    ),
    pytest.param(
        ExamplePair(
            actual={}, expected=OrderedDict({}), expected_message="objects have different types:"
        ),
        id="different types",
    ),
]

MAPPING_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(actual={"a": 1, "b": 2}, expected={"a": 1, "b": 3}, atol=1.0),
        id="flat dict atol=1",
    ),
    pytest.param(
        ExamplePair(actual={"a": 1, "b": {"k": 2}}, expected={"a": 1, "b": {"k": 3}}, atol=1.0),
        id="nested dict atol=1",
    ),
    pytest.param(
        ExamplePair(actual={"a": 1, "b": 2}, expected={"a": 4, "b": -2}, atol=10.0),
        id="flat dict atol=10",
    ),
    pytest.param(
        ExamplePair(actual={"a": 1.0, "b": 2.0}, expected={"a": 1.0001, "b": 1.9999}, atol=1e-3),
        id="flat dict atol=1e-3",
    ),
    # rtol
    pytest.param(
        ExamplePair(actual={"a": 1, "b": 2}, expected={"a": 1, "b": 3}, rtol=1.0),
        id="flat dict rtol=1",
    ),
    pytest.param(
        ExamplePair(actual={"a": 1, "b": {"k": 2}}, expected={"a": 1, "b": {"k": 3}}, rtol=1.0),
        id="nested dict rtol=1",
    ),
    pytest.param(
        ExamplePair(actual={"a": 1, "b": 2}, expected={"a": 4, "b": -2}, rtol=10.0),
        id="flat dict rtol=10",
    ),
    pytest.param(
        ExamplePair(actual={"a": 1.0, "b": 2.0}, expected={"a": 1.0001, "b": 1.9999}, rtol=1e-3),
        id="flat dict rtol=1e-3",
    ),
]

SEQUENCE_EQUAL = [
    pytest.param(ExamplePair(actual=[], expected=[]), id="empty list"),
    pytest.param(ExamplePair(actual=(), expected=()), id="empty tuple"),
    pytest.param(ExamplePair(actual=deque(), expected=deque()), id="empty deque"),
    pytest.param(ExamplePair(actual=[1, 2, 3, "abc"], expected=[1, 2, 3, "abc"]), id="flat list"),
    pytest.param(
        ExamplePair(actual=[1, 2, [3, 4, 5]], expected=[1, 2, [3, 4, 5]]),
        id="nested list",
    ),
    pytest.param(
        ExamplePair(
            actual=(1, 2, 3),
            expected=(1, 2, 3),
        ),
        id="flat tuple",
    ),
    pytest.param(ExamplePair(actual=deque([1, 2, 3]), expected=deque([1, 2, 3])), id="deque"),
]

SEQUENCE_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=[1, 2, 3],
            expected=[1, 2, 4],
            expected_message="numbers are different:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=[1, 2, [3, 4, 5]],
            expected=[1, 2, [3, 4, 6]],
            expected_message="numbers are different:",
        ),
        id="different values - nested",
    ),
    pytest.param(
        ExamplePair(
            actual=[1, 2, 3],
            expected=[1, 2, float("nan")],
            expected_message="objects have different types:",
        ),
        id="different values - nan",
    ),
    pytest.param(
        ExamplePair(
            actual=[1, 2, 3],
            expected=[1, 2],
            expected_message="sequences have different lengths:",
        ),
        id="different lengths",
    ),
    pytest.param(
        ExamplePair(actual=[], expected=(), expected_message="objects have different types:"),
        id="different types",
    ),
]

SEQUENCE_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(actual=[1, 2, 3], expected=[1, 2, 4], atol=1.0), id="flat sequence atol=1"
    ),
    pytest.param(
        ExamplePair(actual=[1, 2, [3, 4, 5]], expected=[1, 2, [4, 5, 6]], atol=1.0),
        id="nested sequence atol=1",
    ),
    pytest.param(
        ExamplePair(actual=[1, 2, 3], expected=[-4, 5, 6], atol=10.0), id="flat sequence atol=10"
    ),
    pytest.param(
        ExamplePair(actual=[1.0, 2.0, 3.0], expected=[1.0001, 1.9999, 3.0001], atol=1e-3),
        id="flat sequence atol=1e-3",
    ),
    # rtol
    pytest.param(
        ExamplePair(actual=[1, 2, 3], expected=[1, 2, 4], rtol=1.0), id="flat sequence rtol=1"
    ),
    pytest.param(
        ExamplePair(actual=[1, 2, [3, 4, 5]], expected=[1, 2, [4, 5, 6]], rtol=1.0),
        id="nested sequence rtol=1",
    ),
    pytest.param(
        ExamplePair(actual=[1, 2, 3], expected=[-4, 5, 6], rtol=10.0), id="flat sequence rtol=10"
    ),
    pytest.param(
        ExamplePair(actual=[1.0, 2.0, 3.0], expected=[1.0001, 1.9999, 3.0001], rtol=1e-3),
        id="flat sequence rtol=1e-3",
    ),
]

COLLECTION_EQUAL = MAPPING_EQUAL + SEQUENCE_EQUAL
COLLECTION_NOT_EQUAL = MAPPING_NOT_EQUAL + SEQUENCE_NOT_EQUAL
COLLECTION_EQUAL_TOLERANCE = MAPPING_EQUAL_TOLERANCE + SEQUENCE_EQUAL_TOLERANCE


###########################################
#     Tests for MappingEqualityTester     #
###########################################


def test_mapping_equality_tester_repr() -> None:
    assert repr(MappingEqualityTester()) == "MappingEqualityTester()"


def test_mapping_equality_tester_str() -> None:
    assert str(MappingEqualityTester()) == "MappingEqualityTester()"


def test_mapping_equality_tester_equal_true() -> None:
    assert MappingEqualityTester() == MappingEqualityTester()


def test_mapping_equality_tester_equal_false_different_type() -> None:
    assert MappingEqualityTester() != 123


def test_mapping_equality_tester_equal_false_different_type_child() -> None:
    class Child(MappingEqualityTester): ...

    assert MappingEqualityTester() != Child()


def test_mapping_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = {"a": 1, "b": 2}
    assert run_tester(MappingEqualityTester(), obj, obj, config)


@pytest.mark.parametrize("example", MAPPING_EQUAL)
def test_mapping_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = MappingEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", MAPPING_EQUAL)
def test_mapping_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = MappingEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", MAPPING_EQUAL)
def test_mapping_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = MappingEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", MAPPING_EQUAL)
def test_mapping_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = MappingEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", MAPPING_NOT_EQUAL)
def test_mapping_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = MappingEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", MAPPING_NOT_EQUAL)
def test_mapping_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = MappingEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", MAPPING_NOT_EQUAL)
def test_mapping_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = MappingEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", MAPPING_NOT_EQUAL)
def test_mapping_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = MappingEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_mapping_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config = replace(config, equal_nan=equal_nan)
    assert (
        run_tester(
            MappingEqualityTester(),
            {"a": float("nan"), "b": float("nan")},
            {"a": float("nan"), "b": float("nan")},
            config,
        )
        == equal_nan
    )


@pytest.mark.parametrize("example", MAPPING_EQUAL_TOLERANCE)
def test_mapping_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config = replace(config, tolerance=Tolerance(atol=example.atol, rtol=example.rtol))
    assert run_tester(MappingEqualityTester(), example.actual, example.expected, config)


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({"a": np.ones((2, 3)), "b": np.zeros(2)}, {"a": np.ones((2, 3)), "b": np.zeros(2)}),
        (
            {"a": np.ones((2, 3)), "b": {"k": np.zeros(2)}},
            {"a": np.ones((2, 3)), "b": {"k": np.zeros(2)}},
        ),
    ],
)
def test_mapping_equality_tester_objects_are_equal_true_numpy(
    actual: Mapping, expected: Mapping, config: EqualityConfig
) -> None:
    assert run_tester(MappingEqualityTester(), actual, expected, config)


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({"a": np.ones((2, 3)), "b": np.zeros(2)}, {"a": np.ones((2, 3)), "b": np.ones(2)}),
        (
            {"a": np.ones((2, 3)), "b": {"k": np.zeros(2)}},
            {"a": np.ones((2, 3)), "b": {"k": np.ones(2)}},
        ),
    ],
)
def test_mapping_equality_tester_objects_are_equal_false_numpy(
    actual: Mapping, expected: Mapping, config: EqualityConfig
) -> None:
    assert not run_tester(MappingEqualityTester(), actual, expected, config)


##################################################
#     Tests for OrderedMappingEqualityTester     #
##################################################


def test_ordered_mapping_equality_tester_repr() -> None:
    assert repr(OrderedMappingEqualityTester()) == "OrderedMappingEqualityTester()"


def test_ordered_mapping_equality_tester_str() -> None:
    assert str(OrderedMappingEqualityTester()) == "OrderedMappingEqualityTester()"


def test_ordered_mapping_equality_tester_equal_true() -> None:
    assert OrderedMappingEqualityTester() == OrderedMappingEqualityTester()


def test_ordered_mapping_equality_tester_equal_false_different_type() -> None:
    assert OrderedMappingEqualityTester() != MappingEqualityTester()


def test_ordered_mapping_equality_tester_objects_are_equal_true_same_order(
    config: EqualityConfig,
) -> None:
    assert run_tester(
        OrderedMappingEqualityTester(), OrderedDict(a=1, b=2), OrderedDict(a=1, b=2), config
    )


def test_ordered_mapping_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = OrderedDict(a=1, b=2)
    assert run_tester(OrderedMappingEqualityTester(), obj, obj, config)


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_order(
    config: EqualityConfig,
) -> None:
    assert not run_tester(
        OrderedMappingEqualityTester(), OrderedDict(a=1, b=2), OrderedDict(b=2, a=1), config
    )


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_order_explaining(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    assert not run_tester(
        OrderedMappingEqualityTester(), OrderedDict(a=1, b=2), OrderedDict(b=2, a=1), config
    )


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_order_explaining_reports(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    run_tester(OrderedMappingEqualityTester(), OrderedDict(a=1, b=2), OrderedDict(b=2, a=1), config)
    assert messages(config)[0].startswith("mappings have the same keys in a different order:")


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_values(
    config: EqualityConfig,
) -> None:
    assert not run_tester(
        OrderedMappingEqualityTester(), OrderedDict(a=1, b=2), OrderedDict(a=1, b=3), config
    )


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_keys(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    assert not run_tester(
        OrderedMappingEqualityTester(), OrderedDict(a=1, b=2), OrderedDict(a=1, c=2), config
    )


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_keys_logs(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    run_tester(OrderedMappingEqualityTester(), OrderedDict(a=1, b=2), OrderedDict(a=1, c=2), config)
    assert messages(config)[0].startswith("mappings have different keys:")


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_type(
    config: EqualityConfig,
) -> None:
    assert not run_tester(
        OrderedMappingEqualityTester(), OrderedDict(a=1, b=2), {"a": 1, "b": 2}, config
    )


############################################
#     Tests for SequenceEqualityTester     #
############################################


def test_sequence_equality_tester_repr() -> None:
    assert repr(SequenceEqualityTester()) == "SequenceEqualityTester()"


def test_sequence_equality_tester_str() -> None:
    assert str(SequenceEqualityTester()) == "SequenceEqualityTester()"


def test_sequence_equality_tester_equal_true() -> None:
    assert SequenceEqualityTester() == SequenceEqualityTester()


def test_sequence_equality_tester_equal_false_different_type() -> None:
    assert SequenceEqualityTester() != 123


def test_sequence_equality_tester_equal_false_different_type_child() -> None:
    class Child(SequenceEqualityTester): ...

    assert SequenceEqualityTester() != Child()


def test_sequence_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = [1, 2, 3]
    assert run_tester(SequenceEqualityTester(), obj, obj, config)


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = SequenceEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = SequenceEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = SequenceEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = SequenceEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", SEQUENCE_NOT_EQUAL)
def test_sequence_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = SequenceEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", SEQUENCE_NOT_EQUAL)
def test_sequence_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = SequenceEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", SEQUENCE_NOT_EQUAL)
def test_sequence_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = SequenceEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", SEQUENCE_NOT_EQUAL)
def test_sequence_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = SequenceEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_sequence_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config = replace(config, equal_nan=equal_nan)
    assert (
        run_tester(
            SequenceEqualityTester(),
            [float("nan"), 2, float("nan")],
            [float("nan"), 2, float("nan")],
            config,
        )
        == equal_nan
    )


@pytest.mark.parametrize("example", SEQUENCE_EQUAL_TOLERANCE)
def test_sequence_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config = replace(config, tolerance=Tolerance(atol=example.atol, rtol=example.rtol))
    assert run_tester(SequenceEqualityTester(), example.actual, example.expected, config)


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.zeros(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.zeros(2))),
        (deque([np.ones((2, 3)), np.zeros(2)]), deque([np.ones((2, 3)), np.zeros(2)])),
    ],
)
def test_sequence_equality_tester_objects_are_equal_true_numpy(
    actual: Sequence, expected: Sequence, config: EqualityConfig
) -> None:
    assert run_tester(SequenceEqualityTester(), actual, expected, config)


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.ones(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.ones(2))),
    ],
)
def test_sequence_equality_tester_objects_are_equal_false_numpy(
    actual: Sequence, expected: Sequence, config: EqualityConfig
) -> None:
    assert not run_tester(SequenceEqualityTester(), actual, expected, config)


######################################
#     Tests for SetEqualityTester    #
######################################


def test_set_equality_tester_repr() -> None:
    assert repr(SetEqualityTester()) == "SetEqualityTester()"


def test_set_equality_tester_equal_true() -> None:
    assert SetEqualityTester() == SetEqualityTester()


def test_set_equality_tester_equal_false_different_type() -> None:
    assert SetEqualityTester() != SequenceEqualityTester()


def test_set_equality_tester_objects_are_equal_true_same_object(config: EqualityConfig) -> None:
    obj = {1, 2}
    assert run_tester(SetEqualityTester(), obj, obj, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (set(), set()),
        ({1, 2, 3}, {3, 2, 1}),
        (frozenset({"a"}), frozenset({"a"})),
        ({(1, 2.5)}, {(1, 2.5)}),
    ],
)
def test_set_equality_tester_objects_are_equal_true(
    config: EqualityConfig, actual: AbstractSet, expected: AbstractSet
) -> None:
    assert run_tester(SetEqualityTester(), actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({1, 2}, {1, 3}),
        ({1, 2}, {1}),
        ({1}, frozenset({1})),
        ({1}, {True}),
    ],
)
def test_set_equality_tester_objects_are_equal_false(
    config: EqualityConfig, actual: AbstractSet, expected: AbstractSet
) -> None:
    assert not run_tester(SetEqualityTester(), actual, expected, config)


def test_set_equality_tester_objects_are_equal_equal_nan() -> None:
    config = EqualityConfig(equal_nan=True)
    assert run_tester(SetEqualityTester(), {float("nan")}, {float("nan")}, config)


def test_set_equality_tester_objects_are_equal_true_does_not_log() -> None:
    config = ExplainingConfig()
    run_tester(SetEqualityTester(), {1}, {1}, config)
    assert not messages(config)


def test_set_equality_tester_objects_are_equal_false_different_type_logs() -> None:
    config = ExplainingConfig()
    run_tester(SetEqualityTester(), {1}, frozenset({1}), config)
    assert "different types" in "\n".join(messages(config))


###################################################
#     Tests for the fast path of the sequences    #
###################################################


@pytest.mark.parametrize(
    ("actual", "expected"),
    [([False], [0]), ((2, False), (2, 0)), ([1, 2], [True, 2]), ([1, "a"], [1.0, "a"])],
)
def test_sequence_equality_tester_exact_scalars_with_different_item_types(
    config: EqualityConfig, actual: Sequence, expected: Sequence
) -> None:
    # ``False == 0`` and ``1 == 1.0``, but the types of the items must match.
    assert not run_tester(SequenceEqualityTester(), actual, expected, config)


def test_sequence_equality_tester_exact_scalars_equal(config: EqualityConfig) -> None:
    assert run_tester(SequenceEqualityTester(), [1, "a", None, True], [1, "a", None, True], config)


def test_sequence_equality_tester_exact_scalars_different(config: EqualityConfig) -> None:
    assert not run_tester(SequenceEqualityTester(), [1, 2, 3], [1, 2, 4], config)


def test_sequence_equality_tester_exact_scalars_different_with_tolerance(
    config: EqualityConfig,
) -> None:
    config = replace(config, tolerance=Tolerance(atol=1.0))
    assert run_tester(SequenceEqualityTester(), [1, 2, 3], [1, 2, 4], config)


def test_sequence_equality_tester_exact_scalars_different_explaining_path(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    run_tester(SequenceEqualityTester(), [1, 2, 3], [1, 2, 4], config)
    assert config.reports[-1][1] == (2,)


def test_sequence_equality_tester_exact_scalars_different_does_not_visit_the_items() -> None:
    config = EqualityConfig()
    tester = SequenceEqualityTester()
    result = tester.objects_are_equal([1, 2, 3], [1, 2, 4], Engine(config))
    assert result is False


def test_mapping_equality_tester_string_keys_in_a_different_order(config: EqualityConfig) -> None:
    assert run_tester(MappingEqualityTester(), {"a": 1, "b": 2}, {"b": 2, "a": 1}, config)


def test_mapping_equality_tester_integer_keys(config: EqualityConfig) -> None:
    assert run_tester(MappingEqualityTester(), {1: "a", 2: "b"}, {2: "b", 1: "a"}, config)


def test_mapping_equality_tester_integer_key_against_boolean_key(config: EqualityConfig) -> None:
    assert not run_tester(MappingEqualityTester(), {1: "a"}, {True: "a"}, config)


def test_mapping_equality_tester_tuple_keys(config: EqualityConfig) -> None:
    assert run_tester(MappingEqualityTester(), {(1, 2): "a"}, {(1, 2): "a"}, config)


def test_mapping_equality_tester_mapping_proxy_keys(config: EqualityConfig) -> None:
    assert run_tester(
        MappingEqualityTester(), MappingProxyType({"a": 1}), MappingProxyType({"a": 1}), config
    )
