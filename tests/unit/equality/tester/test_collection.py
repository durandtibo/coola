from __future__ import annotations

from collections import OrderedDict, deque
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    MappingEqualityTester,
    OrderedMappingEqualityTester,
    SequenceEqualityTester,
    SetEqualityTester,
)
from coola.testing.fixtures import numpy_available
from coola.utils.imports import is_numpy_available
from tests.unit.equality.utils import ExamplePair, messages

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
            expected_message="mappings have different values for key",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual={"a": 1, "b": {"k": 1}},
            expected={"a": 1, "b": {"k": 2}},
            expected_message="mappings have different values for key",
        ),
        id="different values - nested",
    ),
    pytest.param(
        ExamplePair(
            actual={"a": 1, "b": 2},
            expected={"a": 1, "b": float("nan")},
            expected_message="mappings have different values for key",
        ),
        id="different values - nan",
    ),
    pytest.param(
        ExamplePair(
            actual={"a": 1, "b": 2},
            expected={"a": 1, "b": 2, "c": 3},
            expected_message="objects have different lengths:",
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
            expected_message="sequences have different values at index",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=[1, 2, [3, 4, 5]],
            expected=[1, 2, [3, 4, 6]],
            expected_message="sequences have different values at index",
        ),
        id="different values - nested",
    ),
    pytest.param(
        ExamplePair(
            actual=[1, 2, 3],
            expected=[1, 2, float("nan")],
            expected_message="sequences have different values at index",
        ),
        id="different values - nan",
    ),
    pytest.param(
        ExamplePair(
            actual=[1, 2, 3],
            expected=[1, 2],
            expected_message="objects have different lengths:",
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
    assert MappingEqualityTester().equal(MappingEqualityTester())


def test_mapping_equality_tester_equal_false_different_type() -> None:
    assert not MappingEqualityTester().equal(123)


def test_mapping_equality_tester_equal_false_different_type_child() -> None:
    class Child(MappingEqualityTester): ...

    assert not MappingEqualityTester().equal(Child())


def test_mapping_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = {"a": 1, "b": 2}
    assert MappingEqualityTester().objects_are_equal(obj, obj, config)


@pytest.mark.parametrize("example", MAPPING_EQUAL)
def test_mapping_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = MappingEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@pytest.mark.parametrize("example", MAPPING_EQUAL)
def test_mapping_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = MappingEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pytest.mark.parametrize("example", MAPPING_EQUAL)
def test_mapping_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = MappingEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@pytest.mark.parametrize("example", MAPPING_EQUAL)
def test_mapping_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = MappingEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pytest.mark.parametrize("example", MAPPING_NOT_EQUAL)
def test_mapping_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = MappingEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pytest.mark.parametrize("example", MAPPING_NOT_EQUAL)
def test_mapping_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = MappingEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pytest.mark.parametrize("example", MAPPING_NOT_EQUAL)
def test_mapping_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = MappingEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pytest.mark.parametrize("example", MAPPING_NOT_EQUAL)
def test_mapping_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = MappingEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert messages(config)[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_mapping_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        MappingEqualityTester().objects_are_equal(
            actual={"a": float("nan"), "b": float("nan")},
            expected={"a": float("nan"), "b": float("nan")},
            config=config,
        )
        == equal_nan
    )


@pytest.mark.parametrize("example", MAPPING_EQUAL_TOLERANCE)
def test_mapping_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert MappingEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


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
    assert MappingEqualityTester().objects_are_equal(actual, expected, config)


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
    assert not MappingEqualityTester().objects_are_equal(actual, expected, config)


##################################################
#     Tests for OrderedMappingEqualityTester     #
##################################################


def test_ordered_mapping_equality_tester_repr() -> None:
    assert repr(OrderedMappingEqualityTester()) == "OrderedMappingEqualityTester()"


def test_ordered_mapping_equality_tester_str() -> None:
    assert str(OrderedMappingEqualityTester()) == "OrderedMappingEqualityTester()"


def test_ordered_mapping_equality_tester_equal_true() -> None:
    assert OrderedMappingEqualityTester().equal(OrderedMappingEqualityTester())


def test_ordered_mapping_equality_tester_equal_false_different_type() -> None:
    assert not OrderedMappingEqualityTester().equal(MappingEqualityTester())


def test_ordered_mapping_equality_tester_objects_are_equal_true_same_order(
    config: EqualityConfig,
) -> None:
    assert OrderedMappingEqualityTester().objects_are_equal(
        OrderedDict(a=1, b=2), OrderedDict(a=1, b=2), config
    )


def test_ordered_mapping_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = OrderedDict(a=1, b=2)
    assert OrderedMappingEqualityTester().objects_are_equal(obj, obj, config)


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_order(
    config: EqualityConfig,
) -> None:
    assert not OrderedMappingEqualityTester().objects_are_equal(
        OrderedDict(a=1, b=2), OrderedDict(b=2, a=1), config
    )


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_order_explaining(
    config: EqualityConfig,
) -> None:
    config.reports = []
    assert not OrderedMappingEqualityTester().objects_are_equal(
        OrderedDict(a=1, b=2), OrderedDict(b=2, a=1), config
    )


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_order_explaining_reports(
    config: EqualityConfig,
) -> None:
    config.reports = []
    OrderedMappingEqualityTester().objects_are_equal(
        OrderedDict(a=1, b=2), OrderedDict(b=2, a=1), config
    )
    assert messages(config)[0].startswith("mappings have the same keys in a different order:")


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_values(
    config: EqualityConfig,
) -> None:
    assert not OrderedMappingEqualityTester().objects_are_equal(
        OrderedDict(a=1, b=2), OrderedDict(a=1, b=3), config
    )


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_keys(
    config: EqualityConfig,
) -> None:
    config.reports = []
    assert not OrderedMappingEqualityTester().objects_are_equal(
        OrderedDict(a=1, b=2), OrderedDict(a=1, c=2), config
    )


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_keys_logs(
    config: EqualityConfig,
) -> None:
    config.reports = []
    OrderedMappingEqualityTester().objects_are_equal(
        OrderedDict(a=1, b=2), OrderedDict(a=1, c=2), config
    )
    assert messages(config)[0].startswith("mappings have different keys:")


def test_ordered_mapping_equality_tester_objects_are_equal_false_different_type(
    config: EqualityConfig,
) -> None:
    assert not OrderedMappingEqualityTester().objects_are_equal(
        OrderedDict(a=1, b=2), {"a": 1, "b": 2}, config
    )


############################################
#     Tests for SequenceEqualityTester     #
############################################


def test_sequence_equality_tester_repr() -> None:
    assert repr(SequenceEqualityTester()) == "SequenceEqualityTester()"


def test_sequence_equality_tester_str() -> None:
    assert str(SequenceEqualityTester()) == "SequenceEqualityTester()"


def test_sequence_equality_tester_equal_true() -> None:
    assert SequenceEqualityTester().equal(SequenceEqualityTester())


def test_sequence_equality_tester_equal_false_different_type() -> None:
    assert not SequenceEqualityTester().equal(123)


def test_sequence_equality_tester_equal_false_different_type_child() -> None:
    class Child(SequenceEqualityTester): ...

    assert not SequenceEqualityTester().equal(Child())


def test_sequence_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = [1, 2, 3]
    assert SequenceEqualityTester().objects_are_equal(obj, obj, config)


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = SequenceEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = SequenceEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = SequenceEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = SequenceEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pytest.mark.parametrize("example", SEQUENCE_NOT_EQUAL)
def test_sequence_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = SequenceEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pytest.mark.parametrize("example", SEQUENCE_NOT_EQUAL)
def test_sequence_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = SequenceEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pytest.mark.parametrize("example", SEQUENCE_NOT_EQUAL)
def test_sequence_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = SequenceEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pytest.mark.parametrize("example", SEQUENCE_NOT_EQUAL)
def test_sequence_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = SequenceEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert messages(config)[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_sequence_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        SequenceEqualityTester().objects_are_equal(
            actual=[float("nan"), 2, float("nan")],
            expected=[float("nan"), 2, float("nan")],
            config=config,
        )
        == equal_nan
    )


@pytest.mark.parametrize("example", SEQUENCE_EQUAL_TOLERANCE)
def test_sequence_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert SequenceEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


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
    assert SequenceEqualityTester().objects_are_equal(actual, expected, config)


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
    assert not SequenceEqualityTester().objects_are_equal(actual, expected, config)


######################################
#     Tests for SetEqualityTester    #
######################################


def test_set_equality_tester_repr() -> None:
    assert repr(SetEqualityTester()) == "SetEqualityTester()"


def test_set_equality_tester_equal_true() -> None:
    assert SetEqualityTester().equal(SetEqualityTester())


def test_set_equality_tester_equal_false_different_type() -> None:
    assert not SetEqualityTester().equal(SequenceEqualityTester())


def test_set_equality_tester_objects_are_equal_true_same_object(config: EqualityConfig) -> None:
    obj = {1, 2}
    assert SetEqualityTester().objects_are_equal(obj, obj, config)


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
    assert SetEqualityTester().objects_are_equal(actual, expected, config)


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
    assert not SetEqualityTester().objects_are_equal(actual, expected, config)


def test_set_equality_tester_objects_are_equal_equal_nan() -> None:
    config = EqualityConfig(equal_nan=True)
    assert SetEqualityTester().objects_are_equal({float("nan")}, {float("nan")}, config)


def test_set_equality_tester_objects_are_equal_true_does_not_log() -> None:
    config = EqualityConfig(reports=[])
    SetEqualityTester().objects_are_equal({1}, {1}, config)
    assert not messages(config)


def test_set_equality_tester_objects_are_equal_false_different_type_logs() -> None:
    config = EqualityConfig(reports=[])
    SetEqualityTester().objects_are_equal({1}, frozenset({1}), config)
    assert "different types" in "\n".join(messages(config))
