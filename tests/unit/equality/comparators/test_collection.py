from __future__ import annotations

import logging
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators import SequenceEqualityComparator
from coola.equality.comparators.collection import (
    MappingEqualityComparator,
    get_type_comparator_mapping,
)
from coola.equality.testers import EqualityTester
from coola.testing import numpy_available
from coola.utils import is_numpy_available
from tests.unit.equality.comparators.utils import ExamplePair

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


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
            expected_message="mappings have at least one different value:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual={"a": 1, "b": {"k": 1}},
            expected={"a": 1, "b": {"k": 2}},
            expected_message="mappings have at least one different value:",
        ),
        id="different values - nested",
    ),
    pytest.param(
        ExamplePair(
            actual={"a": 1, "b": 2},
            expected={"a": 1, "b": float("nan")},
            expected_message="mappings have at least one different value:",
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
            expected_message="sequences have at least one different value:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=[1, 2, [3, 4, 5]],
            expected=[1, 2, [3, 4, 6]],
            expected_message="sequences have at least one different value:",
        ),
        id="different values - nested",
    ),
    pytest.param(
        ExamplePair(
            actual=[1, 2, 3],
            expected=[1, 2, float("nan")],
            expected_message="sequences have at least one different value:",
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


###############################################
#     Tests for MappingEqualityComparator     #
###############################################


def test_mapping_equality_comparator_str() -> None:
    assert str(MappingEqualityComparator()) == "MappingEqualityComparator()"


@numpy_available
def test_mapping_equality_comparator__eq__true() -> None:
    assert MappingEqualityComparator() == MappingEqualityComparator()


@numpy_available
def test_mapping_equality_comparator__eq__false() -> None:
    assert MappingEqualityComparator() != 123


def test_mapping_equality_comparator_clone() -> None:
    op = MappingEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_mapping_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = {"a": 1, "b": 2}
    assert MappingEqualityComparator().equal(obj, obj, config)


@pytest.mark.parametrize("example", MAPPING_EQUAL)
def test_mapping_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", MAPPING_EQUAL)
def test_mapping_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", MAPPING_NOT_EQUAL)
def test_mapping_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", MAPPING_NOT_EQUAL)
def test_mapping_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_mapping_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        MappingEqualityComparator().equal(
            actual={"a": float("nan"), "b": float("nan")},
            expected={"a": float("nan"), "b": float("nan")},
            config=config,
        )
        == equal_nan
    )


@pytest.mark.parametrize("example", MAPPING_EQUAL_TOLERANCE)
def test_mapping_equality_comparator_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert MappingEqualityComparator().equal(
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
def test_mapping_equality_comparator_equal_true_numpy(
    actual: Mapping, expected: Mapping, config: EqualityConfig
) -> None:
    assert MappingEqualityComparator().equal(actual, expected, config)


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
def test_mapping_equality_comparator_equal_false_numpy(
    actual: Mapping, expected: Mapping, config: EqualityConfig
) -> None:
    assert not MappingEqualityComparator().equal(actual, expected, config)


################################################
#     Tests for SequenceEqualityComparator     #
################################################


def test_sequence_equality_comparator_str() -> None:
    assert str(SequenceEqualityComparator()) == "SequenceEqualityComparator()"


@numpy_available
def test_sequence_equality_comparator__eq__true() -> None:
    assert SequenceEqualityComparator() == SequenceEqualityComparator()


@numpy_available
def test_sequence_equality_comparator__eq__false() -> None:
    assert SequenceEqualityComparator() != 123


def test_sequence_equality_comparator_clone() -> None:
    op = SequenceEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_sequence_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = [1, 2, 3]
    assert SequenceEqualityComparator().equal(obj, obj, config)


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", SEQUENCE_NOT_EQUAL)
def test_sequence_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", SEQUENCE_NOT_EQUAL)
def test_sequence_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_sequence_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        SequenceEqualityComparator().equal(
            actual=[float("nan"), 2, float("nan")],
            expected=[float("nan"), 2, float("nan")],
            config=config,
        )
        == equal_nan
    )


@pytest.mark.parametrize("example", SEQUENCE_EQUAL_TOLERANCE)
def test_sequence_equality_comparator_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert SequenceEqualityComparator().equal(
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
def test_sequence_equality_comparator_equal_true_numpy(
    actual: Sequence, expected: Sequence, config: EqualityConfig
) -> None:
    assert SequenceEqualityComparator().equal(actual, expected, config)


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.ones(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.ones(2))),
    ],
)
def test_sequence_equality_comparator_equal_false_numpy(
    actual: Sequence, expected: Sequence, config: EqualityConfig
) -> None:
    assert not SequenceEqualityComparator().equal(actual, expected, config)


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        Mapping: MappingEqualityComparator(),
        Sequence: SequenceEqualityComparator(),
        dict: MappingEqualityComparator(),
        list: SequenceEqualityComparator(),
        tuple: SequenceEqualityComparator(),
        deque: SequenceEqualityComparator(),
    }
