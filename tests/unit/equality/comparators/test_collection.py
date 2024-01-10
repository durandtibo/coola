from __future__ import annotations

import logging
from collections import OrderedDict
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
    pytest.param(ExamplePair(object1={}, object2={}), id="empty dict"),
    pytest.param(ExamplePair(object1={"a": 1, "b": 2}, object2={"a": 1, "b": 2}), id="flat dict"),
    pytest.param(
        ExamplePair(object1={"a": 1, "b": {"k": 1}}, object2={"a": 1, "b": {"k": 1}}),
        id="nested dict",
    ),
    pytest.param(
        ExamplePair(object1=OrderedDict({"a": 1, "b": 2}), object2=OrderedDict({"a": 1, "b": 2})),
        id="OrderedDict",
    ),
]

MAPPING_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            object1={"a": 1, "b": 2},
            object2={"a": 1, "c": 2},
            expected_message="mappings have different keys:",
        ),
        id="different keys",
    ),
    pytest.param(
        ExamplePair(
            object1={"a": 1, "b": 2},
            object2={"a": 1, "b": 3},
            expected_message="mappings have at least one different value:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            object1={"a": 1, "b": {"k": 1}},
            object2={"a": 1, "b": {"k": 2}},
            expected_message="mappings have at least one different value:",
        ),
        id="different values - nested",
    ),
    pytest.param(
        ExamplePair(
            object1={"a": 1, "b": 2},
            object2={"a": 1, "b": float("nan")},
            expected_message="mappings have at least one different value:",
        ),
        id="different values - nan",
    ),
    pytest.param(
        ExamplePair(
            object1={"a": 1, "b": 2},
            object2={"a": 1, "b": 2, "c": 3},
            expected_message="objects have different lengths:",
        ),
        id="different number of items",
    ),
    pytest.param(
        ExamplePair(
            object1={}, object2=OrderedDict({}), expected_message="objects have different types:"
        ),
        id="different types",
    ),
]

SEQUENCE_EQUAL = [
    pytest.param(ExamplePair(object1=[], object2=[]), id="empty list"),
    pytest.param(ExamplePair(object1=[1, 2, 3, "abc"], object2=[1, 2, 3, "abc"]), id="flat list"),
    pytest.param(
        ExamplePair(object1=[1, 2, [3, 4, 5]], object2=[1, 2, [3, 4, 5]]),
        id="nested list",
    ),
    pytest.param(
        ExamplePair(
            object1=(1, 2, 3),
            object2=(1, 2, 3),
        ),
        id="flat tuple",
    ),
]

SEQUENCE_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=[1, 2, 3],
            object2=[1, 2, 4],
            expected_message="sequences have at least one different value:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            object1=[1, 2, [3, 4, 5]],
            object2=[1, 2, [3, 4, 6]],
            expected_message="sequences have at least one different value:",
        ),
        id="different values - nested",
    ),
    pytest.param(
        ExamplePair(
            object1=[1, 2, 3],
            object2=[1, 2, float("nan")],
            expected_message="sequences have at least one different value:",
        ),
        id="different values - nan",
    ),
    pytest.param(
        ExamplePair(
            object1=[1, 2, 3],
            object2=[1, 2],
            expected_message="objects have different lengths:",
        ),
        id="different lengths",
    ),
    pytest.param(
        ExamplePair(object1=[], object2=(), expected_message="objects have different types:"),
        id="different types",
    ),
]


COLLECTION_EQUAL = MAPPING_EQUAL + SEQUENCE_EQUAL

COLLECTION_NOT_EQUAL = MAPPING_NOT_EQUAL + SEQUENCE_NOT_EQUAL


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
def test_mapping_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", MAPPING_EQUAL)
def test_mapping_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", MAPPING_NOT_EQUAL)
def test_mapping_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
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
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_mapping_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        MappingEqualityComparator().equal(
            object1={"a": float("nan"), "b": float("nan")},
            object2={"a": float("nan"), "b": float("nan")},
            config=config,
        )
        == equal_nan
    )


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({"a": np.ones((2, 3)), "b": np.zeros(2)}, {"a": np.ones((2, 3)), "b": np.zeros(2)}),
        (
            {"a": np.ones((2, 3)), "b": {"k": np.zeros(2)}},
            {"a": np.ones((2, 3)), "b": {"k": np.zeros(2)}},
        ),
    ],
)
def test_mapping_equality_comparator_equal_true_numpy(
    object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    assert MappingEqualityComparator().equal(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({"a": np.ones((2, 3)), "b": np.zeros(2)}, {"a": np.ones((2, 3)), "b": np.ones(2)}),
        (
            {"a": np.ones((2, 3)), "b": {"k": np.zeros(2)}},
            {"a": np.ones((2, 3)), "b": {"k": np.ones(2)}},
        ),
    ],
)
def test_mapping_equality_comparator_equal_false_numpy(
    object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    assert not MappingEqualityComparator().equal(object1, object2, config)


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


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([], []),
        ((), ()),
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), (1, 2, 3)),
        (["abc", "def"], ["abc", "def"]),
        (("abc", "def"), ("abc", "def")),
        ([0, ("a", "b", "c"), 2], [0, ("a", "b", "c"), 2]),
    ],
)
def test_sequence_equality_comparator_equal_true(
    caplog: pytest.LogCaptureFixture, object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1, object2, config)
        assert not caplog.messages


def test_sequence_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=[1], object2=[1], config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", SEQUENCE_EQUAL)
def test_sequence_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", SEQUENCE_NOT_EQUAL)
def test_sequence_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
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
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_sequence_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        SequenceEqualityComparator().equal(
            object1=[float("nan"), 2, float("nan")],
            object2=[float("nan"), 2, float("nan")],
            config=config,
        )
        == equal_nan
    )


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.zeros(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.zeros(2))),
    ],
)
def test_sequence_equality_comparator_equal_true_numpy(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert SequenceEqualityComparator().equal(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.ones(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.ones(2))),
    ],
)
def test_sequence_equality_comparator_equal_false_numpy(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert not SequenceEqualityComparator().equal(object1, object2, config)


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
    }
