from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import pytest

import coola.equality as eq
from coola.equality import (
    ComparisonResult,
    Element,
    MaxDepthExceededError,
    assert_objects_allclose,
    assert_objects_equal,
    compare,
    objects_are_allclose,
    objects_are_equal,
)
from coola.equality._helpers import AttributeName
from coola.equality.tester import EqualityTesterRegistry
from coola.equality.tester.interface import get_default_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

LOGGER_NAME = "coola.equality"


def equal_result(*_args: object) -> ComparisonResult:
    return ComparisonResult(True)


def different_result(
    _actual: object, _expected: object, path: tuple, reason: str
) -> ComparisonResult:
    return ComparisonResult(False, path=path, reason=reason)


def numbers(actual: object, expected: object) -> str:
    return f"numbers are different:\n  actual   : {actual}\n  expected : {expected}"


@pytest.fixture
def logger_state() -> tuple[int, bool, list[logging.Handler]]:
    logger = logging.getLogger(LOGGER_NAME)
    return logger.level, logger.propagate, list(logger.handlers)


######################################
#     Tests for ComparisonResult     #
######################################


def test_comparison_result_defaults() -> None:
    assert ComparisonResult(True) == ComparisonResult(equal=True, path=(), reason=None)


@pytest.mark.parametrize("equal", [True, False])
def test_comparison_result_bool(equal: bool) -> None:
    assert bool(ComparisonResult(equal)) is equal


def test_comparison_result_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field 'equal'"):
        ComparisonResult(True).equal = False  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("path", "root", "expected"),
    [
        ((), "", ""),
        ((), "data", "data"),
        (("a",), "", "['a']"),
        ((2,), "x", "x[2]"),
        (("a", 1, "b"), "data", "data['a'][1]['b']"),
        ((("t", 1),), "", "[('t', 1)]"),
    ],
)
def test_comparison_result_format_path(path: tuple, root: str, expected: str) -> None:
    assert ComparisonResult(False, path=path).format_path(root) == expected


def test_comparison_result_format_message_equal() -> None:
    assert ComparisonResult(True).format_message() == "objects are equal"


def test_comparison_result_format_message_with_reason() -> None:
    result = ComparisonResult(False, path=("a",), reason="numbers are different")
    assert result.format_message("data") == (
        "objects are not equal at data['a']\nnumbers are different"
    )


def test_comparison_result_format_message_default_root() -> None:
    assert (
        ComparisonResult(False, path=(0,), reason="r")
        .format_message()
        .startswith("objects are not equal at actual[0]")
    )


def test_comparison_result_format_message_without_reason() -> None:
    result = ComparisonResult(False, path=("a",))
    assert result.format_message("data") == "objects are not equal at data['a']"


#############################
#     Tests for compare     #
#############################


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (1, 1),
        ("a", "a"),
        ([1, 2], [1, 2]),
        ({"a": [1, {"b": 2}]}, {"a": [1, {"b": 2}]}),
        ([], []),
        ({}, {}),
    ],
)
def test_compare_equal(actual: object, expected: object) -> None:
    assert compare(actual, expected) == equal_result(actual, expected)


@pytest.mark.parametrize(
    ("actual", "expected", "path", "reason"),
    [
        (1, 2, (), numbers(1, 2)),
        ([1, 2, 3], [1, 2, 4], (2,), numbers(3, 4)),
        ({"a": 1}, {"a": 2}, ("a",), numbers(1, 2)),
        ({"a": [1, 2]}, {"a": [1, 3]}, ("a", 1), numbers(2, 3)),
        ({"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}]}, ("a", 1, "b"), numbers(2, 3)),
        ([[1, 2], [3, 4]], [[1, 2], [3, 4 + 1]], (1, 1), numbers(4, 5)),
        (
            {1: {2: {3: "x"}}},
            {1: {2: {3: "y"}}},
            (1, 2, 3),
            "objects are different:\n  actual   : x\n  expected : y",
        ),
        ((1, 2, 3), (1, 2, 4), (2,), numbers(3, 4)),
        (OrderedDict(a=[1, 2]), OrderedDict(a=[1, 3]), ("a", 1), numbers(2, 3)),
        ({"a": (1, {"b": [1]})}, {"a": (1, {"b": [2]})}, ("a", 1, "b", 0), numbers(1, 2)),
        (
            {(1, 2): "x"},
            {(1, 2): "y"},
            ((1, 2),),
            "objects are different:\n  actual   : x\n  expected : y",
        ),
        (
            {"a": 1},
            {"b": 1},
            (),
            "mappings have different keys:\n  missing keys    : ['a']\n  additional keys : ['b']",
        ),
        ({"a": [1]}, {"a": [1, 2]}, ("a",), "sequences have different lengths: 1 vs 2"),
        (
            [1],
            (1,),
            (),
            "objects have different types:\n  actual   : <class 'list'>\n  expected : <class 'tuple'>",
        ),
    ],
)
def test_compare_not_equal(actual: object, expected: object, path: tuple, reason: str) -> None:
    assert compare(actual, expected) == different_result(actual, expected, path, reason)


def test_compare_first_difference_only() -> None:
    assert compare([1, 2, 3], [9, 2, 9]) == different_result(
        [1, 2, 3], [9, 2, 9], (0,), numbers(1, 9)
    )


def test_compare_shared_objects_reused() -> None:
    shared = [1, 2]
    actual, expected = {"a": shared, "b": shared}, {"a": [1, 2], "b": [1, 2]}
    assert compare(actual, expected) == equal_result(actual, expected)


@pytest.mark.parametrize(
    ("actual", "expected", "kwargs"),
    [
        ([1.0], [1.05], {"atol": 0.1}),
        ([100.0], [101.0], {"rtol": 0.05}),
        ([float("nan")], [float("nan")], {"equal_nan": True}),
    ],
)
def test_compare_options_equal(actual: list, expected: list, kwargs: dict) -> None:
    assert compare(actual, expected, **kwargs) == equal_result(actual, expected)


@pytest.mark.parametrize(
    ("actual", "expected", "kwargs", "reason"),
    [
        ([1.0], [1.05], {}, numbers(1.0, 1.05)),
        ([100.0], [101.0], {"rtol": 0.001}, numbers(100.0, 101.0)),
        ([float("nan")], [float("nan")], {}, numbers("nan", "nan")),
    ],
)
def test_compare_options_not_equal(actual: list, expected: list, kwargs: dict, reason: str) -> None:
    assert compare(actual, expected, **kwargs) == different_result(actual, expected, (0,), reason)


def test_compare_default_is_exact() -> None:
    assert not compare(1.0, 1.0 + 1e-9)


def test_compare_numpy_default_tolerance() -> None:
    assert compare(1.0, 1.0 + 1e-9, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (1.0, 1.0),
        (1.0, 1.0 + 1e-9),
        (1.0, 1.000001),
        (1.0, 1.1),
        (100.0, 100.0005),
        (100.0, 100.5),
        (0.0, 1e-9),
        (0.0, 1e-6),
        ([1.0, 2.0], [1.0, 2.0 + 1e-9]),
        ([1.0, 2.0], [1.0, 2.1]),
        ({"a": 1.0}, {"a": 1.0 + 1e-7}),
        ({"a": 1.0}, {"a": 1.01}),
        ((1.0,), (1.0 + 1e-6,)),
        ((1.0,), (1.00001,)),
        ([[1.0]], [[1.0 + 1e-9]]),
        ([[1.0]], [[2.0]]),
        (1, 1),
        (1, 2),
        (1, 1.0),
        ("a", "a"),
    ],
)
def test_allclose_entry_points_agree(actual: object, expected: object) -> None:
    assert objects_are_allclose(actual, expected) == bool(
        compare(actual, expected, rtol=1e-5, atol=1e-8)
    )


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(1.0, 1.0), (1.0, 1.0 + 1e-9), ([1.0], [1.5]), (1, 1.0), ("a", "b")],
)
def test_equal_entry_points_agree(actual: object, expected: object) -> None:
    assert objects_are_equal(actual, expected) == bool(compare(actual, expected))


def test_compare_max_depth() -> None:
    with pytest.raises(MaxDepthExceededError, match=r"Maximum recursion depth \(2\) exceeded"):
        compare([[[[1]]]], [[[[2]]]], max_depth=2)


@pytest.mark.parametrize(
    "function",
    [
        compare,
        assert_objects_equal,
        assert_objects_allclose,
        objects_are_equal,
        objects_are_allclose,
    ],
)
def test_max_depth_not_positive(function: Callable) -> None:
    with pytest.raises(ValueError, match="max_depth"):
        function(1, 1, max_depth=0)


def test_compare_explicit_registry() -> None:
    registry = get_default_registry()
    assert compare([1], [1], registry=registry) == equal_result([1], [1])


def test_compare_explicit_registry_different() -> None:
    registry = get_default_registry()
    assert compare([1], [2], registry=registry) == different_result([1], [2], (0,), numbers(1, 2))


def test_compare_custom_registry_is_used() -> None:
    registry = EqualityTesterRegistry()
    with pytest.raises(KeyError, match="is not registered"):
        compare(1, 2, registry=registry)


def test_compare_does_not_emit_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        compare([1], [2])
    assert not caplog.records


def test_compare_restores_logger(logger_state: tuple) -> None:
    compare([1], [2])
    logger = logging.getLogger(LOGGER_NAME)
    assert (logger.level, logger.propagate, list(logger.handlers)) == logger_state


def test_compare_raises_the_error_of_the_registry() -> None:
    with pytest.raises(KeyError, match="is not registered"):
        compare(1, 2, registry=EqualityTesterRegistry())


def test_compare_restores_logger_on_error(logger_state: tuple) -> None:
    with contextlib.suppress(KeyError):
        compare(1, 2, registry=EqualityTesterRegistry())
    logger = logging.getLogger(LOGGER_NAME)
    assert (logger.level, logger.propagate, list(logger.handlers)) == logger_state


@pytest.fixture
def error_logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger(LOGGER_NAME)
    level, propagate = logger.level, logger.propagate
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    yield logger
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.mark.usefixtures("error_logger")
def test_compare_with_custom_logger_level() -> None:
    assert compare([1], [2]) == different_result([1], [2], (0,), numbers(1, 2))


def test_compare_restores_custom_logger_level(error_logger: logging.Logger) -> None:
    compare([1], [2])
    assert error_logger.level == logging.ERROR


def test_compare_restores_custom_logger_propagate(error_logger: logging.Logger) -> None:
    compare([1], [2])
    assert error_logger.propagate is False


##########################################
#     Tests for assert_objects_equal     #
##########################################


def test_assert_objects_equal_passes() -> None:
    assert_objects_equal([1, {"a": 2}], [1, {"a": 2}])


def test_assert_objects_equal_returns_none() -> None:
    assert assert_objects_equal(1, 1) is None


def test_assert_objects_equal_fails_with_path() -> None:
    with pytest.raises(AssertionError, match=r"data\['a'\]\[1\]"):
        assert_objects_equal({"a": [1, 2]}, {"a": [1, 3]}, root="data")


def test_assert_objects_equal_default_root() -> None:
    with pytest.raises(AssertionError, match=r"actual\[0\]"):
        assert_objects_equal([1], [2])


def test_assert_objects_equal_message_contains_reason() -> None:
    with pytest.raises(AssertionError, match="numbers are different"):
        assert_objects_equal([1], [2])


def test_assert_objects_equal_equal_nan() -> None:
    assert_objects_equal([float("nan")], [float("nan")], equal_nan=True)
    with pytest.raises(AssertionError, match="numbers are different"):
        assert_objects_equal([float("nan")], [float("nan")])


def test_assert_objects_equal_rejects_tolerance() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'atol'"):
        assert_objects_equal(1, 1, atol=0.1)  # ty: ignore[unknown-argument]


def test_assert_objects_equal_is_exact() -> None:
    with pytest.raises(AssertionError, match="numbers are different"):
        assert_objects_equal([1.0], [1.0 + 1e-9])


#############################################
#     Tests for assert_objects_allclose     #
#############################################


def test_assert_objects_allclose_passes() -> None:
    assert_objects_allclose({"a": [1.0, 2.0]}, {"a": [1.0, 2.0 + 1e-9]})


def test_assert_objects_allclose_returns_none() -> None:
    assert assert_objects_allclose(1.0, 1.0) is None


def test_assert_objects_allclose_default_tolerances() -> None:
    assert_objects_allclose([1.0], [1.0 + 1e-9])
    with pytest.raises(AssertionError, match="numbers are different"):
        assert_objects_allclose([1.0], [1.001])


@pytest.mark.parametrize(("kwargs"), [{"atol": 0.1}, {"rtol": 0.1}])
def test_assert_objects_allclose_custom_tolerance(kwargs: dict) -> None:
    assert_objects_allclose([1.0], [1.05], **kwargs)


def test_assert_objects_allclose_fails_with_path() -> None:
    with pytest.raises(AssertionError, match=r"data\['a'\]\[1\]"):
        assert_objects_allclose({"a": [1.0, 2.0]}, {"a": [1.0, 3.0]}, root="data")


def test_assert_objects_allclose_default_root() -> None:
    with pytest.raises(AssertionError, match=r"actual\[0\]"):
        assert_objects_allclose([1.0], [2.0])


def test_assert_objects_allclose_equal_nan() -> None:
    assert_objects_allclose([float("nan")], [float("nan")], equal_nan=True)
    with pytest.raises(AssertionError, match="numbers are different"):
        assert_objects_allclose([float("nan")], [float("nan")])


@pytest.mark.parametrize("name", ["atol", "rtol"])
def test_assert_objects_allclose_negative_tolerance(name: str) -> None:
    kwargs: dict[str, Any] = {name: -1.0}
    with pytest.raises(ValueError, match=name):
        assert_objects_allclose(1.0, 1.0, **kwargs)


####################################
#     Additional compare tests     #
####################################


@pytest.mark.parametrize(
    "name", ["ComparisonResult", "assert_objects_allclose", "assert_objects_equal", "compare"]
)
def test_public_exports_are_in_all(name: str) -> None:
    assert name in eq.__all__


@pytest.mark.parametrize(
    "name", ["ComparisonResult", "assert_objects_allclose", "assert_objects_equal", "compare"]
)
def test_public_exports_are_attributes(name: str) -> None:
    assert hasattr(eq, name)


def test_compare_result_is_independent_between_calls() -> None:
    assert compare([1], [2]) == different_result([1], [2], (0,), numbers(1, 2))


def test_compare_result_is_independent_between_calls_after_a_difference() -> None:
    compare([1], [2])
    assert compare([1], [1]) == equal_result([1], [1])


def test_compare_user_handler_on_logger_receives_nothing() -> None:
    received: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            received.append(record)

    logger = logging.getLogger(LOGGER_NAME)
    handler = Collector()
    logger.addHandler(handler)
    try:
        compare([1], [2])
    finally:
        logger.removeHandler(handler)
    # Differences go to the config reporter, so nothing is logged at all.
    assert not received


def test_compare_parent_logger_handler_receives_nothing() -> None:
    received: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            received.append(record)

    parent = logging.getLogger("coola")
    handler = Collector(level=logging.DEBUG)
    parent.addHandler(handler)
    try:
        compare([1], [2])
    finally:
        parent.removeHandler(handler)
    assert received == []


def test_compare_does_not_leak_records_between_calls() -> None:
    compare([1], [2])
    assert compare({"a": 1}, {"a": 1}) == equal_result({"a": 1}, {"a": 1})


class Spy:
    def __init__(self, seen: list) -> None:
        self._seen = seen

    def __eq__(self, other: object) -> bool:
        logger = logging.getLogger(LOGGER_NAME)
        self._seen.append((logger.level, logger.propagate, list(logger.handlers)))
        return False

    def __hash__(self) -> int:
        return 0


def test_compare_calls_the_spy() -> None:
    seen: list[tuple[int, bool, list[logging.Handler]]] = []
    compare([Spy(seen)], [Spy(seen)])
    assert seen


def test_compare_does_not_modify_logger_while_running() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    before = (logger.level, logger.propagate, list(logger.handlers))
    seen: list[tuple[int, bool, list[logging.Handler]]] = []
    compare([Spy(seen)], [Spy(seen)])
    assert all(state == before for state in seen)


def test_compare_self_referencing_list() -> None:
    a: list = []
    a.append(a)
    b: list = []
    b.append(b)
    assert compare(a, b).equal


class NoRepr:
    def __repr__(self) -> str:
        msg = "repr must not be called"
        raise RuntimeError(msg)


def test_compare_equal_does_not_compute_the_repr() -> None:
    obj = NoRepr()
    assert compare([obj], [obj]).equal


def test_compare_equal_result() -> None:
    assert compare({"a": [1, 2]}, {"a": [1, 2]}) == ComparisonResult(True)


def test_compare_different_with_failing_repr() -> None:
    assert not compare([NoRepr()], [NoRepr()]).equal


@dataclasses.dataclass
class _Inner:
    x: int
    y: list[int]


def test_compare_dataclass_field_path() -> None:
    assert compare(_Inner(1, [1]), _Inner(1, [2])).path == ("y", 0)


def test_compare_dataclass_field_path_is_an_attribute_name() -> None:
    assert isinstance(compare(_Inner(1, [1]), _Inner(1, [2])).path[0], AttributeName)


def test_compare_dataclass_field_path_format() -> None:
    assert compare(_Inner(1, [1]), _Inner(1, [2])).format_path("actual") == "actual.y[0]"


def test_compare_dataclass_field_path_in_container() -> None:
    result = compare({"a": [_Inner(1, [1])]}, {"a": [_Inner(2, [1])]})
    assert result.format_path("actual") == "actual['a'][0].x"


def test_attribute_name_format_path_without_root() -> None:
    assert ComparisonResult(False, (AttributeName("x"), 0)).format_path() == ".x[0]"


def test_format_path_of_a_set_element() -> None:
    assert ComparisonResult(False, (Element(2),)).format_path("actual") == "actual{2}"


def test_compare_sets_path_is_the_element() -> None:
    assert compare({1, 2}, {1, 3}).path == (Element(2),)


def test_compare_sets_path_format() -> None:
    assert compare({1, 2}, {1, 3}).format_path("actual") == "actual{2}"
