from __future__ import annotations

import dataclasses
import logging
import threading
from collections import OrderedDict

import pytest

from coola.equality import (
    ComparisonResult,
    assert_objects_allclose,
    assert_objects_equal,
    compare,
)
from coola.equality.result import _CaptureHandler
from coola.equality.tester import EqualityTesterRegistry
from coola.equality.tester.interface import get_default_registry

LOGGER_NAME = "coola.equality"

np = pytest.importorskip("numpy")


@pytest.fixture
def logger_state() -> tuple[int, bool, list[logging.Handler]]:
    logger = logging.getLogger(LOGGER_NAME)
    return logger.level, logger.propagate, list(logger.handlers)


###################################
#     Tests for ComparisonResult  #
###################################


def test_comparison_result_defaults() -> None:
    result = ComparisonResult(True)
    assert result.equal
    assert result.path == ()
    assert result.reason is None
    assert result.actual == ""
    assert result.expected == ""


@pytest.mark.parametrize("equal", [True, False])
def test_comparison_result_bool(equal: bool) -> None:
    assert bool(ComparisonResult(equal)) is equal


def test_comparison_result_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        ComparisonResult(True).equal = False  # type: ignore[misc]


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


###########################
#     Tests for compare   #
###########################


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
    result = compare(actual, expected)
    assert result.equal
    assert bool(result)
    assert result.path == ()
    assert result.reason is None
    assert result.actual == repr(actual)
    assert result.expected == repr(expected)


@pytest.mark.parametrize(
    ("actual", "expected", "path"),
    [
        (1, 2, ()),
        ([1, 2, 3], [1, 2, 4], (2,)),
        ({"a": 1}, {"a": 2}, ("a",)),
        ({"a": [1, 2]}, {"a": [1, 3]}, ("a", 1)),
        ({"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}]}, ("a", 1, "b")),
        ([[1, 2], [3, 4]], [[1, 2], [3, 5]], (1, 1)),
        ({1: {2: {3: "x"}}}, {1: {2: {3: "y"}}}, (1, 2, 3)),
    ],
)
def test_compare_not_equal_path(actual: object, expected: object, path: tuple) -> None:
    result = compare(actual, expected)
    assert not result
    assert result.path == path
    assert result.reason
    assert result.actual == repr(actual)
    assert result.expected == repr(expected)


def test_compare_first_difference_only() -> None:
    result = compare([1, 2, 3], [9, 2, 9])
    assert result.path == (0,)


def test_compare_reason_is_innermost() -> None:
    result = compare({"a": [1, 2]}, {"a": [1, 3]})
    assert "numbers are different" in result.reason
    assert "actual   : 2" in result.reason
    assert "expected : 3" in result.reason


def test_compare_different_keys() -> None:
    result = compare({"a": 1}, {"b": 1})
    assert not result
    assert result.path == ()
    assert "different keys" in result.reason


def test_compare_different_lengths() -> None:
    result = compare({"a": [1]}, {"a": [1, 2]})
    assert not result
    assert result.path == ("a",)
    assert "different lengths" in result.reason


def test_compare_different_types() -> None:
    result = compare([1], (1,))
    assert not result
    assert result.path == ()
    assert "different types" in result.reason


def test_compare_shared_objects_reused() -> None:
    shared = [1, 2]
    assert compare({"a": shared, "b": shared}, {"a": [1, 2], "b": [1, 2]})


@pytest.mark.parametrize(
    ("actual", "expected", "kwargs", "equal"),
    [
        ([1.0], [1.05], {}, False),
        ([1.0], [1.05], {"atol": 0.1}, True),
        ([100.0], [101.0], {"rtol": 0.05}, True),
        ([100.0], [101.0], {"rtol": 0.001}, False),
        ([float("nan")], [float("nan")], {}, False),
        ([float("nan")], [float("nan")], {"equal_nan": True}, True),
    ],
)
def test_compare_options(actual: list, expected: list, kwargs: dict, equal: bool) -> None:
    assert bool(compare(actual, expected, **kwargs)) is equal


@pytest.mark.parametrize("name", ["atol", "rtol"])
def test_compare_negative_tolerance(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        compare(1, 1, **{name: -1.0})


def test_compare_max_depth() -> None:
    with pytest.raises(RecursionError):
        compare([[[[1]]]], [[[[2]]]], max_depth=2)


def test_compare_explicit_registry() -> None:
    registry = get_default_registry()
    assert compare([1], [1], registry=registry).equal
    assert not compare([1], [2], registry=registry)


def test_compare_custom_registry_is_used() -> None:
    registry = EqualityTesterRegistry()
    with pytest.raises(KeyError):
        compare(1, 1, registry=registry)


def test_compare_does_not_emit_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        compare([1], [2])
    assert not caplog.records


def test_compare_restores_logger(logger_state: tuple) -> None:
    compare([1], [2])
    logger = logging.getLogger(LOGGER_NAME)
    assert (logger.level, logger.propagate, list(logger.handlers)) == logger_state


def test_compare_restores_logger_on_error(logger_state: tuple) -> None:
    with pytest.raises(KeyError):
        compare(1, 1, registry=EqualityTesterRegistry())
    logger = logging.getLogger(LOGGER_NAME)
    assert (logger.level, logger.propagate, list(logger.handlers)) == logger_state


def test_compare_restores_custom_logger_level() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    level, propagate = logger.level, logger.propagate
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    try:
        assert not compare([1], [2])
        assert logger.level == logging.ERROR
        assert logger.propagate is False
    finally:
        logger.setLevel(level)
        logger.propagate = propagate


def test_compare_concurrent_threads() -> None:
    results: dict[int, ComparisonResult] = {}

    def worker(i: int) -> None:
        for _ in range(20):
            results[i] = compare({"k": [0, i]}, {"k": [0, i + 1]})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r.path == ("k", 1) for r in results.values())


#####################################
#     Tests for _CaptureHandler     #
#####################################


def _record() -> logging.LogRecord:
    return logging.LogRecord(LOGGER_NAME, logging.INFO, "", 0, "msg", None, None)


def test_capture_handler_keeps_same_thread_records() -> None:
    handler = _CaptureHandler()
    record = _record()
    handler.emit(record)
    assert handler.records == [record]


def test_capture_handler_ignores_other_threads() -> None:
    handler = _CaptureHandler()
    record = _record()
    record.thread = handler._thread_id + 1
    handler.emit(record)
    assert handler.records == []


def test_capture_handler_level() -> None:
    assert _CaptureHandler().level == logging.INFO


#####################################
#     Tests for assert_objects_equal #
#####################################


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
    with pytest.raises(AssertionError):
        assert_objects_equal([float("nan")], [float("nan")])


def test_assert_objects_equal_rejects_tolerance() -> None:
    with pytest.raises(TypeError):
        assert_objects_equal(1, 1, atol=0.1)  # type: ignore[call-arg]


def test_assert_objects_equal_is_exact() -> None:
    with pytest.raises(AssertionError):
        assert_objects_equal([1.0], [1.0 + 1e-9])


#########################################
#     Tests for assert_objects_allclose #
#########################################


def test_assert_objects_allclose_passes() -> None:
    assert_objects_allclose({"a": [1.0, 2.0]}, {"a": [1.0, 2.0 + 1e-9]})


def test_assert_objects_allclose_returns_none() -> None:
    assert assert_objects_allclose(1.0, 1.0) is None


def test_assert_objects_allclose_default_tolerances() -> None:
    assert_objects_allclose([1.0], [1.0 + 1e-9])
    with pytest.raises(AssertionError):
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
    with pytest.raises(AssertionError):
        assert_objects_allclose([float("nan")], [float("nan")])


@pytest.mark.parametrize("name", ["atol", "rtol"])
def test_assert_objects_allclose_negative_tolerance(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        assert_objects_allclose(1.0, 1.0, **{name: -1.0})


#################################
#     Additional compare tests  #
#################################


def test_public_exports() -> None:
    import coola.equality as eq

    for name in ["ComparisonResult", "assert_objects_allclose", "assert_objects_equal", "compare"]:
        assert name in eq.__all__
        assert hasattr(eq, name)


@pytest.mark.parametrize(
    ("actual", "expected", "path"),
    [
        ((1, 2, 3), (1, 2, 4), (2,)),
        (OrderedDict(a=[1, 2]), OrderedDict(a=[1, 3]), ("a", 1)),
        ({"a": (1, {"b": [1]})}, {"a": (1, {"b": [2]})}, ("a", 1, "b", 0)),
        ({(1, 2): "x"}, {(1, 2): "y"}, ((1, 2),)),
    ],
)
def test_compare_path_other_containers(actual: object, expected: object, path: tuple) -> None:
    assert compare(actual, expected).path == path


def test_compare_numpy_inside_containers() -> None:
    result = compare({"x": [np.array([1, 2])]}, {"x": [np.array([1, 3])]})
    assert not result
    assert result.path == ("x", 0)
    assert compare({"x": [np.ones(2)]}, {"x": [np.ones(2)]})


def test_compare_numpy_allclose() -> None:
    assert compare({"x": np.ones(2)}, {"x": np.ones(2) + 1e-3}, atol=1e-2)
    assert not compare({"x": np.ones(2)}, {"x": np.ones(2) + 1e-3})


def test_compare_result_is_independent_between_calls() -> None:
    first = compare([1], [2])
    second = compare([1], [1])
    assert not first
    assert second
    assert second.path == ()
    assert second.reason is None


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
    # A user handler attached directly to the logger still sees the records;
    # only propagation to ancestors is suppressed.
    assert received


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
    assert compare({"a": 1}, {"a": 1}).reason is None
