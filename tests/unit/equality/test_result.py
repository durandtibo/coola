from __future__ import annotations

import dataclasses
import logging
import threading

import pytest

from coola.equality import ComparisonResult, assert_objects_equal, compare
from coola.equality.result import _CaptureHandler
from coola.equality.tester import EqualityTesterRegistry
from coola.equality.tester.interface import get_default_registry

LOGGER_NAME = "coola.equality"


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


def test_assert_objects_equal_tolerance() -> None:
    assert_objects_equal([1.0], [1.05], atol=0.1)
    with pytest.raises(AssertionError):
        assert_objects_equal([1.0], [1.05])


def test_assert_objects_equal_equal_nan() -> None:
    assert_objects_equal([float("nan")], [float("nan")], equal_nan=True)
    with pytest.raises(AssertionError):
        assert_objects_equal([float("nan")], [float("nan")])


def test_assert_objects_equal_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="atol"):
        assert_objects_equal(1, 1, atol=-1.0)
