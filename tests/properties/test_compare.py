r"""Property-based tests for the guarantees of ``compare``,
``assert_objects_equal`` and ``assert_objects_allclose``."""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest
from hypothesis import given, strategies as st

from coola.equality import (
    assert_objects_allclose,
    assert_objects_equal,
    compare,
    objects_are_allclose,
    objects_are_equal,
)
from tests.properties.strategies import rebuild, small_values, values

MISSING = object()
tolerances = st.floats(min_value=0, max_value=10, allow_nan=False)


def follow(data: Any, path: tuple[Any, ...]) -> Any:
    r"""Return the item of ``data`` at ``path``, or ``MISSING``."""
    for key in path:
        try:
            data = data[key]
        except (KeyError, IndexError, TypeError):
            return MISSING
    return data


#############################
#     Agrees with equal     #
#############################


@given(data=small_values, other=small_values)
def test_agrees_with_objects_are_equal(data: Any, other: Any) -> None:
    assert bool(compare(data, other)) == objects_are_equal(data, other)
    assert compare(data, other).equal == objects_are_equal(data, other)


@given(data=small_values, other=small_values, atol=tolerances, rtol=tolerances)
def test_agrees_with_objects_are_allclose(data: Any, other: Any, atol: float, rtol: float) -> None:
    result = compare(data, other, atol=atol, rtol=rtol)
    assert bool(result) == objects_are_allclose(data, other, atol=atol, rtol=rtol)


@given(data=values)
def test_equal_data_has_no_path_and_no_reason(data: Any) -> None:
    result = compare(data, rebuild(data))
    assert result.equal
    assert result.path == ()
    assert result.reason is None


@given(data=small_values, other=small_values)
def test_different_data_has_a_reason(data: Any, other: Any) -> None:
    result = compare(data, other)
    if not result.equal:
        assert isinstance(result.reason, str)
        assert result.reason


@given(data=small_values, other=small_values)
def test_the_representations_are_the_ones_of_the_inputs(data: Any, other: Any) -> None:
    result = compare(data, other)
    assert result.actual == repr(data)
    assert result.expected == repr(other)


######################
#     The path       #
######################


@given(data=small_values, other=small_values)
def test_path_leads_to_a_difference(data: Any, other: Any) -> None:
    result = compare(data, other)
    if result.equal:
        return
    actual, expected = follow(data, result.path), follow(other, result.path)
    # The path leads to two values that are different. The only other case is a key that is only
    # in one of the mappings.
    if actual is not MISSING and expected is not MISSING:
        assert not objects_are_equal(actual, expected)


@given(key=st.text(max_size=3), depth=st.integers(0, 4))
def test_path_is_found_at_any_depth(key: str, depth: int) -> None:
    actual: Any = {"leaf": 1}
    expected: Any = {"leaf": 2}
    path: tuple[Any, ...] = ("leaf",)
    for _ in range(depth):
        actual, expected, path = {key: actual}, {key: expected}, (key, *path)
    result = compare(actual, expected)
    assert not result.equal
    assert result.path == path


@given(items=st.lists(st.integers(-3, 3), max_size=6), index=st.integers(0, 5))
def test_path_of_a_changed_item_of_a_list_is_its_index(items: list[int], index: int) -> None:
    if index >= len(items):
        return
    other = copy.deepcopy(items)
    other[index] = items[index] + 1
    assert compare(items, other).path == (index,)


@given(data=small_values, other=small_values)
def test_message_names_the_path(data: Any, other: Any) -> None:
    result = compare(data, other)
    if result.equal:
        assert result.format_message() == "objects are equal"
        return
    message = result.format_message()
    assert message.startswith("objects are not equal at actual")
    assert result.format_path("actual") in message
    assert all(repr(key) in result.format_path("actual") for key in result.path)
    assert result.reason is None or result.reason in message
    assert result.format_message(root="data").startswith("objects are not equal at data")


#####################
#     The asserts   #
#####################


@given(data=small_values, other=small_values)
def test_assert_objects_equal_raises_iff_not_equal(data: Any, other: Any) -> None:
    if objects_are_equal(data, other):
        assert_objects_equal(data, other)
    else:
        with pytest.raises(AssertionError, match="objects are not equal at actual"):
            assert_objects_equal(data, other)


@given(data=small_values, other=small_values, atol=tolerances, rtol=tolerances)
def test_assert_objects_allclose_raises_iff_not_close(
    data: Any, other: Any, atol: float, rtol: float
) -> None:
    if objects_are_allclose(data, other, atol=atol, rtol=rtol):
        assert_objects_allclose(data, other, atol=atol, rtol=rtol)
    else:
        with pytest.raises(AssertionError, match="objects are not equal at actual"):
            assert_objects_allclose(data, other, atol=atol, rtol=rtol)


@given(data=small_values, other=small_values)
def test_assert_message_is_the_message_of_compare(data: Any, other: Any) -> None:
    result = compare(data, other)
    if result.equal:
        return
    with pytest.raises(AssertionError) as info:
        assert_objects_equal(data, other)
    assert str(info.value) == result.format_message("actual")


########################
#     Pure & stable    #
########################


@given(data=values, other=values)
def test_compare_does_not_modify_the_inputs_and_is_stable(data: Any, other: Any) -> None:
    data_before, other_before = copy.deepcopy(data), copy.deepcopy(other)
    first = compare(data, other)
    assert compare(data, other) == first
    assert objects_are_equal(data, data_before)
    assert objects_are_equal(other, other_before)


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@given(data=small_values, other=small_values)
def test_compare_does_not_log(data: Any, other: Any) -> None:
    # A fixture such as ``caplog`` cannot be used with ``given``: record the logs of the library.
    handler = RecordingHandler()
    logger = logging.getLogger("coola")
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        compare(data, other)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    assert not handler.records


@given(data=small_values, other=small_values)
def test_compare_is_symmetric_on_the_equality(data: Any, other: Any) -> None:
    assert compare(data, other).equal == compare(other, data).equal
