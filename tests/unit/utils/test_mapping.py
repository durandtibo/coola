from __future__ import annotations

import pytest

from coola.utils.mapping import get_first_value, sort_by_keys, sort_by_values

##################################
#     Tests for sort_by_keys     #
##################################


def test_sort_by_keys_empty() -> None:
    assert sort_by_keys({}) == {}


def test_sort_by_keys() -> None:
    data = sort_by_keys({"dog": 1, "cat": 5, "fish": 2})
    assert data == {"cat": 5, "dog": 1, "fish": 2}


def test_sort_by_keys_order() -> None:
    data = sort_by_keys({"dog": 1, "cat": 5, "fish": 2})
    assert list(data.keys()) == ["cat", "dog", "fish"]


def test_sort_by_keys_single_item() -> None:
    """Test sort_by_keys with single-item mapping."""
    assert sort_by_keys({"key": "value"}) == {"key": "value"}


def test_sort_by_keys_numeric_keys() -> None:
    """Test sort_by_keys with numeric keys."""
    data = sort_by_keys({3: "c", 1: "a", 2: "b"})
    assert data == {1: "a", 2: "b", 3: "c"}


def test_sort_by_keys_numeric_keys_order() -> None:
    """Test sort_by_keys with numeric keys."""
    data = sort_by_keys({3: "c", 1: "a", 2: "b"})
    assert list(data.keys()) == [1, 2, 3]


def test_sort_by_keys_mixed_case() -> None:
    """Test sort_by_keys with mixed case string keys."""
    data = sort_by_keys({"Zebra": 1, "apple": 2, "Banana": 3})
    # Case-sensitive sorting: uppercase comes before lowercase in ASCII
    assert list(data.keys()) == ["Banana", "Zebra", "apple"]


####################################
#     Tests for sort_by_values     #
####################################


def test_sort_by_values_empty() -> None:
    assert sort_by_values({}) == {}


def test_sort_by_values() -> None:
    data = sort_by_values({"dog": 1, "cat": 5, "fish": 2})
    assert data == {"dog": 1, "fish": 2, "cat": 5}


def test_sort_by_values_order() -> None:
    data = sort_by_values({"dog": 1, "cat": 5, "fish": 2})
    assert list(data.keys()) == ["dog", "fish", "cat"]


def test_sort_by_values_single_item() -> None:
    """Test sort_by_values with single-item mapping."""
    assert sort_by_values({"key": 1}) == {"key": 1}


def test_sort_by_values_string_values() -> None:
    """Test sort_by_values with string values."""
    data = sort_by_values({"a": "zebra", "b": "apple", "c": "banana"})
    assert data == {"b": "apple", "c": "banana", "a": "zebra"}


def test_sort_by_values_string_values_order() -> None:
    """Test sort_by_values with string values."""
    data = sort_by_values({"a": "zebra", "b": "apple", "c": "banana"})
    assert list(data.keys()) == ["b", "c", "a"]


def test_sort_by_values_negative_values() -> None:
    """Test sort_by_values with negative numeric values."""
    data = sort_by_values({"a": -5, "b": 10, "c": -2})
    assert data == {"a": -5, "c": -2, "b": 10}


def test_sort_by_values_negative_values_order() -> None:
    """Test sort_by_values with negative numeric values."""
    data = sort_by_values({"a": -5, "b": 10, "c": -2})
    assert list(data.keys()) == ["a", "c", "b"]


def test_sort_by_values_identical_values() -> None:
    """Test sort_by_values with identical values (order undefined but
    stable)."""
    data = sort_by_values({"a": 1, "b": 1, "c": 1})
    assert all(v == 1 for v in data.values())


def test_sort_by_values_identical_values_keeps_the_length() -> None:
    data = sort_by_values({"a": 1, "b": 1, "c": 1})
    assert len(data) == 3


def test_sort_by_keys_mixed_key_types() -> None:
    assert list(sort_by_keys({"b": 1, 2: 2, "a": 3, 1: 4})) == [1, 2, "a", "b"]


def test_sort_by_keys_keys_of_the_same_type_that_cannot_be_compared() -> None:
    assert list(sort_by_keys({(1,): 1, ("a",): 2})) == [("a",), (1,)]


def test_sort_by_keys_keys_of_the_same_type_that_cannot_be_compared_is_deterministic() -> None:
    assert sort_by_keys({("a",): 2, (1,): 1}) == sort_by_keys({(1,): 1, ("a",): 2})


def test_sort_by_keys_mixed_types_that_cannot_be_compared_and_tuples() -> None:
    assert list(sort_by_keys({("a",): 1, 2: 2, (1,): 3, "b": 4})) == [2, "b", ("a",), (1,)]


def test_sort_by_keys_does_not_change_the_input() -> None:
    data = {("a",): 2, (1,): 1}
    sort_by_keys(data)
    assert list(data) == [("a",), (1,)]


#####################################
#     Tests for get_first_value     #
#####################################


def test_get_first_value_empty() -> None:
    with pytest.raises(
        ValueError, match=r"First value cannot be returned because the mapping is empty"
    ):
        get_first_value({})


def test_get_first_value() -> None:
    assert get_first_value({"key1": 1, "key2": 2}) == 1
