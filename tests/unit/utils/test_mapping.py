from __future__ import annotations

from coola.utils.mapping import sort_by_keys, sort_by_values

##################################
#     Tests for sort_by_keys     #
##################################


def test_sort_by_keys_empty() -> None:
    assert sort_by_keys({}) == {}


def test_sort_by_keys() -> None:
    data = sort_by_keys({"dog": 1, "cat": 5, "fish": 2})
    assert data == {"cat": 5, "dog": 1, "fish": 2}
    assert list(data.keys()) == ["cat", "dog", "fish"]


def test_sort_by_keys_single_item() -> None:
    """Test sort_by_keys with single-item mapping."""
    assert sort_by_keys({"key": "value"}) == {"key": "value"}


def test_sort_by_keys_numeric_keys() -> None:
    """Test sort_by_keys with numeric keys."""
    data = sort_by_keys({3: "c", 1: "a", 2: "b"})
    assert data == {1: "a", 2: "b", 3: "c"}
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
    assert list(data.keys()) == ["dog", "fish", "cat"]


def test_sort_by_values_single_item() -> None:
    """Test sort_by_values with single-item mapping."""
    assert sort_by_values({"key": 1}) == {"key": 1}


def test_sort_by_values_string_values() -> None:
    """Test sort_by_values with string values."""
    data = sort_by_values({"a": "zebra", "b": "apple", "c": "banana"})
    assert data == {"b": "apple", "c": "banana", "a": "zebra"}
    assert list(data.keys()) == ["b", "c", "a"]


def test_sort_by_values_negative_values() -> None:
    """Test sort_by_values with negative numeric values."""
    data = sort_by_values({"a": -5, "b": 10, "c": -2})
    assert data == {"a": -5, "c": -2, "b": 10}
    assert list(data.keys()) == ["a", "c", "b"]


def test_sort_by_values_identical_values() -> None:
    """Test sort_by_values with identical values (order undefined but stable)."""
    data = sort_by_values({"a": 1, "b": 1, "c": 1})
    assert all(v == 1 for v in data.values())
    assert len(data) == 3
