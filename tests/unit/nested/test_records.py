from __future__ import annotations

from collections import deque

import pytest

from coola.nested import convert_to_dict_of_lists, convert_to_list_of_dicts

##############################################
#     Tests for convert_to_dict_of_lists     #
##############################################


def test_convert_to_dict_of_lists_empty_list() -> None:
    """Test convert_to_dict_of_lists with empty list returns empty
    dict."""
    assert convert_to_dict_of_lists([]) == {}


def test_convert_to_dict_of_lists_empty_dict() -> None:
    """Test convert_to_dict_of_lists with list containing empty dict."""
    assert convert_to_dict_of_lists([{}]) == {}


def test_convert_to_dict_of_lists() -> None:
    """Test convert_to_dict_of_lists with standard list of dicts."""
    assert convert_to_dict_of_lists(
        [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}]
    ) == {
        "key1": [1, 2, 3],
        "key2": [10, 20, 30],
    }


def test_convert_to_dict_of_lists_single_item() -> None:
    """Test convert_to_dict_of_lists with single-item list."""
    assert convert_to_dict_of_lists([{"key1": 1, "key2": 2}]) == {
        "key1": [1],
        "key2": [2],
    }


def test_convert_to_dict_of_lists_different_types() -> None:
    """Test convert_to_dict_of_lists with different value types."""
    assert convert_to_dict_of_lists([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]) == {
        "name": ["Alice", "Bob"],
        "age": [30, 25],
    }


def test_convert_to_dict_of_lists_missing_key_raises_error() -> None:
    """Test convert_to_dict_of_lists raises when a later mapping is
    missing a key present in the first mapping."""
    with pytest.raises(ValueError, match="All the mappings must have the same keys"):
        convert_to_dict_of_lists([{"key1": 1, "key2": 10}, {"key1": 2}])


def test_convert_to_dict_of_lists_extra_key_raises_error() -> None:
    """Test convert_to_dict_of_lists raises when a later mapping has a
    key not present in the first mapping."""
    with pytest.raises(ValueError, match="All the mappings must have the same keys"):
        convert_to_dict_of_lists([{"key1": 1}, {"key1": 2, "key2": 20}])


def test_convert_to_dict_of_lists_accepts_a_generator() -> None:
    assert convert_to_dict_of_lists({"a": i, "b": i * 2} for i in range(3)) == {
        "a": [0, 1, 2],
        "b": [0, 2, 4],
    }


def test_convert_to_dict_of_lists_accepts_an_iterator() -> None:
    assert convert_to_dict_of_lists(iter([{"a": 1}, {"a": 2}])) == {"a": [1, 2]}


def test_convert_to_dict_of_lists_empty_generator() -> None:
    assert convert_to_dict_of_lists(x for x in []) == {}


def test_convert_to_dict_of_lists_generator_with_different_keys_raises() -> None:
    with pytest.raises(ValueError, match="same keys"):
        convert_to_dict_of_lists(d for d in [{"a": 1}, {"b": 2}])


def test_convert_to_dict_of_lists_keeps_the_order_of_the_first_mapping() -> None:
    assert list(convert_to_dict_of_lists([{"b": 1, "a": 2}, {"a": 3, "b": 4}])) == ["b", "a"]


##############################################
#     Tests for convert_to_list_of_dicts     #
##############################################


def test_convert_to_list_of_dicts_empty_dict() -> None:
    """Test convert_to_list_of_dicts with empty dict returns empty
    list."""
    assert convert_to_list_of_dicts({}) == []


def test_convert_to_list_of_dicts_empty_list() -> None:
    """Test convert_to_list_of_dicts with empty lists for all keys."""
    assert convert_to_list_of_dicts({"key1": [], "key2": []}) == []


def test_convert_to_list_of_dicts() -> None:
    """Test convert_to_list_of_dicts with standard dict of lists."""
    assert convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20, 30]}) == [
        {"key1": 1, "key2": 10},
        {"key1": 2, "key2": 20},
        {"key1": 3, "key2": 30},
    ]


def test_convert_to_list_of_dicts_single_value() -> None:
    """Test convert_to_list_of_dicts with single-value lists."""
    assert convert_to_list_of_dicts({"key1": [1], "key2": [2]}) == [{"key1": 1, "key2": 2}]


def test_convert_to_list_of_dicts_different_types() -> None:
    """Test convert_to_list_of_dicts with different value types."""
    assert convert_to_list_of_dicts({"name": ["Alice", "Bob"], "age": [30, 25]}) == [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]


def test_convert_to_list_of_dicts_mismatched_length_raises_error() -> None:
    """Test convert_to_list_of_dicts raises when sequences have
    different lengths."""
    with pytest.raises(ValueError, match="All the sequences must have the same length"):
        convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20]})


@pytest.mark.parametrize("value", ["ab", b"ab", bytearray(b"ab")])
def test_convert_to_list_of_dicts_string_value_raises(value: object) -> None:
    with pytest.raises(TypeError, match="must be a sequence of items"):
        convert_to_list_of_dicts({"a": value, "b": [1, 2]})  # ty: ignore[invalid-argument-type]


def test_convert_to_list_of_dicts_set_value_raises() -> None:
    with pytest.raises(TypeError, match="must be a sequence of items, not a set"):
        convert_to_list_of_dicts({"a": {1, 2}})  # ty: ignore[invalid-argument-type]


def test_convert_to_list_of_dicts_frozenset_value_raises() -> None:
    with pytest.raises(TypeError, match="not a frozenset"):
        convert_to_list_of_dicts({"a": frozenset({1})})  # ty: ignore[invalid-argument-type]


def test_convert_to_list_of_dicts_generator_value_raises_a_type_error_with_the_type() -> None:
    with pytest.raises(TypeError, match="not a generator"):
        convert_to_list_of_dicts({"a": (i for i in range(2))})  # ty: ignore[invalid-argument-type]


def test_convert_to_list_of_dicts_iterator_value_raises() -> None:
    with pytest.raises(TypeError, match="not a list_iterator"):
        convert_to_list_of_dicts({"a": iter([1])})  # ty: ignore[invalid-argument-type]


def test_convert_to_list_of_dicts_names_the_key() -> None:
    with pytest.raises(TypeError, match="value of 'b'"):
        convert_to_list_of_dicts({"a": [1], "b": {1}})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("value", [(1, 2), range(2), deque([1, 2])])
def test_convert_to_list_of_dicts_accepts_a_sequence(value: object) -> None:
    assert convert_to_list_of_dicts({"a": value}) == [{"a": value[0]}, {"a": value[1]}]  # ty: ignore[invalid-argument-type, not-subscriptable]
