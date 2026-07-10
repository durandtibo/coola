from __future__ import annotations


import pytest

from coola.nested import (
    get_first_value,
    merge_list_of_mappings,
    remove_keys_starting_with,
)

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


############################################
#     Tests for merge_list_of_mappings     #
############################################


def test_merge_list_of_mappings_empty_list() -> None:
    assert merge_list_of_mappings([]) == {}


def test_merge_list_of_mappings_empty_mappings() -> None:
    assert merge_list_of_mappings([{}, {}]) == {}


def test_merge_list_of_mappings_single_mapping() -> None:
    assert merge_list_of_mappings([{"a": 1, "b": 2}]) == {"a": 1, "b": 2}


def test_merge_list_of_mappings_no_duplicate_keys() -> None:
    assert merge_list_of_mappings([{"a": 1}, {"b": 2}, {"c": 3}]) == {"a": 1, "b": 2, "c": 3}


def test_merge_list_of_mappings_incorrect_on_duplicate() -> None:
    with pytest.raises(ValueError, match="Incorrect on_duplicate value"):
        merge_list_of_mappings([{"a": 1}], on_duplicate="incorrect")


@pytest.mark.parametrize(
    ("mappings", "on_duplicate", "expected"),
    [
        pytest.param(
            [{"a": 1, "b": 2}, {"b": 3, "c": 4}],
            "first",
            {"a": 1, "b": 2, "c": 4},
            id="first-two-mappings",
        ),
        pytest.param(
            [{"a": 1, "b": 2}, {"b": 3, "c": 4}],
            "last",
            {"a": 1, "b": 3, "c": 4},
            id="last-two-mappings",
        ),
        pytest.param(
            [{"a": 1, "b": 2}, {"b": 3, "c": 4}],
            "suffix",
            {"a": 1, "b": 2, "b_1": 3, "c": 4},
            id="suffix-two-mappings",
        ),
        pytest.param(
            [{"a": 1}, {"a": 2}, {"a": 3}],
            "first",
            {"a": 1},
            id="first-multiple-occurrences",
        ),
        pytest.param(
            [{"a": 1}, {"a": 2}, {"a": 3}],
            "last",
            {"a": 3},
            id="last-multiple-occurrences",
        ),
        pytest.param(
            [{"a": 1}, {"a": 2}, {"a": 3}],
            "suffix",
            {"a": 1, "a_1": 2, "a_2": 3},
            id="suffix-multiple-occurrences",
        ),
        pytest.param(
            [{"a": 1, "b": 2}, {"c": 3}, {"a": 4, "d": 5}],
            "suffix",
            {"a": 1, "b": 2, "c": 3, "a_1": 4, "d": 5},
            id="suffix-non-adjacent-duplicate",
        ),
    ],
)
def test_merge_list_of_mappings(mappings: list[dict], on_duplicate: str, expected: dict) -> None:
    assert merge_list_of_mappings(mappings, on_duplicate=on_duplicate) == expected


def test_merge_list_of_mappings_default_on_duplicate_is_raise() -> None:
    with pytest.raises(KeyError, match="Duplicate key found"):
        merge_list_of_mappings([{"a": 1}, {"a": 2}])


def test_merge_list_of_mappings_raise_on_duplicate() -> None:
    with pytest.raises(KeyError, match="Duplicate key found"):
        merge_list_of_mappings([{"a": 1, "b": 2}, {"b": 3}], on_duplicate="raise")


def test_merge_list_of_mappings_raise_no_duplicate() -> None:
    assert merge_list_of_mappings([{"a": 1}, {"b": 2}], on_duplicate="raise") == {"a": 1, "b": 2}


def test_merge_list_of_mappings_iterable_generator() -> None:
    # merge_list_of_mappings should accept any iterable, not just a list.
    gen = ({"a": i} for i in range(3))
    assert merge_list_of_mappings(gen, on_duplicate="last") == {"a": 2}


def test_merge_list_of_mappings_non_string_keys() -> None:
    assert merge_list_of_mappings([{1: "a"}, {(2, 3): "b"}]) == {1: "a", (2, 3): "b"}


###############################################
#     Tests for remove_keys_starting_with     #
###############################################


def test_remove_keys_starting_with_empty() -> None:
    assert remove_keys_starting_with({}, "key") == {}


def test_remove_keys_starting_with() -> None:
    assert remove_keys_starting_with(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "key"
    ) == {
        "abc": 3,
        "abc.key": 4,
        1: 5,
        (2, 3): 6,
    }


def test_remove_keys_starting_with_another_key() -> None:
    assert remove_keys_starting_with(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "key."
    ) == {
        "key": 1,
        "abc": 3,
        "abc.key": 4,
        1: 5,
        (2, 3): 6,
    }
