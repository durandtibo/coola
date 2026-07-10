from __future__ import annotations

import pytest

from coola.nested import (
    flatten_mapping,
    get_first_value,
    merge_mappings,
    remove_keys_starting_with,
)

#####################################
#     Tests for flatten_mapping     #
#####################################


def test_flatten_mapping_empty_mapping() -> None:
    assert flatten_mapping({}) == {}


def test_flatten_mapping_empty_inner_mappings() -> None:
    assert flatten_mapping({"module1": {}, "module2": {}}) == {}


def test_flatten_mapping_single_outer_key() -> None:
    assert flatten_mapping({"module": {"a": 1, "b": 2}}) == {"a": 1, "b": 2}


def test_flatten_mapping_no_duplicate_keys() -> None:
    assert flatten_mapping({"module1": {"a": 1}, "module2": {"b": 2}}) == {"a": 1, "b": 2}


def test_flatten_mapping_incorrect_on_duplicate() -> None:
    with pytest.raises(ValueError, match="Incorrect on_duplicate value"):
        flatten_mapping({"module": {"a": 1}}, on_duplicate="incorrect")


def test_flatten_mapping_same_value_duplicate_not_triggered() -> None:
    # Duplicate inner keys with equal values should not trigger on_duplicate,
    # even with the default "raise" strategy.
    assert flatten_mapping(
        {"module1": {"a": 1, "b": 2, "c": 3}, "module2": {"a": 1, "b": 2, "d": 4}}
    ) == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_flatten_mapping_same_value_multiple_outer_keys() -> None:
    assert flatten_mapping({"module1": {"a": 1}, "module2": {"a": 1}, "module3": {"a": 1}}) == {
        "a": 1
    }


@pytest.mark.parametrize(
    ("mapping", "on_duplicate", "expected"),
    [
        pytest.param(
            {"module1": {"a": 1, "b": 2}, "module2": {"b": 3, "c": 4}},
            "first",
            {"a": 1, "b": 2, "c": 4},
            id="first-two-outer-keys",
        ),
        pytest.param(
            {"module1": {"a": 1, "b": 2}, "module2": {"b": 3, "c": 4}},
            "last",
            {"a": 1, "b": 3, "c": 4},
            id="last-two-outer-keys",
        ),
        pytest.param(
            {"module1": {"a": 1, "b": 2}, "module2": {"b": 3, "c": 4}},
            "prefix",
            {"a": 1, "module1.b": 2, "module2.b": 3, "c": 4},
            id="prefix-two-outer-keys",
        ),
        pytest.param(
            {"module1": {"a": 1}, "module2": {"a": 2}, "module3": {"a": 3}},
            "first",
            {"a": 1},
            id="first-multiple-occurrences",
        ),
        pytest.param(
            {"module1": {"a": 1}, "module2": {"a": 2}, "module3": {"a": 3}},
            "last",
            {"a": 3},
            id="last-multiple-occurrences",
        ),
        pytest.param(
            {"module1": {"a": 1}, "module2": {"a": 2}, "module3": {"a": 3}},
            "prefix",
            {"module1.a": 1, "module2.a": 2, "module3.a": 3},
            id="prefix-multiple-occurrences",
        ),
        pytest.param(
            {"module1": {"a": 1, "b": 2}, "module2": {"c": 3}, "module3": {"a": 4, "d": 5}},
            "prefix",
            {"module1.a": 1, "b": 2, "c": 3, "module3.a": 4, "d": 5},
            id="prefix-non-adjacent-duplicate",
        ),
        pytest.param(
            # Same value first, then a real conflict: only the genuinely
            # conflicting occurrences should be prefixed.
            {"module1": {"a": 1}, "module2": {"a": 1}, "module3": {"a": 2}},
            "prefix",
            {"module1.a": 1, "module3.a": 2},
            id="prefix-repeated-then-conflicting",
        ),
        pytest.param(
            # Mix of equal-value duplicates and conflicting duplicates.
            {"module1": {"a": 1, "b": 2}, "module2": {"a": 1, "b": 3}},
            "last",
            {"a": 1, "b": 3},
            id="last-mixed-equal-and-conflicting",
        ),
    ],
)
def test_flatten_mapping(mapping: dict, on_duplicate: str, expected: dict) -> None:
    assert flatten_mapping(mapping, on_duplicate=on_duplicate) == expected


def test_flatten_mapping_default_on_duplicate_is_raise() -> None:
    with pytest.raises(KeyError, match="Duplicate key found"):
        flatten_mapping({"module1": {"a": 1}, "module2": {"a": 2}})


def test_flatten_mapping_raise_on_duplicate() -> None:
    with pytest.raises(KeyError, match="Duplicate key found"):
        flatten_mapping({"module1": {"a": 1, "b": 2}, "module2": {"b": 3}}, on_duplicate="raise")


def test_flatten_mapping_raise_no_duplicate() -> None:
    assert flatten_mapping({"module1": {"a": 1}, "module2": {"b": 2}}, on_duplicate="raise") == {
        "a": 1,
        "b": 2,
    }


def test_flatten_mapping_raise_same_value_no_error() -> None:
    assert flatten_mapping({"module1": {"a": 1}, "module2": {"a": 1}}, on_duplicate="raise") == {
        "a": 1
    }


def test_flatten_mapping_non_string_inner_keys() -> None:
    assert flatten_mapping({"module1": {1: "a"}, "module2": {(2, 3): "b"}}) == {
        1: "a",
        (2, 3): "b",
    }


@pytest.mark.parametrize("separator", [".", "/", "@", "[SEP]"])
def test_flatten_mapping_separator(separator: str) -> None:
    assert flatten_mapping(
        {"module1": {"a": 1}, "module2": {"a": 2}},
        on_duplicate="prefix",
        separator=separator,
    ) == {f"module1{separator}a": 1, f"module2{separator}a": 2}


def test_flatten_mapping_always_prefix_empty_mapping() -> None:
    assert flatten_mapping({}, always_prefix=True) == {}


def test_flatten_mapping_always_prefix_no_duplicates() -> None:
    assert flatten_mapping({"module1": {"a": 1}, "module2": {"b": 2}}, always_prefix=True) == {
        "module1.a": 1,
        "module2.b": 2,
    }


def test_flatten_mapping_always_prefix_with_duplicates() -> None:
    # Every key is prefixed, so a "duplicate" inner key across outer
    # mappings never collides and on_duplicate is never triggered.
    assert flatten_mapping(
        {"module1": {"a": 1}, "module2": {"a": 2}}, always_prefix=True, on_duplicate="raise"
    ) == {"module1.a": 1, "module2.a": 2}


def test_flatten_mapping_always_prefix_same_value() -> None:
    assert flatten_mapping({"module1": {"a": 1}, "module2": {"a": 1}}, always_prefix=True) == {
        "module1.a": 1,
        "module2.a": 1,
    }


@pytest.mark.parametrize("separator", [".", "/", "@", "[SEP]"])
def test_flatten_mapping_always_prefix_separator(separator: str) -> None:
    assert flatten_mapping(
        {"module1": {"a": 1}, "module2": {"b": 2}},
        always_prefix=True,
        separator=separator,
    ) == {f"module1{separator}a": 1, f"module2{separator}b": 2}


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


####################################
#     Tests for merge_mappings     #
####################################


def test_merge_mappings_empty_list() -> None:
    assert merge_mappings([]) == {}


def test_merge_mappings_empty_mappings() -> None:
    assert merge_mappings([{}, {}]) == {}


def test_merge_mappings_single_mapping() -> None:
    assert merge_mappings([{"a": 1, "b": 2}]) == {"a": 1, "b": 2}


def test_merge_mappings_no_duplicate_keys() -> None:
    assert merge_mappings([{"a": 1}, {"b": 2}, {"c": 3}]) == {"a": 1, "b": 2, "c": 3}


def test_merge_mappings_incorrect_on_duplicate() -> None:
    with pytest.raises(ValueError, match="Incorrect on_duplicate value"):
        merge_mappings([{"a": 1}], on_duplicate="incorrect")


def test_merge_mappings_same_value_duplicate_not_triggered() -> None:
    # Duplicate keys with equal values should not trigger on_duplicate,
    # even with the default "raise" strategy.
    assert merge_mappings([{"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "d": 4}]) == {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }


def test_merge_mappings_same_value_multiple_mappings() -> None:
    assert merge_mappings([{"a": 1}, {"a": 1}, {"a": 1}]) == {"a": 1}


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
        pytest.param(
            # Same value first, then a real conflict: suffix numbering
            # should only count the genuine conflict.
            [{"a": 1}, {"a": 1}, {"a": 2}],
            "suffix",
            {"a": 1, "a_1": 2},
            id="suffix-repeated-then-conflicting",
        ),
        pytest.param(
            # Mix of equal-value duplicates and conflicting duplicates.
            [{"a": 1, "b": 2}, {"a": 1, "b": 3}],
            "last",
            {"a": 1, "b": 3},
            id="last-mixed-equal-and-conflicting",
        ),
    ],
)
def test_merge_mappings(mappings: list[dict], on_duplicate: str, expected: dict) -> None:
    assert merge_mappings(mappings, on_duplicate=on_duplicate) == expected


def test_merge_mappings_default_on_duplicate_is_raise() -> None:
    with pytest.raises(KeyError, match="Duplicate key found"):
        merge_mappings([{"a": 1}, {"a": 2}])


def test_merge_mappings_raise_on_duplicate() -> None:
    with pytest.raises(KeyError, match="Duplicate key found"):
        merge_mappings([{"a": 1, "b": 2}, {"b": 3}], on_duplicate="raise")


def test_merge_mappings_raise_no_duplicate() -> None:
    assert merge_mappings([{"a": 1}, {"b": 2}], on_duplicate="raise") == {"a": 1, "b": 2}


def test_merge_mappings_raise_same_value_no_error() -> None:
    assert merge_mappings([{"a": 1}, {"a": 1}], on_duplicate="raise") == {"a": 1}


def test_merge_mappings_iterable_generator() -> None:
    # merge_mappings should accept any iterable, not just a list.
    gen = ({"a": i} for i in range(3))
    assert merge_mappings(gen, on_duplicate="last") == {"a": 2}


def test_merge_mappings_non_string_keys() -> None:
    assert merge_mappings([{1: "a"}, {(2, 3): "b"}]) == {1: "a", (2, 3): "b"}


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
