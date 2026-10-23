from __future__ import annotations

from collections import OrderedDict
from typing import NamedTuple, Self

import pytest

from coola.nested import (
    DuplicateKeyError,
    add_prefix_suffix_to_keys,
    flatten_mapping,
    get_first_value,
    merge_mappings,
    remove_keys_containing,
    remove_keys_if,
    remove_keys_starting_with,
)


class _PassesValidationOnly(str):
    """A string that equals its value only for the first comparison.

    It lets a test get past the ``on_duplicate`` validation (a set
    membership check) while matching none of the strategy branches
    afterwards, to exercise the defensive ``assert_never`` branches.
    """

    def __new__(cls, value: str) -> Self:
        obj = super().__new__(cls, value)
        obj._remaining = 1
        return obj

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        if self._remaining > 0:
            self._remaining -= 1
            return str.__eq__(self, other)
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


###############################################
#     Tests for add_prefix_suffix_to_keys     #
###############################################


def test_add_prefix_suffix_to_keys_empty_mapping() -> None:
    assert add_prefix_suffix_to_keys({}, prefix="prefix_") == {}


def test_add_prefix_suffix_to_keys_prefix_only() -> None:
    assert add_prefix_suffix_to_keys({"key1": 1, "key2": 2}, prefix="prefix_") == {
        "prefix_key1": 1,
        "prefix_key2": 2,
    }


def test_add_prefix_suffix_to_keys_suffix_only() -> None:
    assert add_prefix_suffix_to_keys({"key1": 1, "key2": 2}, suffix="_suffix") == {
        "key1_suffix": 1,
        "key2_suffix": 2,
    }


def test_add_prefix_suffix_to_keys_prefix_and_suffix() -> None:
    assert add_prefix_suffix_to_keys({"key1": 1, "key2": 2}, prefix="pre_", suffix="_suf") == {
        "pre_key1_suf": 1,
        "pre_key2_suf": 2,
    }


def test_add_prefix_suffix_to_keys_no_prefix_or_suffix() -> None:
    assert add_prefix_suffix_to_keys({"key1": 1, "key2": 2}) == {"key1": 1, "key2": 2}


def test_add_prefix_suffix_to_keys_non_string_key() -> None:
    assert add_prefix_suffix_to_keys({1: "a", "key": "b"}, prefix="prefix_") == {
        1: "a",
        "prefix_key": "b",
    }


def test_add_prefix_suffix_to_keys_not_recursive_by_default() -> None:
    assert add_prefix_suffix_to_keys({"key1": 1, "key2": {"key3": 3}}, prefix="prefix_") == {
        "prefix_key1": 1,
        "prefix_key2": {"key3": 3},
    }


def test_add_prefix_suffix_to_keys_recursive() -> None:
    assert add_prefix_suffix_to_keys(
        {"key1": 1, "key2": {"key3": 3, "key4": {"key5": 5}}},
        prefix="prefix_",
        recursive=True,
    ) == {
        "prefix_key1": 1,
        "prefix_key2": {"prefix_key3": 3, "prefix_key4": {"prefix_key5": 5}},
    }


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


def test_flatten_mapping_unhandled_on_duplicate() -> None:
    with pytest.raises(AssertionError):
        flatten_mapping(
            {"x": {"a": 1}, "y": {"a": 2}},
            on_duplicate=_PassesValidationOnly("prefix"),  # type: ignore[arg-type]
        )


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


def test_merge_mappings_unhandled_on_duplicate() -> None:
    with pytest.raises(AssertionError):
        merge_mappings([{"a": 1}, {"a": 2}], on_duplicate=_PassesValidationOnly("first"))  # type: ignore[arg-type]


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


def test_merge_mappings_flatten_mapping_first_occurrence_asymmetry() -> None:
    """Test and document the documented asymmetry between
    ``merge_mappings(on_duplicate='suffix')``, which leaves the first
    occurrence of a conflicting key under the plain key, and
    ``flatten_mapping(on_duplicate='prefix')``, which renames every
    occurrence, including the first, once a real conflict is found."""
    merged = merge_mappings([{"a": 1}, {"a": 2}], on_duplicate="suffix")
    assert merged == {"a": 1, "a_1": 2}  # first occurrence keeps the plain key

    flattened = flatten_mapping({"m1": {"a": 1}, "m2": {"a": 2}}, on_duplicate="prefix")
    assert flattened == {"m1.a": 1, "m2.a": 2}  # first occurrence is renamed too


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


####################################
#     Tests for remove_keys_if     #
####################################


def test_remove_keys_if_empty() -> None:
    assert remove_keys_if({}, lambda key: key == "key") == {}


def test_remove_keys_if() -> None:
    assert remove_keys_if(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6},
        lambda key: isinstance(key, str) and "key" in key,
    ) == {
        "abc": 3,
        1: 5,
        (2, 3): 6,
    }


def test_remove_keys_if_non_string_predicate() -> None:
    assert remove_keys_if(
        {"a": 1, "b": 2, 1: 3, 2: 4}, lambda key: isinstance(key, int) and key % 2 == 0
    ) == {"a": 1, "b": 2, 1: 3}


def test_remove_keys_if_nested_mapping() -> None:
    assert remove_keys_if(
        {"abc": {"key": 1, "abc": 2}, "key": {"abc": 3}},
        lambda key: isinstance(key, str) and "key" in key,
    ) == {"abc": {"abc": 2}}


def test_remove_keys_if_nested_list() -> None:
    assert remove_keys_if(
        {"list": [{"key": 1, "abc": 2}, {"key": 3}], "abc": 4},
        lambda key: isinstance(key, str) and "key" in key,
    ) == {"list": [{"abc": 2}, {}], "abc": 4}


def test_remove_keys_if_nested_tuple() -> None:
    assert remove_keys_if(
        {"tuple": ({"key": 1, "abc": 2},)}, lambda key: isinstance(key, str) and "key" in key
    ) == {"tuple": ({"abc": 2},)}


def test_remove_keys_if_nested_set() -> None:
    assert remove_keys_if(
        {"set": {1, 2, 3}, "key": 4}, lambda key: isinstance(key, str) and "key" in key
    ) == {"set": {1, 2, 3}}


def test_remove_keys_if_preserves_mapping_type() -> None:
    out = remove_keys_if(
        OrderedDict(key=1, abc=2), lambda key: isinstance(key, str) and "key" in key
    )
    assert out == {"abc": 2}
    assert isinstance(out, OrderedDict)


def test_remove_keys_if_preserves_namedtuple() -> None:
    class Point(NamedTuple):
        x: int
        y: dict

    out = remove_keys_if(
        {"point": Point(1, {"key": 2, "abc": 3})},
        lambda key: isinstance(key, str) and "key" in key,
    )
    assert out == {"point": Point(1, {"abc": 3})}
    assert isinstance(out["point"], Point)


############################################
#     Tests for remove_keys_containing     #
############################################


def test_remove_keys_containing_empty() -> None:
    assert remove_keys_containing({}, "key") == {}


def test_remove_keys_containing() -> None:
    assert remove_keys_containing(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "key"
    ) == {
        "abc": 3,
        1: 5,
        (2, 3): 6,
    }


def test_remove_keys_containing_another_substring() -> None:
    assert remove_keys_containing(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "xyz"
    ) == {
        "key": 1,
        "key.abc": 2,
        "abc": 3,
        "abc.key": 4,
        1: 5,
        (2, 3): 6,
    }


def test_remove_keys_containing_nested_mapping() -> None:
    assert remove_keys_containing({"abc": {"key": 1, "abc": 2}, "key": {"abc": 3}}, "key") == {
        "abc": {"abc": 2}
    }


def test_remove_keys_containing_nested_list() -> None:
    assert remove_keys_containing(
        {"list": [{"key": 1, "abc": 2}, {"key": 3}], "abc": 4}, "key"
    ) == {"list": [{"abc": 2}, {}], "abc": 4}


def test_remove_keys_containing_nested_tuple() -> None:
    assert remove_keys_containing({"tuple": ({"key": 1, "abc": 2},)}, "key") == {
        "tuple": ({"abc": 2},)
    }


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


def test_remove_keys_starting_with_nested_mapping() -> None:
    assert remove_keys_starting_with({"abc": {"key": 1, "abc": 2}, "key": {"abc": 3}}, "key") == {
        "abc": {"abc": 2}
    }


def test_duplicate_key_error_is_key_error_and_value_error() -> None:
    assert issubclass(DuplicateKeyError, KeyError)
    assert issubclass(DuplicateKeyError, ValueError)


def test_duplicate_key_error_str_is_not_quoted() -> None:
    assert str(DuplicateKeyError("Duplicate key found: 'a'")) == "Duplicate key found: 'a'"


def test_merge_mappings_raises_duplicate_key_error() -> None:
    with pytest.raises(DuplicateKeyError, match="Duplicate key found"):
        merge_mappings([{"a": 1}, {"a": 2}])


def test_flatten_mapping_raises_duplicate_key_error() -> None:
    with pytest.raises(ValueError, match="Duplicate key found"):
        flatten_mapping({"x": {"a": 1}, "y": {"a": 2}})


def test_merge_mappings_suffix_does_not_overwrite_existing_key() -> None:
    # "a_1" already exists, so the duplicate "a" must not overwrite it.
    result = merge_mappings([{"a": 1, "a_1": 10}, {"a": 2}], on_duplicate="suffix")
    assert result == {"a": 1, "a_1": 10, "a_2": 2}


def test_merge_mappings_suffix_key_colliding_with_later_key_raises() -> None:
    # The suffixed duplicate "a_1" collides with the original key "a_1" of the third mapping.
    with pytest.raises(DuplicateKeyError, match="collides"):
        merge_mappings([{"a": 1}, {"a": 2}, {"a_1": 3}], on_duplicate="suffix")


def test_remove_keys_if_uses_the_predicate_of_each_call() -> None:
    data = {"a": 1, "b": {"a": 2, "b": 3}}
    assert remove_keys_if(data, lambda key: key == "a") == {"b": {"b": 3}}
    assert remove_keys_if(data, lambda key: key == "b") == {"a": 1}
    assert remove_keys_if(data, lambda key: False) == data  # noqa: ARG005


def test_remove_keys_if_values_equal_to_the_removed_key_are_kept() -> None:
    assert remove_keys_if({"x": "a", "a": 1, "y": ["a"]}, lambda key: key == "a") == {
        "x": "a",
        "y": ["a"],
    }


def test_merge_mappings_suffix_compares_with_all_kept_values() -> None:
    assert merge_mappings([{"a": 1}, {"a": 2}, {"a": 2}], on_duplicate="suffix") == {
        "a": 1,
        "a_1": 2,
    }


def test_merge_mappings_suffix_compares_with_first_value() -> None:
    assert merge_mappings([{"a": 1}, {"a": 2}, {"a": 1}], on_duplicate="suffix") == {
        "a": 1,
        "a_1": 2,
    }


def test_merge_mappings_last_keeps_the_last_value() -> None:
    assert merge_mappings([{"a": 1}, {"a": 2}, {"a": 1}], on_duplicate="last") == {"a": 1}


def test_merge_mappings_nan_are_different_by_default() -> None:
    with pytest.raises(DuplicateKeyError, match="Duplicate key"):
        merge_mappings([{"a": float("nan")}, {"a": float("nan")}])


def test_merge_mappings_equal_nan() -> None:
    result = merge_mappings([{"a": float("nan")}, {"a": float("nan")}], equal_nan=True)
    assert list(result) == ["a"]


def test_flatten_mapping_nan_are_different_by_default() -> None:
    with pytest.raises(DuplicateKeyError, match="Duplicate key"):
        flatten_mapping({"x": {"a": float("nan")}, "y": {"a": float("nan")}})


def test_flatten_mapping_equal_nan() -> None:
    result = flatten_mapping({"x": {"a": float("nan")}, "y": {"a": float("nan")}}, equal_nan=True)
    assert list(result) == ["a"]


def test_add_prefix_suffix_to_keys_recursive_list_of_dicts() -> None:
    assert add_prefix_suffix_to_keys(
        {"a": [{"b": 1}, [{"c": 2}], 3]}, prefix="p_", recursive=True
    ) == {"p_a": [{"p_b": 1}, [{"p_c": 2}], 3]}


def test_add_prefix_suffix_to_keys_recursive_tuple_of_dicts() -> None:
    assert add_prefix_suffix_to_keys({"a": ({"b": 1},)}, suffix="_s", recursive=True) == {
        "a_s": ({"b_s": 1},)
    }


def test_add_prefix_suffix_to_keys_recursive_keeps_strings() -> None:
    assert add_prefix_suffix_to_keys({"a": ["abc", b"x"]}, prefix="p_", recursive=True) == {
        "p_a": ["abc", b"x"]
    }


def test_add_prefix_suffix_to_keys_not_recursive_keeps_lists() -> None:
    assert add_prefix_suffix_to_keys({"a": [{"b": 1}]}, prefix="p_") == {"p_a": [{"b": 1}]}


def test_flatten_mapping_always_prefix_colliding_keys_raises() -> None:
    with pytest.raises(DuplicateKeyError, match="collides"):
        flatten_mapping({"a.b": {"c": 1}, "a": {"b.c": 2}}, always_prefix=True)


def test_flatten_mapping_prefix_colliding_with_plain_key_raises() -> None:
    with pytest.raises(DuplicateKeyError, match="collides"):
        flatten_mapping({"m1": {"b": 1}, "m2": {"b": 2}, "x": {"m1.b": 5}}, on_duplicate="prefix")


def test_flatten_mapping_prefix_colliding_with_prefixed_key_raises() -> None:
    with pytest.raises(DuplicateKeyError, match="collides"):
        flatten_mapping({"x": {"m1.b": 5}, "m1": {"b": 1}, "m2": {"b": 2}}, on_duplicate="prefix")
