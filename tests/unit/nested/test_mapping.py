from __future__ import annotations

from typing import Self

import pytest

from coola.nested import (
    DuplicateKeyError,
    flatten_mapping,
    merge_mappings,
)


class _PassesValidationOnly(str):
    """A string that equals its value only the first time they match.

    It lets a test get past the ``on_duplicate`` validation (a
    membership check) while matching none of the strategy branches
    afterwards, to exercise the defensive ``assert_never`` branches.
    """

    _remaining: int

    def __new__(cls, value: str) -> Self:
        obj = super().__new__(cls, value)
        obj._remaining = 1
        return obj

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        if self._remaining > 0 and str.__eq__(self, other):
            self._remaining -= 1
            return True
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


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
            on_duplicate=_PassesValidationOnly("prefix"),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )


def test_flatten_mapping_incorrect_on_duplicate() -> None:
    with pytest.raises(ValueError, match=r"^on_duplicate must be one of \(.*\), got 'incorrect'$"):
        flatten_mapping({"module": {"a": 1}}, on_duplicate="incorrect")  # ty: ignore[invalid-argument-type]


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
    assert flatten_mapping(mapping, on_duplicate=on_duplicate) == expected  # ty: ignore[invalid-argument-type]


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
    with pytest.raises(ValueError, match=r"^on_duplicate must be one of \(.*\), got 'incorrect'$"):
        merge_mappings([{"a": 1}], on_duplicate="incorrect")  # ty: ignore[invalid-argument-type]


def test_merge_mappings_unhandled_on_duplicate() -> None:
    with pytest.raises(AssertionError):
        merge_mappings([{"a": 1}, {"a": 2}], on_duplicate=_PassesValidationOnly("first"))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


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
    assert merge_mappings(mappings, on_duplicate=on_duplicate) == expected  # ty: ignore[invalid-argument-type]


def test_merge_mappings_first_occurrence_keeps_the_plain_key() -> None:
    """Test and document the documented asymmetry between
    ``merge_mappings(on_duplicate='suffix')``, which leaves the first
    occurrence of a conflicting key under the plain key, and
    ``flatten_mapping(on_duplicate='prefix')``, which renames every
    occurrence, including the first, once a real conflict is found."""
    merged = merge_mappings([{"a": 1}, {"a": 2}], on_duplicate="suffix")
    assert merged == {"a": 1, "a_1": 2}  # first occurrence keeps the plain key


def test_flatten_mapping_first_occurrence_is_renamed() -> None:
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


def test_duplicate_key_error_is_key_error_and_value_error_key_error() -> None:
    assert issubclass(DuplicateKeyError, KeyError)


def test_duplicate_key_error_is_key_error_and_value_error_value_error() -> None:
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


def test_merge_mappings_suffix_input_key_equal_to_a_generated_key_is_kept() -> None:
    # The suffixed duplicate "a_1" would collide with the input key "a_1" of the third mapping.
    assert merge_mappings([{"a": 1}, {"a": 2}, {"a_1": 3}], on_duplicate="suffix") == {
        "a": 1,
        "a_2": 2,
        "a_1": 3,
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


def test_flatten_mapping_always_prefix_colliding_keys_raises() -> None:
    with pytest.raises(DuplicateKeyError, match="collides"):
        flatten_mapping({"a.b": {"c": 1}, "a": {"b.c": 2}}, always_prefix=True)


def test_flatten_mapping_prefix_colliding_with_plain_key_raises() -> None:
    with pytest.raises(DuplicateKeyError, match="collides"):
        flatten_mapping({"m1": {"b": 1}, "m2": {"b": 2}, "x": {"m1.b": 5}}, on_duplicate="prefix")


def test_flatten_mapping_prefix_colliding_with_prefixed_key_raises() -> None:
    with pytest.raises(DuplicateKeyError, match="collides"):
        flatten_mapping({"x": {"m1.b": 5}, "m1": {"b": 1}, "m2": {"b": 2}}, on_duplicate="prefix")


def test_flatten_mapping_value_that_is_not_a_mapping_raises() -> None:
    with pytest.raises(TypeError, match=r"value of the key 'a' must be an instance of"):
        flatten_mapping({"a": 1})  # ty: ignore[invalid-argument-type]


def test_flatten_mapping_value_that_is_not_a_mapping_names_the_type() -> None:
    with pytest.raises(TypeError, match=r"got list"):
        flatten_mapping({"a": {"x": 1}, "b": [1, 2]})  # ty: ignore[invalid-argument-type]


def test_flatten_mapping_value_that_is_not_a_mapping_raises_with_always_prefix() -> None:
    with pytest.raises(TypeError, match="must be an instance of"):
        flatten_mapping({"a": None}, always_prefix=True)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "mappings",
    [
        [{"a": 1}, {"a": 2}, {"a_1": 3}],
        [{"a": 1}, {"a_1": 3}, {"a": 2}],
        [{"a_1": 3}, {"a": 1}, {"a": 2}],
    ],
)
def test_merge_mappings_suffix_has_the_same_entries_for_every_order(mappings: list) -> None:
    result = merge_mappings(mappings, on_duplicate="suffix")
    assert sorted(result.items()) == [("a", 1), ("a_1", 3), ("a_2", 2)]


def test_merge_mappings_suffix_input_key_first_whole_output() -> None:
    assert merge_mappings([{"a_1": 3}, {"a": 1}, {"a": 2}], on_duplicate="suffix") == {
        "a_1": 3,
        "a": 1,
        "a_2": 2,
    }


def test_merge_mappings_suffix_skips_several_input_keys() -> None:
    mappings = [{"a": 1}, {"a": 2}, {"a": 3}, {"a_1": 4, "a_2": 5}]
    assert merge_mappings(mappings, on_duplicate="suffix") == {
        "a": 1,
        "a_3": 2,
        "a_4": 3,
        "a_1": 4,
        "a_2": 5,
    }


def test_merge_mappings_accepts_a_generator() -> None:
    assert merge_mappings(({"a": i} for i in range(1)), on_duplicate="suffix") == {"a": 0}


def test_merge_mappings_generator_with_suffix_sees_the_later_keys() -> None:
    mappings = ({k: v} for k, v in [("a", 1), ("a", 2), ("a_1", 3)])
    assert merge_mappings(mappings, on_duplicate="suffix") == {"a": 1, "a_2": 2, "a_1": 3}


def test_merge_mappings_keeps_the_order_of_the_input() -> None:
    assert list(merge_mappings([{"c": 1, "a": 2}, {"b": 3}])) == ["c", "a", "b"]


def test_flatten_mapping_prefix_keeps_the_order_of_the_input() -> None:
    metrics = {"train": {"loss": 1}, "val": {"acc": 2, "loss": 3}}
    assert list(flatten_mapping(metrics, on_duplicate="prefix")) == [
        "train.loss",
        "acc",
        "val.loss",
    ]


def test_flatten_mapping_prefix_whole_output_of_the_guide_example() -> None:
    metrics = {"train": {"loss": 1, "acc": 2}, "val": {"loss": 3}}
    assert flatten_mapping(metrics, on_duplicate="prefix") == {
        "train.loss": 1,
        "acc": 2,
        "val.loss": 3,
    }


def test_flatten_mapping_prefix_renamed_key_collides_with_an_existing_key() -> None:
    mapping = {"a": {"x": 1}, "c": {"a.x": 5}, "b": {"x": 2}}
    with pytest.raises(DuplicateKeyError, match="collides"):
        flatten_mapping(mapping, on_duplicate="prefix")


def test_flatten_mapping_last_keeps_the_position_of_the_first_occurrence() -> None:
    mapping = {"a": {"x": 1, "y": 2}, "b": {"x": 3}}
    assert list(flatten_mapping(mapping, on_duplicate="last").items()) == [("x", 3), ("y", 2)]


def test_flatten_mapping_options_are_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional"):
        flatten_mapping({"a": {"b": 1}}, "raise")  # ty: ignore[too-many-positional-arguments]


def test_merge_mappings_options_are_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional"):
        merge_mappings([{"a": 1}], "raise")  # ty: ignore[too-many-positional-arguments]
