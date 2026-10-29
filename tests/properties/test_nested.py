r"""Property-based tests for the guarantees of the functions of
``coola.nested``."""

from __future__ import annotations

import contextlib
import copy
from typing import TYPE_CHECKING, Any, Literal

import pytest
from hypothesis import assume, given, settings, strategies as st

from coola.equality import objects_are_equal
from coola.nested import (
    DuplicateKeyError,
    add_prefix_suffix_to_keys,
    convert_to_jsonable,
    flatten_mapping,
    from_flat_dict,
    merge_mappings,
    remove_keys_if,
    to_flat_dict,
)
from tests.properties.strategies import (
    flat_leaves,
    int_mappings,
    keys,
    nested_dicts,
    rebuild,
    values,
)

if TYPE_CHECKING:
    import random

MergeMode = Literal["first", "last", "suffix"]
FlattenMode = Literal["first", "last", "prefix"]

separators = st.sampled_from([".", "/", "::", "->"])
strategies = st.sampled_from(["first", "last", "suffix"])


def leaves(data: Any) -> int:
    r"""Count the leaves of nested dictionaries (an empty dictionary is
    a leaf)."""
    if isinstance(data, dict) and data:
        return sum(leaves(value) for value in data.values())
    return 1


###########################
#     Flat dictionary     #
###########################


@given(data=nested_dicts)
def test_from_flat_dict_inverts_to_flat_dict(data: dict[str, Any]) -> None:
    assert from_flat_dict(to_flat_dict(data)) == data


lists_and_dicts = st.recursive(
    flat_leaves,
    lambda children: st.one_of(
        st.lists(children, min_size=1, max_size=3),
        st.dictionaries(keys, children, min_size=1, max_size=3),
    ),
    max_leaves=8,
)


@given(data=st.dictionaries(keys, lists_and_dicts, max_size=4))
def test_from_flat_dict_restores_the_lists(data: dict[str, Any]) -> None:
    assert from_flat_dict(to_flat_dict(data), restore_lists=True) == data


@given(data=st.dictionaries(keys, st.binary(max_size=4), max_size=4))
def test_to_flat_dict_keeps_bytes_as_leaves(data: dict[str, bytes]) -> None:
    assert to_flat_dict(data) == data


@given(data=nested_dicts, separator=separators)
def test_round_trip_with_any_separator(data: dict[str, Any], separator: str) -> None:
    flat = to_flat_dict(data, separator=separator)
    assert from_flat_dict(flat, separator=separator) == data


@given(data=nested_dicts)
def test_flat_dict_has_string_keys(data: dict[str, Any]) -> None:
    assert all(isinstance(key, str) for key in to_flat_dict(data))


@given(data=nested_dicts)
def test_flat_dict_has_no_nested_dict(data: dict[str, Any]) -> None:
    flat = to_flat_dict(data)
    assert all(not isinstance(value, dict) or value == {} for value in flat.values())


@given(data=nested_dicts)
def test_flat_dict_has_one_entry_for_each_leaf(data: dict[str, Any]) -> None:
    expected = sum(leaves(value) for value in data.values())
    assert len(to_flat_dict(data)) == expected


@given(data=nested_dicts)
def test_flat_dict_is_a_fixed_point(data: dict[str, Any]) -> None:
    flat = to_flat_dict(data)
    assert to_flat_dict(from_flat_dict(flat)) == flat


@given(data=nested_dicts, prefix=keys)
def test_prefix_is_added_to_all_the_keys(data: dict[str, Any], prefix: str) -> None:
    assume(data)
    flat = to_flat_dict(data, prefix=prefix)
    expected = {f"{prefix}.{key}": value for key, value in to_flat_dict(data).items()}
    assert flat == expected


@given(first=keys, second=keys, value=flat_leaves, other=flat_leaves)
def test_a_key_that_is_a_scalar_and_a_prefix_is_a_conflict(
    first: str, second: str, value: Any, other: Any
) -> None:
    with pytest.raises(ValueError, match="conflict"):
        from_flat_dict({first: value, f"{first}.{second}": other})


@given(first=keys, second=keys, value=flat_leaves, other=flat_leaves)
def test_a_key_that_is_a_scalar_and_a_prefix_is_a_conflict_in_any_order(
    first: str, second: str, value: Any, other: Any
) -> None:
    with pytest.raises(ValueError, match="conflict"):
        from_flat_dict({f"{first}.{second}": other, first: value})


@given(value=flat_leaves, other=flat_leaves)
def test_keys_that_have_the_same_string_collide(value: Any, other: Any) -> None:
    with pytest.raises(ValueError, match="collision"):
        to_flat_dict({1: value, "1": other})


@given(key=keys, value=flat_leaves)
def test_a_single_key_does_not_collide(key: str, value: Any) -> None:
    assert to_flat_dict({key: value}) == {key: value}


@given(data=nested_dicts)
def test_to_flat_dict_does_not_modify_the_input(data: dict[str, Any]) -> None:
    expected = copy.deepcopy(data)
    to_flat_dict(data)
    assert data == expected


@given(data=nested_dicts)
def test_from_flat_dict_does_not_modify_the_input(data: dict[str, Any]) -> None:
    flat = to_flat_dict(data)
    expected = copy.deepcopy(flat)
    from_flat_dict(flat)
    assert flat == expected


@given(data=nested_dicts, seed=st.randoms(use_true_random=False))
def test_from_flat_dict_does_not_depend_on_the_order_of_the_keys(
    data: dict[str, Any], seed: random.Random
) -> None:
    items = list(to_flat_dict(data).items())
    seed.shuffle(items)
    assert from_flat_dict(dict(items)) == data


@given(first=keys, second=keys, leaf=st.dictionaries(keys, flat_leaves, min_size=1, max_size=3))
def test_from_flat_dict_raises_for_a_dict_given_as_a_value(
    first: str, second: str, leaf: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="conflict"):
        from_flat_dict({first: leaf, f"{first}.{second}": 0})


@given(first=keys, second=keys, leaf=st.dictionaries(keys, flat_leaves, min_size=1, max_size=3))
def test_from_flat_dict_does_not_modify_a_dict_given_as_a_value(
    first: str, second: str, leaf: dict[str, Any]
) -> None:
    expected = copy.deepcopy(leaf)
    with contextlib.suppress(ValueError):
        from_flat_dict({first: leaf, f"{first}.{second}": 0})
    assert leaf == expected


@given(values=st.lists(flat_leaves, min_size=1, max_size=4), key=keys)
def test_to_flat_dict_flattens_a_tuple_like_a_list(values: list[Any], key: str) -> None:
    assert to_flat_dict({key: tuple(values)}) == to_flat_dict({key: values})


@given(value=flat_leaves, key=keys)
def test_to_flat_dict_accepts_a_container_shared_in_two_places(value: Any, key: str) -> None:
    shared = [value]
    assert to_flat_dict({"a": shared, "b": shared, key + "x": shared}) == {
        "a.0": value,
        "b.0": value,
        key + "x.0": value,
    }


############################
#     Merge the mappings   #
############################


@given(mapping=int_mappings, on_duplicate=strategies)
def test_merge_single_mapping_is_unchanged(
    mapping: dict[str, int], on_duplicate: MergeMode
) -> None:
    assert merge_mappings([mapping], on_duplicate=on_duplicate) == mapping


@given(mapping=int_mappings, on_duplicate=strategies)
def test_merge_is_idempotent(mapping: dict[str, int], on_duplicate: MergeMode) -> None:
    assert merge_mappings([mapping, mapping], on_duplicate=on_duplicate) == mapping


@given(first=int_mappings, second=int_mappings, on_duplicate=strategies)
def test_merge_of_a_merged_mapping_is_unchanged(
    first: dict[str, int], second: dict[str, int], on_duplicate: MergeMode
) -> None:
    merged = merge_mappings([first, second], on_duplicate=on_duplicate)
    assert merge_mappings([merged, merged], on_duplicate=on_duplicate) == merged


@pytest.mark.parametrize("on_duplicate", ["first", "last"])
@given(first=int_mappings, second=int_mappings)
def test_merge_keeps_every_key(
    on_duplicate: MergeMode, first: dict[str, int], second: dict[str, int]
) -> None:
    merged = merge_mappings([first, second], on_duplicate=on_duplicate)
    assert set(merged) == set(first) | set(second)


@given(first=int_mappings, second=int_mappings)
def test_merge_first_keeps_the_first_value(first: dict[str, int], second: dict[str, int]) -> None:
    merged = merge_mappings([first, second], on_duplicate="first")
    assert merged == {**second, **first}


@given(first=int_mappings, second=int_mappings)
def test_merge_last_keeps_the_last_value(first: dict[str, int], second: dict[str, int]) -> None:
    merged = merge_mappings([first, second], on_duplicate="last")
    assert merged == {**first, **second}


@given(first=int_mappings, second=int_mappings)
def test_merge_raise_raises_if_a_key_has_two_values(
    first: dict[str, int], second: dict[str, int]
) -> None:
    assume(first)
    key = next(iter(first))
    with pytest.raises(DuplicateKeyError):
        merge_mappings([first, {**second, key: first[key] + 1}], on_duplicate="raise")


@given(first=int_mappings, second=int_mappings)
def test_merge_raise_merges_if_no_key_has_two_values(
    first: dict[str, int], second: dict[str, int]
) -> None:
    assume(not any(key in second and second[key] != value for key, value in first.items()))
    merged = merge_mappings([first, second], on_duplicate="raise")
    assert merged == {**second, **first}


@given(first=int_mappings, second=int_mappings)
def test_merge_suffix_loses_no_value(first: dict[str, int], second: dict[str, int]) -> None:
    merged = merge_mappings([first, second], on_duplicate="suffix")
    assert all(
        value in {v for k, v in merged.items() if k == key or k.startswith(f"{key}_")}
        for mapping in (first, second)
        for key, value in mapping.items()
    )


@pytest.mark.parametrize("on_duplicate", ["first", "last"])
@given(first=int_mappings, second=int_mappings, third=int_mappings)
def test_merge_is_associative_for_the_keys(
    on_duplicate: MergeMode, first: dict[str, int], second: dict[str, int], third: dict[str, int]
) -> None:
    left = merge_mappings(
        [merge_mappings([first, second], on_duplicate=on_duplicate), third],
        on_duplicate=on_duplicate,
    )
    right = merge_mappings(
        [first, merge_mappings([second, third], on_duplicate=on_duplicate)],
        on_duplicate=on_duplicate,
    )
    assert left == right


#############################
#     Flatten a mapping     #
#############################

nested_mappings = st.dictionaries(keys, int_mappings, max_size=4)


def inner_values(mapping: dict[str, dict[str, int]]) -> dict[str, set[int]]:
    r"""Collect the values found for each key of the inner mappings."""
    seen: dict[str, set[int]] = {}
    for inner_mapping in mapping.values():
        for key, value in inner_mapping.items():
            seen.setdefault(key, set()).add(value)
    return seen


@given(mapping=nested_mappings, separator=separators)
def test_flatten_always_prefix_keeps_every_value(
    mapping: dict[str, dict[str, int]], separator: str
) -> None:
    flat = flatten_mapping(mapping, always_prefix=True, separator=separator)
    expected = {
        f"{outer}{separator}{inner}": value
        for outer, inner_mapping in mapping.items()
        for inner, value in inner_mapping.items()
    }
    assert flat == expected


@given(mapping=nested_mappings, on_duplicate=st.sampled_from(["first", "last", "prefix"]))
def test_flatten_is_idempotent_on_a_flat_mapping(
    mapping: dict[str, dict[str, int]], on_duplicate: FlattenMode
) -> None:
    flat = flatten_mapping(mapping, on_duplicate=on_duplicate)
    assert flatten_mapping({"all": flat}, on_duplicate=on_duplicate) == flat


@given(mapping=nested_mappings)
def test_flatten_prefix_loses_no_value(mapping: dict[str, dict[str, int]]) -> None:
    flat = flatten_mapping(mapping, on_duplicate="prefix")
    assert all(
        value in flat.values()
        for inner_mapping in mapping.values()
        for value in inner_mapping.values()
    )


@given(mapping=nested_mappings)
def test_flatten_raise_raises_if_an_inner_key_has_two_values(
    mapping: dict[str, dict[str, int]],
) -> None:
    assume(any(len(found) > 1 for found in inner_values(mapping).values()))
    with pytest.raises(DuplicateKeyError):
        flatten_mapping(mapping, on_duplicate="raise")


@given(mapping=nested_mappings)
def test_flatten_raise_flattens_if_no_inner_key_has_two_values(
    mapping: dict[str, dict[str, int]],
) -> None:
    seen = inner_values(mapping)
    assume(all(len(found) == 1 for found in seen.values()))
    flat = flatten_mapping(mapping, on_duplicate="raise")
    assert objects_are_equal(flat, {key: next(iter(found)) for key, found in seen.items()})


#######################################
#     Cycles, depth, types, purity    #
#######################################


def make_cycle(value: Any) -> dict[str, Any]:
    r"""Create a dictionary that contains itself."""
    data: dict[str, Any] = {"value": value}
    data["self"] = data
    return data


def nest_dicts(depth: int) -> dict[str, Any]:
    r"""Create dictionaries nested ``depth`` times."""
    data: Any = 1
    for _ in range(depth):
        data = {"a": data}
    return data


KEY_FUNCTIONS = [
    lambda data: remove_keys_if(data, lambda key: key == "never"),
    lambda data: add_prefix_suffix_to_keys(data, "x_", recursive=True),
    convert_to_jsonable,
    to_flat_dict,
]


@pytest.mark.parametrize("function", KEY_FUNCTIONS)
@given(value=values)
def test_a_container_that_contains_itself_raises_a_value_error(function: Any, value: Any) -> None:
    with pytest.raises(ValueError, match="Cycle detected"):
        function(make_cycle(value))


@pytest.mark.parametrize("function", KEY_FUNCTIONS)
@settings(max_examples=5, deadline=None)
@given(depth=st.integers(min_value=1_000, max_value=5_000))
def test_the_depth_is_not_limited(function: Any, depth: int) -> None:
    function(nest_dicts(depth))


@given(value=values)
def test_remove_keys_if_with_a_predicate_that_never_matches_is_the_identity(value: Any) -> None:
    data = {"root": value}
    assert objects_are_equal(remove_keys_if(data, lambda key: False), data)  # noqa: ARG005


@given(value=values)
def test_add_prefix_suffix_to_keys_with_no_prefix_and_suffix_is_the_identity(value: Any) -> None:
    data = {"root": value}
    assert objects_are_equal(add_prefix_suffix_to_keys(data, recursive=True), data)


@pytest.mark.parametrize("function", KEY_FUNCTIONS[:3])
@given(value=values)
def test_the_functions_do_not_modify_the_input(function: Any, value: Any) -> None:
    data = {"root": value}
    expected = copy.deepcopy(data)
    function(data)
    assert objects_are_equal(data, expected)


@given(value=values)
def test_convert_to_jsonable_does_not_depend_on_the_order_of_the_input(value: Any) -> None:
    data = {"root": value}
    assert objects_are_equal(convert_to_jsonable(data), convert_to_jsonable(rebuild(data)))
