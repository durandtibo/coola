r"""Property-based tests for the guarantees of the functions of
``coola.nested``."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, strategies as st

from coola.equality import objects_are_equal
from coola.nested import (
    DuplicateKeyError,
    flatten_mapping,
    from_flat_dict,
    merge_mappings,
    to_flat_dict,
)
from tests.properties.strategies import flat_leaves, int_mappings, keys, nested_dicts

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


@given(data=nested_dicts, separator=separators)
def test_round_trip_with_any_separator(data: dict[str, Any], separator: str) -> None:
    flat = to_flat_dict(data, separator=separator)
    assert from_flat_dict(flat, separator=separator) == data


@given(data=nested_dicts)
def test_flat_dict_has_string_keys_and_no_nested_dict(data: dict[str, Any]) -> None:
    flat = to_flat_dict(data)
    assert all(isinstance(key, str) for key in flat)
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
    flat = to_flat_dict(data, prefix=prefix)
    expected = {f"{prefix}.{key}": value for key, value in to_flat_dict(data).items()}
    if data:
        assert flat == expected


@given(first=keys, second=keys, value=flat_leaves, other=flat_leaves)
def test_a_key_that_is_a_scalar_and_a_prefix_is_a_conflict(
    first: str, second: str, value: Any, other: Any
) -> None:
    with pytest.raises(ValueError, match="conflict"):
        from_flat_dict({first: value, f"{first}.{second}": other})
    with pytest.raises(ValueError, match="conflict"):
        from_flat_dict({f"{first}.{second}": other, first: value})


@given(key=keys, value=flat_leaves, other=flat_leaves)
def test_keys_that_have_the_same_string_collide(key: str, value: Any, other: Any) -> None:
    with pytest.raises(ValueError, match="collision"):
        to_flat_dict({1: value, "1": other})
    assert to_flat_dict({key: value}) == {key: value}


############################
#     Merge the mappings   #
############################


@given(mapping=int_mappings, on_duplicate=strategies)
def test_merge_single_mapping_is_unchanged(mapping: dict[str, int], on_duplicate: str) -> None:
    assert merge_mappings([mapping], on_duplicate=on_duplicate) == mapping


@given(mapping=int_mappings, on_duplicate=strategies)
def test_merge_is_idempotent(mapping: dict[str, int], on_duplicate: str) -> None:
    assert merge_mappings([mapping, mapping], on_duplicate=on_duplicate) == mapping


@given(first=int_mappings, second=int_mappings, on_duplicate=strategies)
def test_merge_of_a_merged_mapping_is_unchanged(
    first: dict[str, int], second: dict[str, int], on_duplicate: str
) -> None:
    merged = merge_mappings([first, second], on_duplicate=on_duplicate)
    assert merge_mappings([merged, merged], on_duplicate=on_duplicate) == merged


@given(first=int_mappings, second=int_mappings)
def test_merge_keeps_every_key(first: dict[str, int], second: dict[str, int]) -> None:
    for on_duplicate in ("first", "last"):
        merged = merge_mappings([first, second], on_duplicate=on_duplicate)
        assert set(merged) == set(first) | set(second)


@given(first=int_mappings, second=int_mappings)
def test_merge_first_keeps_the_first_value_and_last_the_last_value(
    first: dict[str, int], second: dict[str, int]
) -> None:
    kept_first = merge_mappings([first, second], on_duplicate="first")
    kept_last = merge_mappings([first, second], on_duplicate="last")
    for key in set(first) | set(second):
        assert kept_first[key] == first.get(key, second.get(key))
        assert kept_last[key] == second.get(key, first.get(key))


@given(first=int_mappings, second=int_mappings)
def test_merge_raise_raises_iff_a_key_has_two_values(
    first: dict[str, int], second: dict[str, int]
) -> None:
    conflict = any(key in second and second[key] != value for key, value in first.items())
    if conflict:
        with pytest.raises(DuplicateKeyError):
            merge_mappings([first, second], on_duplicate="raise")
    else:
        merged = merge_mappings([first, second], on_duplicate="raise")
        assert merged == {**second, **first}


@given(first=int_mappings, second=int_mappings)
def test_merge_suffix_loses_no_value(first: dict[str, int], second: dict[str, int]) -> None:
    merged = merge_mappings([first, second], on_duplicate="suffix")
    for mapping in (first, second):
        for key, value in mapping.items():
            assert value in {v for k, v in merged.items() if k == key or k.startswith(f"{key}_")}


@given(first=int_mappings, second=int_mappings, third=int_mappings, on_duplicate=strategies)
def test_merge_is_associative_for_the_keys(
    first: dict[str, int], second: dict[str, int], third: dict[str, int], on_duplicate: str
) -> None:
    if on_duplicate == "suffix":
        return
    left = merge_mappings([merge_mappings([first, second], on_duplicate), third], on_duplicate)
    right = merge_mappings([first, merge_mappings([second, third], on_duplicate)], on_duplicate)
    assert left == right


#############################
#     Flatten a mapping     #
#############################

nested_mappings = st.dictionaries(keys, int_mappings, max_size=4)


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
    mapping: dict[str, dict[str, int]], on_duplicate: str
) -> None:
    flat = flatten_mapping(mapping, on_duplicate=on_duplicate)
    assert flatten_mapping({"all": flat}, on_duplicate=on_duplicate) == flat


@given(mapping=nested_mappings)
def test_flatten_prefix_loses_no_value(mapping: dict[str, dict[str, int]]) -> None:
    flat = flatten_mapping(mapping, on_duplicate="prefix")
    for inner_mapping in mapping.values():
        for value in inner_mapping.values():
            assert value in flat.values()


@given(mapping=nested_mappings)
def test_flatten_raise_raises_iff_an_inner_key_has_two_values(
    mapping: dict[str, dict[str, int]],
) -> None:
    seen: dict[str, set[int]] = {}
    for inner_mapping in mapping.values():
        for key, value in inner_mapping.items():
            seen.setdefault(key, set()).add(value)
    if any(len(found) > 1 for found in seen.values()):
        with pytest.raises(DuplicateKeyError):
            flatten_mapping(mapping, on_duplicate="raise")
    else:
        flat = flatten_mapping(mapping, on_duplicate="raise")
        assert objects_are_equal(flat, {key: next(iter(found)) for key, found in seen.items()})
