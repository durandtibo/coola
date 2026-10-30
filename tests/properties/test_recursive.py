r"""Property-based tests for the properties of ``recursive_apply``.

The properties are documented in ``docs/docs/uguide/recursive.md``. The
generic guarantees shared with the iterators (identity, composition, one
call per leaf) are in ``test_traversal.py``.
"""

from __future__ import annotations

import copy
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import Any, NamedTuple

import pytest
from hypothesis import (
    given,
    strategies as st,
)

from coola.iterator import dfs_iterate
from coola.recursive import (
    DefaultTransformer,
    TransformerRegistry,
    get_default_registry,
    recursive_apply,
)
from tests.properties.strategies import nested, scalars, small_values, values

# Nested data without sets: the order of the leaves of a set is not specified.
ordered_values = nested(scalars, with_sets=False)


class Point(NamedTuple):
    x: Any
    y: Any


@dataclass
class Config:
    a: Any
    b: Any


def containers(data: Any) -> list[int]:
    r"""Return the ids of all the containers of nested data."""
    if isinstance(data, dict):
        return [id(data)] + [i for v in data.values() for i in containers(v)]
    if isinstance(data, (list, tuple, frozenset)):
        return [id(data)] + [i for v in data for i in containers(v)]
    return []


def shape(data: Any) -> Any:
    r"""Return the structure of nested data, with the leaves erased."""
    if isinstance(data, dict):
        return {k: shape(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data).__name__, [shape(v) for v in data]
    if isinstance(data, frozenset):
        return "frozenset", len(data)
    return None


##########################
#     Leaves visited     #
##########################


@given(data=ordered_values)
def test_leaves_are_visited_in_dfs_order(data: Any) -> None:
    seen: list[Any] = []

    def record(value: Any) -> Any:
        seen.append(value)
        return value

    recursive_apply(data, record)
    assert [repr(x) for x in seen] == [repr(x) for x in dfs_iterate(data)]


@given(data=values)
def test_function_is_never_called_on_a_container(data: Any) -> None:
    seen: list[Any] = []

    def record(value: Any) -> Any:
        seen.append(value)
        return value

    recursive_apply(data, record)
    assert not any(isinstance(x, (dict, list, tuple, frozenset)) for x in seen)


@given(data=ordered_values)
def test_leaves_of_the_output_are_the_transformed_leaves(data: Any) -> None:
    out = recursive_apply(data, repr)
    assert list(dfs_iterate(out)) == [repr(x) for x in dfs_iterate(data)]


###############################
#     Structure preserved     #
###############################


@given(data=ordered_values)
def test_structure_is_independent_of_the_function(data: Any) -> None:
    assert shape(recursive_apply(data, repr)) == shape(data)


@given(data=st.dictionaries(st.text(max_size=5), scalars, max_size=5))
def test_mapping_keys_are_not_transformed(data: dict) -> None:
    assert list(recursive_apply(data, repr)) == list(data)


@given(data=ordered_values)
def test_output_shares_no_container_with_the_input(data: Any) -> None:
    assert not set(containers(data)) & set(containers(recursive_apply(data, lambda x: x)))


@given(data=values)
def test_input_is_not_modified(data: Any) -> None:
    expected = copy.deepcopy(data)
    recursive_apply(data, repr)
    assert data == expected


@given(data=values)
def test_result_is_deterministic(data: Any) -> None:
    assert repr(recursive_apply(data, repr)) == repr(recursive_apply(data, repr))


@given(data=values, copies=st.integers(min_value=1, max_value=3))
def test_nesting_in_a_list_nests_the_result(data: Any, copies: int) -> None:
    wrapped = [data] * copies
    assert repr(recursive_apply(wrapped, repr)) == repr([recursive_apply(data, repr)] * copies)


######################
#     Subclasses     #
######################


@given(x=scalars, y=scalars)
def test_named_tuple_keeps_its_type_and_fields(x: Any, y: Any) -> None:
    assert recursive_apply(Point(x, y), repr) == Point(repr(x), repr(y))


@given(x=scalars, y=scalars)
def test_named_tuple_keeps_its_type(x: Any, y: Any) -> None:
    assert type(recursive_apply(Point(x, y), repr)) is Point


@given(items=st.lists(st.integers(), max_size=6), maxlen=st.integers(1, 6))
def test_deque_keeps_its_maxlen(items: list[int], maxlen: int) -> None:
    assert recursive_apply(deque(items, maxlen=maxlen), str).maxlen == maxlen


@given(data=st.dictionaries(st.text(max_size=3), st.integers(), max_size=5))
def test_ordered_dict_keeps_its_type_and_order(data: dict) -> None:
    out = recursive_apply(OrderedDict(data), str)
    assert (type(out), list(out)) == (OrderedDict, list(data))


@given(data=st.dictionaries(st.text(max_size=3), st.integers(), max_size=5))
def test_defaultdict_keeps_its_factory(data: dict) -> None:
    assert recursive_apply(defaultdict(int, data), str).default_factory is int


#############################
#     Errors and cycles     #
#############################


@given(data=st.lists(st.just(1), min_size=1, max_size=4))
def test_exception_of_function_propagates_unchanged(data: list) -> None:
    with pytest.raises(KeyError, match="1"):
        recursive_apply(data, {}.__getitem__)


@given(data=st.lists(scalars, max_size=4))
def test_list_containing_itself_raises_value_error(data: list) -> None:
    data.append(data)
    with pytest.raises(ValueError, match="Cycle detected"):
        recursive_apply(data, repr)


@given(key=st.text(max_size=3), data=st.lists(scalars, max_size=3))
def test_cycle_through_a_mapping_raises_value_error(key: str, data: list) -> None:
    mapping: dict = {key: data}
    data.append(mapping)
    with pytest.raises(ValueError, match="Cycle detected"):
        recursive_apply(mapping, repr)


@given(data=ordered_values)
def test_shared_reference_is_not_a_cycle(data: Any) -> None:
    out = recursive_apply([data, data], repr)
    assert out[0] == out[1]


####################################
#     Registry and dataclasses     #
####################################


@given(a=scalars, b=scalars)
def test_dataclass_is_entered_by_default(a: Any, b: Any) -> None:
    out = recursive_apply(Config(a, b), repr)
    assert repr(out) == repr(Config(repr(a), repr(b)))


@given(a=small_values, b=small_values)
def test_dataclass_registered_as_a_leaf(a: Any, b: Any) -> None:
    registry = get_default_registry().copy()
    registry.register(Config, DefaultTransformer())
    config = Config(a, b)
    assert recursive_apply(config, lambda x: x, registry=registry) is config


@given(data=st.text(max_size=5))
def test_leaf_only_registry_applies_the_function_to_the_root(data: str) -> None:
    registry = TransformerRegistry({object: DefaultTransformer()})
    assert recursive_apply(data, str.upper, registry=registry) == data.upper()
