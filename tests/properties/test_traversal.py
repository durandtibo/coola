r"""Property-based tests for the guarantees of the iterators and of
``recursive_apply``."""

from __future__ import annotations

from collections import Counter
from typing import Any

from hypothesis import given, strategies as st

from coola.iterator import bfs_iterate, dfs_iterate
from coola.recursive import recursive_apply
from tests.properties.strategies import values


def leaves_of(data: Any) -> list[Any]:
    r"""Return the leaves of nested containers (the empty containers have
    no leaf)."""
    if isinstance(data, dict):
        return [leaf for value in data.values() for leaf in leaves_of(value)]
    if isinstance(data, (list, tuple, frozenset)):
        return [leaf for value in data for leaf in leaves_of(value)]
    return [data]


def counts(leaves: Any) -> Counter[str]:
    # The leaves are compared by their type and representation: a ``1`` and a ``True`` are equal.
    return Counter(f"{type(leaf).__name__}:{leaf!r}" for leaf in leaves)


#####################
#     Iterators     #
#####################


@given(data=values)
def test_dfs_iterate_yields_the_leaves(data: Any) -> None:
    assert counts(dfs_iterate(data)) == counts(leaves_of(data))


@given(data=values)
def test_bfs_iterate_yields_the_leaves(data: Any) -> None:
    assert counts(bfs_iterate(data)) == counts(leaves_of(data))


@given(data=values)
def test_bfs_and_dfs_iterate_yield_the_same_leaves(data: Any) -> None:
    assert counts(bfs_iterate(data)) == counts(dfs_iterate(data))


@given(data=st.lists(values, max_size=4))
def test_dfs_iterate_keeps_the_order_of_a_list(data: list[Any]) -> None:
    expected = [leaf for value in data for leaf in dfs_iterate(value)]
    assert [repr(leaf) for leaf in dfs_iterate(data)] == [repr(leaf) for leaf in expected]


@given(data=values)
def test_iterate_does_not_modify_the_data(data: Any) -> None:
    before = repr(data)
    list(dfs_iterate(data))
    list(bfs_iterate(data))
    assert repr(data) == before


###########################
#     recursive_apply     #
###########################


@given(data=values)
def test_recursive_apply_identity_returns_equal_data(data: Any) -> None:
    assert recursive_apply(data, lambda x: x) == data


@given(data=values)
def test_recursive_apply_keeps_the_type_of_the_containers(data: Any) -> None:
    assert type(recursive_apply(data, lambda x: x)) is type(data)


@given(data=values)
def test_recursive_apply_composes_the_functions(data: Any) -> None:
    assert recursive_apply(recursive_apply(data, repr), len) == recursive_apply(
        data, lambda x: len(repr(x))
    )


@given(data=values)
def test_recursive_apply_applies_the_function_to_every_leaf(data: Any) -> None:
    seen: list[Any] = []

    def record(value: Any) -> Any:
        seen.append(value)
        return value

    recursive_apply(data, record)
    assert counts(seen) == counts(leaves_of(data))
