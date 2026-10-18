r"""Define the Hypothesis strategies shared by the property-based tests."""

from __future__ import annotations

import copy
from typing import Any

from hypothesis import strategies as st

__all__ = ["lengths", "literal_values", "nested", "rebuild", "scalars", "small_values", "values"]

# Scalars whose ``repr`` can be evaluated by ``ast.literal_eval``.
literal_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.text(),
    st.binary(),
)
scalars = st.one_of(
    literal_scalars,
    st.floats(allow_nan=False),
    st.complex_numbers(allow_nan=False),
)
lengths = st.integers(min_value=1, max_value=64).map(lambda n: 2 * n)


def nested(base: st.SearchStrategy, *, with_sets: bool = True) -> st.SearchStrategy:
    r"""Create a strategy of nested containers of ``base`` values."""
    return st.recursive(
        base,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.tuples(children, children),
            st.dictionaries(st.text(max_size=5), children, max_size=4),
            *(
                [st.frozensets(st.one_of(st.integers(), st.text(max_size=5)), max_size=4)]
                if with_sets
                else []
            ),
        ),
        max_leaves=12,
    )


values = nested(scalars)
# The ``repr`` of a set is not always a Python literal.
literal_values = nested(literal_scalars, with_sets=False)
# A tiny domain, so that independent draws are often equal.
small_values = nested(
    st.one_of(st.none(), st.booleans(), st.integers(-2, 2), st.sampled_from(["", "a", "b"]))
)


def rebuild(data: Any) -> Any:
    r"""Rebuild equal data with the keys of the mappings in another
    order."""
    if isinstance(data, dict):
        return {k: rebuild(v) for k, v in reversed(data.items())}
    if isinstance(data, list):
        return [rebuild(v) for v in data]
    if isinstance(data, tuple):
        return tuple(rebuild(v) for v in data)
    if isinstance(data, frozenset):
        return frozenset(reversed(list(data)))
    return copy.deepcopy(data)
