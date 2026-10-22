r"""Define the Hypothesis strategies shared by the property-based
tests."""

from __future__ import annotations

import copy
from typing import Any

from hypothesis import strategies as st

__all__ = [
    "flat_leaves",
    "float_lists",
    "int_mappings",
    "keys",
    "lengths",
    "literal_values",
    "nan_values",
    "nested",
    "nested_dicts",
    "quantile_lists",
    "rebuild",
    "scalars",
    "small_values",
    "values",
]

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
# Nested data whose leaves can be NaN (also in the complex numbers).
nan_values = nested(
    st.one_of(literal_scalars, st.floats(allow_nan=True), st.complex_numbers(allow_nan=True))
)
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


# Keys of the dictionaries that are flattened: no separator, no underscore (used by the suffixes).
keys = st.text(alphabet="abcd", min_size=1, max_size=3)
flat_leaves = st.one_of(st.none(), st.booleans(), st.integers(-100, 100), st.text(max_size=4))
# Nested dictionaries with only dictionaries and scalars, so that flattening them is lossless.
nested_dicts = st.dictionaries(
    keys,
    st.recursive(
        flat_leaves,
        lambda children: st.dictionaries(keys, children, max_size=3),
        max_leaves=8,
    ),
    max_size=4,
)
int_mappings = st.dictionaries(keys, st.integers(-5, 5), max_size=4)
float_lists = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=30,
)
quantile_lists = st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=5)
