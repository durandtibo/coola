r"""Property-based tests for the properties of ``to_jsonable``.

The properties are documented in the docstring of ``coola.jsonable``.
"""

from __future__ import annotations

import copy
import datetime
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from hypothesis import (
    given,
    strategies as st,
)

from coola.equality import objects_are_equal
from coola.jsonable import to_jsonable
from tests.properties.strategies import rebuild


class Color(Enum):
    RED = 1
    GREEN = (2, "g")


@dataclass
class Box:
    value: Any


# The leaves have no NaN: the output of ``json.loads`` is then equal to the output of
# ``to_jsonable``. The text of the keys has no digit and is not "true", "false" or "null", so
# the keys of different types never give the same string.
json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.text(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.binary(),
    st.sampled_from([Color.RED, Color.GREEN]),
    st.just(PurePosixPath("a/b")),
    st.decimals(allow_nan=False, allow_infinity=False),
    st.dates(),
    st.datetimes(),
    st.times(),
    st.timedeltas(max_value=datetime.timedelta(days=1000)),
)
json_keys = st.one_of(
    st.text(alphabet="abcd", min_size=1, max_size=3),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.just(None),
    st.just(PurePosixPath("k")),
)


def nested_json(base: st.SearchStrategy) -> st.SearchStrategy:
    r"""Create a strategy of nested containers of ``base`` values."""
    return st.recursive(
        base,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.tuples(children, children),
            st.dictionaries(json_keys, children, max_size=4),
            st.frozensets(st.one_of(st.integers(), st.text(max_size=5)), max_size=4),
            st.builds(Box, children),
        ),
        max_leaves=12,
    )


@given(data=nested_json(json_scalars))
def test_to_jsonable_is_accepted_by_json_and_round_trips(data: Any) -> None:
    result = to_jsonable(data)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@given(data=nested_json(json_scalars))
def test_to_jsonable_is_idempotent(data: Any) -> None:
    result = to_jsonable(data)
    assert to_jsonable(result) == result


@given(data=nested_json(json_scalars))
def test_to_jsonable_does_not_depend_on_the_order_of_the_input(data: Any) -> None:
    assert to_jsonable(rebuild(data)) == to_jsonable(data)


@given(data=nested_json(json_scalars))
def test_to_jsonable_does_not_modify_the_input(data: Any) -> None:
    expected = copy.deepcopy(data)
    to_jsonable(data)
    assert objects_are_equal(data, expected)
