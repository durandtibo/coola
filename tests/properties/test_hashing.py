r"""Property-based tests for the properties of ``hash_object``.

The properties are documented in ``docs/docs/uguide/hashing.md``.
"""

from __future__ import annotations

import copy
import datetime as dt
import enum
import os
import re
from collections import OrderedDict, deque
from pathlib import PurePosixPath
from typing import Any, NamedTuple

import hypothesis
import pytest
from hypothesis import (
    assume,
    given,
    settings,
    strategies as st,
)

from coola.equality import objects_are_equal
from coola.hashing import HasherRegistry, SequenceHasher, hash_object, tagged
from coola.hashing.base import BaseHasher
from tests.integration.helpers import run_interpreter
from tests.properties.strategies import (
    lengths,
    literal_values,
    rebuild,
    tiny_values,
    values,
)

HEX_PATTERN = re.compile(r"^[0-9a-f]+$")

#######################
#     Fixed width     #
#######################


@given(data=values, length=lengths)
def test_fixed_width_output_is_lowercase_hex_of_given_length(data: Any, length: int) -> None:
    assert len(hash_object(data, length=length)) == length


@given(data=values, length=lengths)
def test_fixed_width_output_is_lowercase_hex(data: Any, length: int) -> None:
    assert HEX_PATTERN.match(hash_object(data, length=length))


@given(data=values, length=st.integers(min_value=0, max_value=64).map(lambda n: 2 * n + 1))
def test_fixed_width_odd_length_raises(data: Any, length: int) -> None:
    with pytest.raises(ValueError, match=r"length"):
        hash_object(data, length=length)


#########################
#     Deterministic     #
#########################


@given(data=values)
def test_deterministic_same_value_same_hash(data: Any) -> None:
    assert hash_object(data) == hash_object(data)


@given(data=values)
def test_deterministic_independent_of_object_identity(data: Any) -> None:
    assert hash_object(data) == hash_object(copy.deepcopy(data))


@pytest.mark.parametrize("seed", ["0", "12345"])
@settings(max_examples=10, deadline=None)
@given(data=literal_values)
def test_deterministic_independent_of_pythonhashseed(seed: str, data: Any) -> None:
    code = (
        "import ast, sys\n"
        "from coola.hashing import hash_object\n"
        "print(hash_object(ast.literal_eval(sys.argv[1])))"
    )
    out = run_interpreter("-c", code, repr(data), env={**os.environ, "PYTHONHASHSEED": seed})
    assert out.stdout.strip() == hash_object(data)


###############################
#     Equality-consistent     #
###############################


@given(data=values)
def test_equality_consistent_rebuilt_objects_are_equal(data: Any) -> None:
    assert objects_are_equal(data, rebuild(data))


@given(data=values)
def test_equality_consistent_equal_objects_have_same_hash(data: Any) -> None:
    assert hash_object(data) == hash_object(rebuild(data))


@given(data=tiny_values, other=tiny_values)
def test_equality_consistent_for_arbitrary_pairs(data: Any, other: Any) -> None:
    assume(objects_are_equal(data, other))
    assert hash_object(data) == hash_object(other)


@given(data=tiny_values, other=tiny_values)
def test_equality_converse_holds_for_builtins(data: Any, other: Any) -> None:
    # The converse can fail for a few types that share a hash on purpose, but
    # not for the builtin types.
    assume(hash_object(data) == hash_object(other))
    assert objects_are_equal(data, other)


def test_equality_consistent_negative_zero() -> None:
    assert hash_object(-0.0) == hash_object(0.0)


def test_equality_consistent_negative_nan() -> None:
    assert hash_object(float("nan")) == hash_object(-float("nan"))


##########################
#     Type-sensitive     #
##########################


@given(x=st.integers())
def test_type_sensitive_int_differs_from_other_types(x: int) -> None:
    hashes = {
        hash_object(x),
        hash_object(float(x)) if abs(x) < 2**53 else hash_object(x + 0.5),
        hash_object(str(x)),
        hash_object(str(x).encode()),
        hash_object([x]),
        hash_object((x,)),
    }
    assert len(hashes) == 6


@pytest.mark.parametrize("x", [0, 1])
def test_type_sensitive_bool_differs_from_int_and_float(x: int) -> None:
    assert len({hash_object(bool(x)), hash_object(x), hash_object(float(x))}) == 3


@given(s=st.text())
def test_type_sensitive_str_differs_from_bytes(s: str) -> None:
    assert hash_object(s) != hash_object(s.encode())


@given(items=st.lists(st.integers(), max_size=5))
def test_type_sensitive_containers_with_same_items_differ(items: list[int]) -> None:
    hashes = {
        hash_object(items),
        hash_object(tuple(items)),
        hash_object(deque(items)),
        hash_object(set(items)),
        hash_object(frozenset(items)),
    }
    assert len(hashes) == 5


@given(data=st.dictionaries(st.text(max_size=5), st.integers(), max_size=4))
def test_type_sensitive_dict_differs_from_ordered_dict(data: dict) -> None:
    assert hash_object(data) != hash_object(OrderedDict(data))


class Color(enum.Enum):
    RED = 1
    BLUE = 2


class Point(NamedTuple):
    x: int
    y: int


@given(data=st.binary())
def test_type_sensitive_bytes_differs_from_bytearray(data: bytes) -> None:
    assert hash_object(data) != hash_object(bytearray(data))


def test_type_sensitive_enum_member_differs_from_its_value_and_siblings() -> None:
    assert (
        len({hash_object(Color.RED), hash_object(Color.BLUE), hash_object(1), hash_object(2)}) == 4
    )


@given(x=st.integers(), y=st.integers())
def test_type_sensitive_namedtuple_differs_from_tuple(x: int, y: int) -> None:
    assert hash_object(Point(x, y)) != hash_object((x, y))


@given(x=st.integers(), y=st.integers())
def test_type_sensitive_namedtuple_hash_is_stable(x: int, y: int) -> None:
    assert hash_object(Point(x, y)) == hash_object(Point(x, y))


@given(
    value=st.datetimes(timezones=st.just(dt.UTC)),
    offset=st.integers(-12, 12),
)
def test_aware_datetime_same_instant_in_other_timezone_hashes_the_same(
    value: dt.datetime, offset: int
) -> None:
    hypothesis.assume(dt.datetime.min.replace(tzinfo=dt.UTC) + dt.timedelta(days=1) < value)
    hypothesis.assume(value < dt.datetime.max.replace(tzinfo=dt.UTC) - dt.timedelta(days=1))
    other = value.astimezone(dt.timezone(dt.timedelta(hours=offset)))
    assert hash_object(value) == hash_object(other)


@given(value=st.datetimes())
def test_naive_datetime_differs_from_aware_datetime(value: dt.datetime) -> None:
    assert hash_object(value) != hash_object(value.replace(tzinfo=dt.UTC))


@given(parts=st.lists(st.text(alphabet="abc", min_size=1, max_size=3), min_size=1, max_size=4))
def test_path_relative_differs_from_absolute(parts: list[str]) -> None:
    assert hash_object(PurePosixPath(*parts)) != hash_object(PurePosixPath("/", *parts))


@given(parts=st.lists(st.text(alphabet="abc", min_size=1, max_size=3), min_size=1, max_size=4))
def test_path_hash_is_stable(parts: list[str]) -> None:
    assert hash_object(PurePosixPath(*parts)) == hash_object(PurePosixPath(*parts))


###############################
#     Structure-sensitive     #
###############################


@given(x=values, y=values)
def test_structure_sensitive_nesting_matters(x: Any, y: Any) -> None:
    assert len({hash_object([[x], y]), hash_object([x, [y]]), hash_object([x, y])}) == 3


@given(s=st.text(min_size=1), data=st.data())
def test_structure_sensitive_item_boundaries_matter(s: str, data: st.DataObject) -> None:
    i = data.draw(st.integers(min_value=0, max_value=len(s)))
    j = data.draw(st.integers(min_value=0, max_value=len(s)).filter(lambda k: k != i))
    assert hash_object([s[:i], s[i:]]) != hash_object([s[:j], s[j:]])


#######################
#     Order rules     #
#######################


@pytest.mark.parametrize("cls", [list, tuple, deque])
@given(items=st.lists(st.integers(), unique=True, min_size=2, max_size=8), data=st.data())
def test_order_ordered_containers_are_order_sensitive(
    cls: type, items: list[int], data: st.DataObject
) -> None:
    shuffled = data.draw(st.permutations(items))
    hypothesis.assume(shuffled != items)
    assert hash_object(cls(items)) != hash_object(cls(shuffled))


@pytest.mark.parametrize("cls", [set, frozenset])
@given(items=st.lists(st.integers(), unique=True, max_size=8), data=st.data())
def test_order_unordered_containers_are_order_insensitive(
    cls: type, items: list[int], data: st.DataObject
) -> None:
    shuffled = data.draw(st.permutations(items))
    assert hash_object(cls(items)) == hash_object(cls(shuffled))


@given(
    mapping=st.dictionaries(st.text(max_size=5), st.integers(), max_size=6),
    data=st.data(),
)
def test_order_mapping_is_insensitive_to_key_order(mapping: dict, data: st.DataObject) -> None:
    keys = data.draw(st.permutations(list(mapping)))
    assert hash_object(mapping) == hash_object({k: mapping[k] for k in keys})


@given(
    mapping=st.dictionaries(st.text(max_size=5), st.integers(), min_size=2, max_size=6),
    data=st.data(),
)
def test_order_ordered_dict_is_sensitive_to_key_order(mapping: dict, data: st.DataObject) -> None:
    keys = list(mapping)
    shuffled = data.draw(st.permutations(keys))
    hypothesis.assume(shuffled != keys)
    assert hash_object(OrderedDict(mapping)) != hash_object(
        OrderedDict((k, mapping[k]) for k in shuffled)
    )


######################
#     Composable     #
######################


class Constant:
    r"""An object whose hash is set by ``ConstantHasher``."""

    def __init__(self, value: int) -> None:
        self.value = value


class ConstantHasher(BaseHasher[Constant]):
    r"""Hash a ``Constant`` by ``value % 2``."""

    def encode(self, data: Constant) -> bytes:
        return tagged("constant", str(data.value % 2))


@given(a=st.lists(st.integers(), max_size=5), data=st.data())
def test_composable_container_hash_only_depends_on_children_hashes(
    a: list[int], data: st.DataObject
) -> None:
    registry = HasherRegistry({Constant: ConstantHasher(), list: SequenceHasher()})
    same = [data.draw(st.integers().filter(lambda n, v=v: n % 2 == v % 2)) for v in a]
    assert registry.hash([Constant(v) for v in a]) == registry.hash([Constant(v) for v in same])


@given(a=st.lists(st.integers(), min_size=1, max_size=5), data=st.data())
def test_composable_changing_a_child_hash_changes_the_container_hash(
    a: list[int], data: st.DataObject
) -> None:
    registry = HasherRegistry({Constant: ConstantHasher(), list: SequenceHasher()})
    i = data.draw(st.integers(0, len(a) - 1))
    other = [*a[:i], a[i] + 1, *a[i + 1 :]]
    assert registry.hash([Constant(v) for v in a]) != registry.hash([Constant(v) for v in other])


#############################
#     Unsupported types     #
#############################


class Unsupported:
    r"""A type with no registered hasher."""


def test_no_silent_false_equality_unsupported_type_raises() -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_object(Unsupported())


@given(x=values)
def test_no_silent_false_equality_unsupported_type_in_a_list_raises(x: Any) -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_object([x, Unsupported()])


@given(x=values)
def test_no_silent_false_equality_unsupported_type_in_a_dict_raises(x: Any) -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_object({"a": Unsupported(), "b": x})


@given(x=values)
def test_no_silent_false_equality_unsupported_type_in_a_tuple_raises(x: Any) -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_object((Unsupported(), x))


@given(x=values)
def test_no_silent_false_equality_placeholder_warns(x: Any) -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        hash_object([x, Unsupported()], ignore_unhashable=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
@given(x=values)
def test_no_silent_false_equality_placeholder_has_the_default_length(x: Any) -> None:
    assert len(hash_object([x, Unsupported()], ignore_unhashable=True)) == 64


#################################
#     Cross-library scalars     #
#################################


class MyInt(int):
    pass


class MyFloat(float):
    pass


class MyComplex(complex):
    pass


@given(x=st.integers())
def test_scalar_int_subclass_hashes_like_int(x: int) -> None:
    assert hash_object(MyInt(x)) == hash_object(x)


@given(x=st.floats(allow_nan=False))
def test_scalar_float_subclass_hashes_like_float(x: float) -> None:
    assert hash_object(MyFloat(x)) == hash_object(x)


@given(x=st.complex_numbers(allow_nan=False))
def test_scalar_complex_subclass_hashes_like_complex(x: complex) -> None:
    assert hash_object(MyComplex(x)) == hash_object(x)


@given(x=st.floats(allow_nan=False))
def test_scalar_numpy_float64_hashes_like_float(x: float) -> None:
    np = pytest.importorskip("numpy")
    assert hash_object(np.float64(x)) == hash_object(x)


################
#     Pure     #
################


@given(data=values)
def test_pure_hashing_does_not_modify_the_input(data: Any) -> None:
    before = copy.deepcopy(data)
    hash_object(data)
    assert objects_are_equal(data, before)
