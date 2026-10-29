from __future__ import annotations

import hashlib
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import pytest

from coola.hashing import (
    BaseHasher,
    HasherRegistry,
    Node,
    get_default_registry,
    hash_object,
    tagged,
)
from coola.hashing.engine import digest_of, hash_hex, validate_hash_length
from coola.registry import TypeNotRegisteredError
from tests.unit.hashing.helpers import Unhashable, leaf_hash

if TYPE_CHECKING:
    from collections.abc import Iterator


def blake2b(data: bytes) -> str:
    r"""Return the first 64 characters of the digest of the format."""
    return hashlib.blake2b(data, digest_size=64, person=b"coola-hash-2").hexdigest()[:64]


class Counting(BaseHasher[Any]):
    """A hasher that counts the calls of ``encode`` and describes a
    list."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, data: Any) -> Node:
        self.calls += 1
        return Node("counting", data)


class Wrapper:
    def __init__(self, *children: object) -> None:
        self.children = list(children)

    def __iter__(self) -> Iterator[object]:
        return iter(self.children)


class WrapperHasher(BaseHasher[Wrapper]):
    def encode(self, data: Wrapper) -> Node:
        return Node("wrapper", data.children)


class Failing:
    pass


class FailingHasher(BaseHasher[Failing]):
    def encode(self, data: Failing) -> Node:  # noqa: ARG002
        msg = "boom"
        raise RuntimeError(msg)


@pytest.fixture
def registry() -> HasherRegistry:
    registry = get_default_registry().copy()
    registry.register(Wrapper, WrapperHasher())
    return registry


###############################
#     Tests for the format     #
###############################


def test_leaf_digest_is_the_blake2b_of_the_marked_encoding() -> None:
    assert hash_object(1) == blake2b(b"\x00int\x001")


def test_leaf_digest_helper_matches_the_format() -> None:
    assert leaf_hash(tagged("int", "1")) == hash_object(1)


def test_node_digest_inlines_a_short_leaf_with_its_size() -> None:
    assert hash_object([1]) == blake2b(b"\x01list\x00" + b"\x05int\x001")


def test_node_digest_uses_the_digest_of_a_long_leaf() -> None:
    long = "a" * 100
    leaf = hashlib.blake2b(
        b"\x00" + b"str\x00" + long.encode(), digest_size=64, person=b"coola-hash-2"
    ).digest()
    assert hash_object([long]) == blake2b(b"\x01list\x00\xff" + leaf)


def test_node_digest_inlines_a_leaf_of_64_bytes() -> None:
    text = "a" * 60  # b"str\x00" + 60 bytes = 64 bytes
    assert hash_object([text]) == blake2b(b"\x01list\x00\x40str\x00" + text.encode())


def test_node_digest_uses_the_digest_of_a_leaf_of_65_bytes() -> None:
    text = "a" * 61
    leaf = hashlib.blake2b(
        b"\x00str\x00" + text.encode(), digest_size=64, person=b"coola-hash-2"
    ).digest()
    assert hash_object([text]) == blake2b(b"\x01list\x00\xff" + leaf)


def test_node_digest_uses_the_digest_of_a_nested_node() -> None:
    inner = hashlib.blake2b(b"\x01list\x00", digest_size=64, person=b"coola-hash-2").digest()
    assert hash_object([[]]) == blake2b(b"\x01list\x00\xff" + inner)


def test_empty_node_digest_is_the_blake2b_of_the_marked_tag() -> None:
    assert hash_object([]) == blake2b(b"\x01list\x00")


def test_unordered_node_digest_sorts_the_entries() -> None:
    entries = sorted([b"\x05int\x001", b"\x05int\x002"])
    assert hash_object({1, 2}) == blake2b(b"\x01set\x00" + b"".join(entries))


def test_inlined_leaf_cannot_be_confused_with_a_digest() -> None:
    assert hash_object(["\xff" * 30]) != hash_object([["\xff" * 30]])


def test_long_leaf_in_a_mapping() -> None:
    assert len(hash_object({"k": "x" * 1000})) == 64


def test_leaf_and_node_with_the_same_tag_differ() -> None:
    class Leaf(BaseHasher[Wrapper]):
        def encode(self, data: Wrapper) -> bytes:  # noqa: ARG002
            return tagged("wrapper")

    leaf_registry = HasherRegistry({Wrapper: Leaf()})
    node_registry = HasherRegistry({Wrapper: WrapperHasher()})
    assert leaf_registry.hash(Wrapper()) != node_registry.hash(Wrapper())


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            [1, 2, 3],
            "b783328a416de7ddccf0c133be234184f06068efa172807d0eff2e721b3aec89",
            id="list",
        ),
        pytest.param(
            (1, 2, 3),
            "d942ea47e6839ee5f213ce809e2b56396653855379f1ebe28bff2156559afcd8",
            id="tuple",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            "9497920ce4a85b76ad7deb523668be68e98bfa66d982dc2887f95958f165423c",
            id="dict",
        ),
        pytest.param(
            {1, 2, 3},
            "3ba373d2e386945cb8447dc96f44dd32e698cdca37c9fccd338d23a063d28863",
            id="set",
        ),
        pytest.param(
            {"a": [1, (2, 3)], "b": {"c": {4}}},
            "8054bd4682c55847d2ea9dec43eff76c8e10ed4a505ceef6a708d4ae7494fb89",
            id="nested",
        ),
        pytest.param(
            OrderedDict(a=1),
            "99ad84d5308d0594d0db9085a8e44be56b458f08a03736f5a5a613e8a14d5c43",
            id="ordered_dict",
        ),
    ],
)
def test_hash_object_golden_vector(data: Any, expected: str) -> None:
    assert hash_object(data) == expected


#########################
#     Tests for order     #
#########################


def test_dict_does_not_depend_on_the_insertion_order() -> None:
    assert hash_object({"a": 1, "b": 2}) == hash_object({"b": 2, "a": 1})


def test_set_does_not_depend_on_the_insertion_order() -> None:
    assert hash_object({1, 2, 3}) == hash_object({3, 2, 1})


def test_ordered_dict_depends_on_the_insertion_order() -> None:
    assert hash_object(OrderedDict([("a", 1), ("b", 2)])) != hash_object(
        OrderedDict([("b", 2), ("a", 1)])
    )


def test_list_depends_on_the_order() -> None:
    assert hash_object([1, 2]) != hash_object([2, 1])


def test_dict_keys_and_values_are_not_interchangeable() -> None:
    assert hash_object({"a": "b"}) != hash_object({"b": "a"})


def test_dict_entries_are_not_mixed() -> None:
    assert hash_object({"a": 1, "b": 2}) != hash_object({"a": 2, "b": 1})


def test_unordered_node_with_single_children_sorts_the_children() -> None:
    class Bag(BaseHasher[Wrapper]):
        def encode(self, data: Wrapper) -> Node:
            return Node("bag", data.children, ordered=False)

    registry = get_default_registry().copy()
    registry.register(Wrapper, Bag())
    assert registry.hash(Wrapper(1, 2)) == registry.hash(Wrapper(2, 1))


###########################
#     Tests for structure     #
###########################


def test_nesting_changes_the_hash() -> None:
    assert hash_object([[1], 2]) != hash_object([1, [2]])


def test_nesting_changes_the_hash_compared_to_flat() -> None:
    assert hash_object([[1], 2]) != hash_object([1, 2])


def test_item_boundaries_matter() -> None:
    assert hash_object(["ab", "c"]) != hash_object(["a", "bc"])


###########################
#     Tests for sharing     #
###########################


def test_children_of_a_shared_container_are_encoded_once() -> None:
    counting = Counting()
    registry = HasherRegistry({Wrapper: WrapperHasher(), str: _CountingLeaf(counting)})
    shared = Wrapper("a")
    registry.hash(Wrapper(shared, shared, shared))
    assert counting.calls == 1


def test_shared_container_has_the_same_hash_as_equal_copies() -> None:
    registry = get_default_registry().copy()
    registry.register(Wrapper, WrapperHasher())
    shared = Wrapper(1)
    assert registry.hash(Wrapper(shared, shared)) == registry.hash(Wrapper(Wrapper(1), Wrapper(1)))


def test_shared_list_has_the_same_hash_as_equal_copies() -> None:
    shared = [1, 2]
    assert hash_object([shared, shared]) == hash_object([[1, 2], [1, 2]])


def test_dag_with_many_paths_is_linear() -> None:
    data: list[Any] = []
    for _ in range(60):  # 2^60 paths
        data = [data, data]
    assert len(hash_object(data)) == 64


def test_dag_with_many_paths_has_the_same_hash_as_the_tree() -> None:
    shared: list[Any] = []
    tree: list[Any] = []
    for _ in range(8):
        shared = [shared, shared]
        tree = [list(tree), list(tree)]
    assert hash_object(shared) == hash_object(tree)


def test_shared_leaf_is_encoded_each_time() -> None:
    counting = Counting()
    registry = HasherRegistry({Wrapper: WrapperHasher(), str: _CountingLeaf(counting)})
    registry.hash(Wrapper("a", "a"))
    assert counting.calls == 2


class _CountingLeaf(BaseHasher[str]):
    def __init__(self, counter: Counting) -> None:
        self._counter = counter

    def encode(self, data: str) -> bytes:
        self._counter.calls += 1
        return tagged("str", data)


###########################
#     Tests for cycles     #
###########################


def test_cycle_through_a_list_raises() -> None:
    data: list[Any] = []
    data.append(data)
    with pytest.raises(ValueError, match="self-referencing"):
        hash_object(data)


def test_cycle_through_a_dict_raises() -> None:
    data: dict[str, Any] = {}
    data["self"] = data
    with pytest.raises(ValueError, match="self-referencing"):
        hash_object(data)


def test_cycle_through_a_tuple_in_a_list_raises() -> None:
    data: list[Any] = []
    data.append((data,))
    with pytest.raises(ValueError, match="self-referencing"):
        hash_object(data)


def test_cycle_error_names_the_type() -> None:
    data: list[Any] = []
    data.append(data)
    with pytest.raises(ValueError, match="<class 'list'>"):
        hash_object(data)


def test_cycle_through_a_custom_hasher_that_returns_its_own_object_raises() -> None:
    class SelfHasher(BaseHasher[Wrapper]):
        def encode(self, data: Wrapper) -> Node:
            return Node("self", [data])

    registry = HasherRegistry({Wrapper: SelfHasher()})
    with pytest.raises(ValueError, match="self-referencing"):
        registry.hash(Wrapper())


def test_cycle_of_a_non_ancestor_is_not_reported() -> None:
    inner: list[Any] = []
    assert len(hash_object([inner, [inner]])) == 64


def test_state_is_not_kept_after_a_cycle() -> None:
    data: list[Any] = []
    data.append(data)
    with pytest.raises(ValueError, match="self-referencing"):
        hash_object(data)
    assert hash_object([1]) == hash_object([1])


###########################
#     Tests for depth     #
###########################


def test_depth_beyond_the_recursion_limit_does_not_raise() -> None:
    data: list[Any] = []
    for _ in range(5000):
        data = [data]
    assert len(hash_object(data)) == 64


def test_depth_hash_depends_on_the_depth() -> None:
    first: list[Any] = []
    second: list[Any] = []
    for _ in range(2000):
        first = [first]
    for _ in range(2001):
        second = [second]
    assert hash_object(first) != hash_object(second)


###########################
#     Tests for errors     #
###########################


def test_hasher_exception_propagates_unchanged() -> None:
    registry = HasherRegistry({Failing: FailingHasher()})
    with pytest.raises(RuntimeError, match="boom"):
        registry.hash(Failing())


def test_hasher_exception_of_a_nested_object_propagates_unchanged(
    registry: HasherRegistry,
) -> None:
    registry.register(Failing, FailingHasher())
    with pytest.raises(RuntimeError, match="boom"):
        registry.hash(Wrapper(1, Failing()))


def test_registry_is_usable_after_a_hasher_exception(registry: HasherRegistry) -> None:
    registry.register(Failing, FailingHasher())
    with pytest.raises(RuntimeError, match="boom"):
        registry.hash(Wrapper(Failing()))
    assert registry.hash(Wrapper(1)) == registry.hash(Wrapper(1))


def test_missing_hasher_raises_type_not_registered_error() -> None:
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        hash_object(Unhashable())


def test_missing_hasher_error_names_the_remedies() -> None:
    with pytest.raises(TypeNotRegisteredError, match="__coola_hash__"):
        hash_object(Unhashable())


def test_missing_hasher_of_a_nested_object_raises() -> None:
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        hash_object([1, {"a": Unhashable()}])


def test_missing_hasher_of_a_key_raises() -> None:
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        hash_object({Unhashable(): 1})


class _BoomError(KeyError):
    """A ``KeyError`` that is not a ``TypeNotRegisteredError``."""


def test_key_error_of_the_registry_that_is_not_a_missing_type_propagates() -> None:
    class BrokenRegistry(HasherRegistry):
        def find_hasher(self, data_type: type) -> Any:
            raise _BoomError(data_type)

    with pytest.raises(_BoomError, match="class 'int'"):
        BrokenRegistry().hash(1)


#######################################
#     Tests for re-entrance and isolation     #
#######################################


def test_hasher_that_calls_hash_object_is_reentrant() -> None:
    class Inner(BaseHasher[Wrapper]):
        def encode(self, data: Wrapper) -> bytes:
            return tagged("inner", hash_object(data.children))

    registry = HasherRegistry({Wrapper: Inner()})
    assert registry.hash(Wrapper(1, 2)) == registry.hash(Wrapper(1, 2))


def test_hasher_that_calls_hash_object_depends_on_the_inner_hash() -> None:
    class Inner(BaseHasher[Wrapper]):
        def encode(self, data: Wrapper) -> bytes:
            return tagged("inner", hash_object(data.children))

    registry = HasherRegistry({Wrapper: Inner()})
    assert registry.hash(Wrapper(1, 2)) != registry.hash(Wrapper(2, 1))


def test_custom_registry_does_not_change_the_default_registry() -> None:
    before = hash_object([1, 2])
    HasherRegistry({list: Counting(), int: _LeafInt()}).hash([1, 2])
    assert hash_object([1, 2]) == before


def test_hasher_registered_during_a_call_is_not_used_by_this_call() -> None:
    registry = HasherRegistry({int: _LeafInt(), Wrapper: WrapperHasher()})

    class Registering(BaseHasher[str]):
        def encode(self, data: str) -> bytes:
            registry.register(int, _OtherInt(), exist_ok=True)
            return tagged("str", data)

    registry.register(str, Registering())
    first = registry.hash(Wrapper(1, "a", 2))
    expected = HasherRegistry({int: _LeafInt(), str: _StrLeaf(), Wrapper: WrapperHasher()}).hash(
        Wrapper(1, "a", 2)
    )
    assert first == expected


def test_hasher_registered_during_a_call_is_used_by_the_next_call() -> None:
    registry = HasherRegistry({int: _LeafInt(), Wrapper: WrapperHasher()})
    before = registry.hash(Wrapper(1))
    registry.register(int, _OtherInt(), exist_ok=True)
    assert registry.hash(Wrapper(1)) != before


class _LeafInt(BaseHasher[int]):
    def encode(self, data: int) -> bytes:
        return tagged("int", str(data))


class _OtherInt(BaseHasher[int]):
    def encode(self, data: int) -> bytes:
        return tagged("other", str(data))


class _StrLeaf(BaseHasher[str]):
    def encode(self, data: str) -> bytes:
        return tagged("str", data)


##########################
#     Tests for length     #
##########################


@pytest.mark.parametrize("length", [2, 16, 64, 128])
def test_hash_object_length(length: int) -> None:
    assert len(hash_object({"a": [1, 2]}, length=length)) == length


@pytest.mark.parametrize("length", [2, 16, 64])
def test_hash_is_the_start_of_the_longer_hash(length: int) -> None:
    data = {"a": [1, 2], "b": {3}}
    assert hash_object(data, length=length) == hash_object(data, length=128)[:length]


def test_default_hash_is_the_first_half_of_the_digest() -> None:
    assert hash_object([1, 2]) == hash_object([1, 2], length=128)[:64]


def test_hash_object_length_of_a_leaf_is_the_start_of_the_digest() -> None:
    assert hash_object(1, length=16) == blake2b(b"\x00int\x001")[:16]


@pytest.mark.parametrize("length", [0, -2, 130])
def test_hash_object_length_out_of_range_raises(length: int) -> None:
    with pytest.raises(ValueError, match=r"length must be in the range \[2, 128\]"):
        hash_object([1], length=length)


@pytest.mark.parametrize("length", [3, 63])
def test_hash_object_odd_length_raises(length: int) -> None:
    with pytest.raises(ValueError, match="length must be an even number"):
        hash_object([1], length=length)


@pytest.mark.parametrize("length", [True, 2.0, 64.0, "64", None])
def test_hash_object_length_that_is_not_an_int_raises_type_error(length: Any) -> None:
    with pytest.raises(TypeError, match="length must be an int"):
        hash_object([1], length=length)


def test_hash_object_length_error_message_contains_the_value() -> None:
    with pytest.raises(ValueError, match="got 3"):
        hash_object([1], length=3)


def test_invalid_length_is_detected_before_the_traversal() -> None:
    counting = Counting()
    with pytest.raises(ValueError, match="length must be an even number"):
        HasherRegistry({Wrapper: counting}).hash(Wrapper(), length=3)
    assert counting.calls == 0


def test_registry_hash_length_is_keyword_only() -> None:
    args: tuple[Any, ...] = ([1], 16)
    with pytest.raises(TypeError, match="positional"):
        get_default_registry().hash(*args)


def test_validate_hash_length_accepts_a_valid_length() -> None:
    assert validate_hash_length(64) is None


def test_hash_hex_returns_a_prefix_of_the_digest() -> None:
    registry = get_default_registry()
    assert hash_hex([1], registry, length=8) == digest_of([1], registry).hex()[:8]


###########################################
#     Tests for ignore_unhashable     #
###########################################


@pytest.mark.filterwarnings("ignore:No hasher registered:UserWarning")
def test_placeholder_depends_on_the_type_only() -> None:
    assert hash_object(Unhashable(), ignore_unhashable=True) == hash_object(
        Unhashable(), ignore_unhashable=True
    )


@pytest.mark.filterwarnings("ignore:No hasher registered:UserWarning")
def test_placeholder_differs_between_types() -> None:
    class Other:
        pass

    assert hash_object(Unhashable(), ignore_unhashable=True) != hash_object(
        Other(), ignore_unhashable=True
    )


@pytest.mark.filterwarnings("ignore:No hasher registered:UserWarning")
def test_placeholder_is_the_leaf_of_the_qualified_type_name() -> None:
    expected = leaf_hash(tagged("unhashable", f"{Unhashable.__module__}.Unhashable"))
    assert hash_object(Unhashable(), ignore_unhashable=True) == expected


@pytest.mark.filterwarnings("ignore:No hasher registered:UserWarning")
def test_nested_object_is_replaced() -> None:
    assert len(hash_object([1, {"a": Unhashable()}], ignore_unhashable=True)) == 64


@pytest.mark.filterwarnings("ignore:No hasher registered:UserWarning")
def test_key_is_replaced() -> None:
    assert len(hash_object({Unhashable(): 1}, ignore_unhashable=True)) == 64


def test_one_warning_per_type_per_call() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hash_object([Unhashable() for _ in range(5)], ignore_unhashable=True)
    assert len(caught) == 1


def test_one_warning_per_type_for_two_types() -> None:
    class Other:
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hash_object([Unhashable(), Other(), Unhashable(), Other()], ignore_unhashable=True)
    assert len(caught) == 2


def test_warning_is_emitted_again_on_the_next_call() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hash_object(Unhashable(), ignore_unhashable=True)
        hash_object(Unhashable(), ignore_unhashable=True)
    assert len(caught) == 2


def test_warning_points_at_the_caller_of_hash_object() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hash_object([Unhashable()], ignore_unhashable=True)
    assert caught[0].filename == __file__


def test_warning_points_at_the_caller_of_registry_hash() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_default_registry().hash([Unhashable()], ignore_unhashable=True)
    assert caught[0].filename == __file__


def test_warning_message_names_the_type() -> None:
    with pytest.warns(UserWarning, match="Unhashable"):
        hash_object(Unhashable(), ignore_unhashable=True)


def test_ignore_unhashable_has_no_effect_when_all_the_types_are_hashable() -> None:
    assert hash_object([1, "a"], ignore_unhashable=True) == hash_object([1, "a"])


class _IntSubclass(int):
    pass


def test_validate_hash_length_accepts_an_int_subclass() -> None:
    assert validate_hash_length(_IntSubclass(64)) is None


def test_validate_hash_length_odd_length_message() -> None:
    with pytest.raises(ValueError, match=r"^length must be an even number, got 3$"):
        validate_hash_length(3)
