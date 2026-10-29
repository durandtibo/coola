from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from coola.hashing import (
    BaseHasher,
    HasherRegistry,
    MappingHasher,
    ReprHasher,
    SequenceHasher,
    StrHasher,
    StringHasher,
    get_default_registry,
    tagged,
)
from coola.registry import AlreadyRegisteredError, TypeNotRegisteredError, TypeRegistry
from coola.utils import fork
from tests.helpers import nest
from tests.unit.hashing.helpers import Unhashable, hash_with, leaf_hash

if TYPE_CHECKING:
    from coola.hashing import Hasher


class CustomList(list):
    r"""Custom class that inherits from list."""


####################################
#     Tests for HasherRegistry     #
####################################


def test_hasher_registry_init_empty() -> None:
    registry = HasherRegistry()
    assert len(registry._state) == 0


def test_hasher_registry_init_with_state() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    assert list in registry._state


def test_hasher_registry_init_with_state_keeps_the_hasher() -> None:
    hasher = SequenceHasher()
    registry = HasherRegistry({list: hasher})
    assert registry._state[list] is hasher


def test_hasher_registry_init_with_state_copies_the_state() -> None:
    # Verify it's a copy — mutations to the source dict don't affect the registry.
    initial_state: dict[type, Hasher[Any]] = {list: SequenceHasher()}
    registry = HasherRegistry(initial_state)
    initial_state[tuple] = ReprHasher()
    assert tuple not in registry._state


def test_hasher_registry_repr() -> None:
    assert repr(HasherRegistry()).startswith("HasherRegistry(")


def test_hasher_registry_str() -> None:
    assert str(HasherRegistry()).startswith("HasherRegistry(")


##############################
#     Tests for register     #
##############################


def test_hasher_registry_register_new_type_has_hasher() -> None:
    registry = HasherRegistry()
    hasher = SequenceHasher()
    registry.register(list, hasher)
    assert registry.has_hasher(list)


def test_hasher_registry_register_new_type_keeps_the_hasher() -> None:
    registry = HasherRegistry()
    hasher = SequenceHasher()
    registry.register(list, hasher)
    assert registry._state[list] is hasher


def test_hasher_registry_register_existing_type_without_exist_ok() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register(list, MappingHasher(), exist_ok=False)


def test_hasher_registry_register_existing_type_with_exist_ok() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    hasher = MappingHasher()
    registry.register(list, hasher, exist_ok=True)
    assert registry._state[list] is hasher


###################################
#     Tests for register_many     #
###################################


def test_hasher_registry_register_many_list() -> None:
    registry = HasherRegistry()
    registry.register_many(
        {
            list: SequenceHasher(),
            dict: MappingHasher(),
            str: StringHasher(),
        }
    )
    assert registry.has_hasher(list)


def test_hasher_registry_register_many_dict() -> None:
    registry = HasherRegistry()
    registry.register_many(
        {
            list: SequenceHasher(),
            dict: MappingHasher(),
            str: StringHasher(),
        }
    )
    assert registry.has_hasher(dict)


def test_hasher_registry_register_many_str() -> None:
    registry = HasherRegistry()
    registry.register_many(
        {
            list: SequenceHasher(),
            dict: MappingHasher(),
            str: StringHasher(),
        }
    )
    assert registry.has_hasher(str)


def test_hasher_registry_register_many_with_existing_type() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register_many({list: MappingHasher(), dict: MappingHasher()}, exist_ok=False)


def test_hasher_registry_register_many_with_exist_ok() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    hasher = MappingHasher()
    registry.register_many({list: hasher, dict: MappingHasher()}, exist_ok=True)
    assert registry._state[list] is hasher


################################
#     Tests for has_hasher     #
################################


def test_hasher_registry_has_hasher_true() -> None:
    assert HasherRegistry({list: SequenceHasher()}).has_hasher(list)


def test_hasher_registry_has_hasher_false() -> None:
    assert not HasherRegistry().has_hasher(list)


def test_hasher_registry_has_hasher_does_not_check_mro() -> None:
    # has_hasher checks direct registration only — not MRO fallback.
    assert not HasherRegistry({object: ReprHasher()}).has_hasher(list)


#################################
#     Tests for find_hasher     #
#################################


def test_hasher_registry_find_hasher_direct_match() -> None:
    hasher = SequenceHasher()
    registry = HasherRegistry({list: hasher})
    assert registry.find_hasher(list) is hasher


def test_hasher_registry_find_hasher_mro_lookup() -> None:
    # CustomList inherits from list, so it resolves to the list hasher.
    hasher = SequenceHasher()
    registry = HasherRegistry({list: hasher})
    assert registry.find_hasher(CustomList) is hasher


def test_hasher_registry_find_hasher_most_specific() -> None:
    base_hasher = SequenceHasher()
    specific_hasher = MappingHasher()
    registry = HasherRegistry(
        {object: ReprHasher(), list: base_hasher, CustomList: specific_hasher}
    )
    assert registry.find_hasher(CustomList) is specific_hasher


def test_hasher_registry_find_hasher_abc_not_in_mro() -> None:
    # Sequence is an ABC not in list.__mro__, so list falls back to object.
    registry = HasherRegistry({object: ReprHasher(), Sequence: SequenceHasher()})
    assert isinstance(registry.find_hasher(list), ReprHasher)


##########################
#     Tests for hash     #
##########################


def test_hasher_registry_hash_with_list() -> None:
    assert (
        HasherRegistry({object: StrHasher(), list: SequenceHasher()}).hash([1, 2, 3])
        == "d4b06c73382157bf177e93c51702d837aa65bdde42ac5569528babb18797a387"
    )


def test_hasher_registry_hash_with_dict() -> None:
    assert (
        HasherRegistry({object: StrHasher(), dict: MappingHasher()}).hash({"a": 1, "b": 2})
        == "e6952362146927fc104d6be3f2c359d55a545e263b71d1feac43b2625e2dcf48"
    )


def test_hasher_registry_hash_with_nested_structure() -> None:
    registry = HasherRegistry(
        {
            object: StrHasher(),
            list: SequenceHasher(),
            dict: MappingHasher(),
        }
    )
    assert (
        registry.hash({"a": [1, 2], "b": [3, 4]})
        == "90260e69f9de2d904ed4b0c12e9109f47619f42cd8a9107a3bb8a05d78430dc1"
    )


def test_hasher_registry_hash_with_empty_list() -> None:
    assert (
        HasherRegistry({list: SequenceHasher()}).hash([])
        == "a22c70eed7e818075d623bbe39c012daef9b7450cca490bb1c48bb92a53fbac0"
    )


def test_hasher_registry_hash_with_empty_dict() -> None:
    assert (
        HasherRegistry({dict: MappingHasher()}).hash({})
        == "6662e425fd8dcb2032f333732915004156c02bc14bcacf74f8c311f80a732b34"
    )


def test_hasher_registry_hash_with_length() -> None:
    result = HasherRegistry({object: ReprHasher()}).hash(42, length=16)
    assert result == "523ebc8accc11fcf"


def test_hasher_registry_hash_with_length_length() -> None:
    result = HasherRegistry({object: ReprHasher()}).hash(42, length=16)
    assert len(result) == 16


def test_hasher_registry_hash_registry_isolation_first_registry() -> None:
    hasher1 = SequenceHasher()
    hasher2 = MappingHasher()
    registry1 = HasherRegistry({list: hasher1})
    HasherRegistry({list: hasher2})
    assert registry1._state[list] is hasher1


def test_hasher_registry_hash_registry_isolation_second_registry() -> None:
    hasher1 = SequenceHasher()
    hasher2 = MappingHasher()
    HasherRegistry({list: hasher1})
    registry2 = HasherRegistry({list: hasher2})
    assert registry2._state[list] is hasher2


def test_hasher_registry_hash_unhashable_raises_by_default() -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        HasherRegistry().hash(Unhashable())


def test_hasher_registry_hash_unhashable_raises_when_explicitly_false() -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        HasherRegistry().hash(Unhashable(), ignore_unhashable=False)


def test_hasher_registry_hash_ignore_unhashable_returns_str_warns() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        HasherRegistry().hash(Unhashable(), ignore_unhashable=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hasher_registry_hash_ignore_unhashable_returns_str() -> None:
    result = HasherRegistry().hash(Unhashable(), ignore_unhashable=True)
    assert isinstance(result, str)


def test_hasher_registry_hash_ignore_unhashable_respects_length_warns() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        HasherRegistry().hash(Unhashable(), length=16, ignore_unhashable=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hasher_registry_hash_ignore_unhashable_respects_length() -> None:
    result = HasherRegistry().hash(Unhashable(), length=16, ignore_unhashable=True)
    assert len(result) == 16


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hasher_registry_hash_ignore_unhashable_is_deterministic() -> None:
    registry = HasherRegistry()
    first = registry.hash(Unhashable(), ignore_unhashable=True)
    second = registry.hash(Unhashable(), ignore_unhashable=True)
    assert first == second


def test_hasher_registry_hash_ignore_unhashable_nested_in_list_warns() -> None:
    # No `object` fallback registered, so the `Unhashable` element would
    # normally be unresolvable; ignore_unhashable=True must handle it.
    registry = HasherRegistry({int: StrHasher(), list: SequenceHasher()})
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        registry.hash([1, Unhashable(), 3], ignore_unhashable=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hasher_registry_hash_ignore_unhashable_nested_in_list() -> None:
    # No `object` fallback registered, so the `Unhashable` element would
    # normally be unresolvable; ignore_unhashable=True must handle it.
    registry = HasherRegistry({int: StrHasher(), list: SequenceHasher()})
    result = registry.hash([1, Unhashable(), 3], ignore_unhashable=True)
    assert isinstance(result, str)


def test_hasher_registry_hash_ignore_unhashable_nested_in_list_raises_by_default() -> None:
    # No `object` fallback registered, so an `Unhashable` element inside
    # the list cannot be resolved.
    registry = HasherRegistry({list: SequenceHasher()})
    with pytest.raises(KeyError, match=r"is not registered"):
        registry.hash([1, Unhashable(), 3])


def test_hasher_registry_hash_ignore_unhashable_nested_in_dict_warns() -> None:
    # No `object` fallback registered, so the `Unhashable` value would
    # normally be unresolvable; ignore_unhashable=True must handle it.
    registry = HasherRegistry({str: StrHasher(), dict: MappingHasher()})
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        registry.hash({"a": Unhashable()}, ignore_unhashable=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hasher_registry_hash_ignore_unhashable_nested_in_dict() -> None:
    # No `object` fallback registered, so the `Unhashable` value would
    # normally be unresolvable; ignore_unhashable=True must handle it.
    registry = HasherRegistry({str: StrHasher(), dict: MappingHasher()})
    result = registry.hash({"a": Unhashable()}, ignore_unhashable=True)
    assert isinstance(result, str)


def test_hasher_registry_hash_ignore_unhashable_nested_in_dict_raises_by_default() -> None:
    registry = HasherRegistry({str: StrHasher(), dict: MappingHasher()})
    with pytest.raises(KeyError, match=r"is not registered"):
        registry.hash({"a": Unhashable()})


def test_hasher_registry_accepts_protocol_implementation_without_subclassing() -> None:
    class PlainHasher:
        def encode(self, data: Any) -> bytes:  # noqa: ARG002
            return tagged("plain", "x")

    registry = HasherRegistry({object: PlainHasher()})
    assert registry.hash(1, length=4) == leaf_hash(tagged("plain", "x"), length=4)


def test_hasher_registry_hash_ignore_unhashable_warns() -> None:
    with pytest.warns(UserWarning, match="placeholder"):
        HasherRegistry().hash(Unhashable(), ignore_unhashable=True)


def test_hasher_registry_hash_self_referencing_raises() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    data: list = []
    data.append(data)
    with pytest.raises(ValueError, match="self-referencing"):
        registry.hash(data)


def test_hasher_registry_hash_same_object_twice_is_not_a_cycle() -> None:
    registry = HasherRegistry({list: SequenceHasher(), int: StrHasher()})
    shared = [1]
    assert isinstance(registry.hash([shared, shared]), str)


########################################
#     Tests for the fallbacks          #
########################################


class Marked:
    r"""A type matched by a fallback."""


def is_marked(data_type: type) -> bool:
    return issubclass(data_type, Marked)


class Other:
    r"""A type matched by another fallback."""


def is_other(data_type: type) -> bool:
    return issubclass(data_type, Other)


def test_hasher_registry_find_hasher_uses_the_fallback() -> None:
    hasher = StrHasher()
    registry = HasherRegistry({list: SequenceHasher()})
    registry.register_fallback(is_marked, hasher)
    assert registry.find_hasher(Marked) is hasher


def test_hasher_registry_find_hasher_fallback_for_a_subclass() -> None:
    class Child(Marked): ...

    hasher = StrHasher()
    registry = HasherRegistry()
    registry.register_fallback(is_marked, hasher)
    assert registry.find_hasher(Child) is hasher


def test_hasher_registry_find_hasher_registered_type_wins_over_the_fallback() -> None:
    registered = ReprHasher()
    registry = HasherRegistry({Marked: registered})
    registry.register_fallback(is_marked, StrHasher())
    assert registry.find_hasher(Marked) is registered


def test_hasher_registry_find_hasher_registered_parent_wins_over_the_fallback() -> None:
    class Child(Marked): ...

    registered = ReprHasher()
    registry = HasherRegistry({Marked: registered})
    registry.register_fallback(is_marked, StrHasher())
    assert registry.find_hasher(Child) is registered


def test_hasher_registry_find_hasher_first_matching_fallback_wins() -> None:
    first, second = StrHasher(), ReprHasher()
    registry = HasherRegistry()
    registry.register_fallback(is_marked, first)
    registry.register_fallback(lambda tp: tp is Marked, second)
    assert registry.find_hasher(Marked) is first


def test_hasher_registry_find_hasher_skips_the_fallback_that_does_not_match() -> None:
    first, second = StrHasher(), ReprHasher()
    registry = HasherRegistry()
    registry.register_fallback(lambda _: False, first)
    registry.register_fallback(is_marked, second)
    assert registry.find_hasher(Marked) is second


def test_hasher_registry_find_hasher_no_matching_fallback_raises() -> None:
    registry = HasherRegistry()
    registry.register_fallback(is_marked, StrHasher())
    with pytest.raises(KeyError, match="is not registered"):
        registry.find_hasher(Unhashable)


def test_hasher_registry_fallback_wins_over_the_object_hasher() -> None:
    fallback = StrHasher()
    registry = HasherRegistry({object: ReprHasher()})
    registry.register_fallback(is_marked, fallback)
    assert registry.find_hasher(Marked) is fallback


class ConstantHasher(BaseHasher[Any]):
    def encode(self, data: Any) -> bytes:  # noqa: ARG002
        return tagged("constant")


def test_hasher_registry_fallback_is_used_by_hash() -> None:
    registry = HasherRegistry()
    registry.register_fallback(is_marked, ConstantHasher())
    assert registry.hash(Marked()) == leaf_hash(tagged("constant"))


def test_hasher_registry_has_hasher_ignores_the_fallbacks() -> None:
    registry = HasherRegistry()
    registry.register_fallback(is_marked, StrHasher())
    assert not registry.has_hasher(Marked)


def test_hasher_registry_repr_does_not_show_the_fallbacks_without_fallback() -> None:
    assert "(fallback)" not in repr(HasherRegistry())


def test_hasher_registry_repr_shows_the_fallbacks() -> None:
    registry = HasherRegistry()
    registry.register_fallback(is_marked, StrHasher())
    assert "(fallback)" in repr(registry)


def test_hasher_registry_repr_shows_the_fallback_hashers() -> None:
    registry = HasherRegistry()
    registry.register_fallback(is_marked, StrHasher())
    assert "StrHasher()" in repr(registry)


@pytest.mark.parametrize(
    "text", ["register a hasher", "__coola_hash__", "ignore_unhashable=True", "Unhashable"]
)
def test_hasher_registry_unhashable_error_names_the_remedies(text: str) -> None:
    registry = HasherRegistry()
    with pytest.raises(KeyError, match="is not registered") as exc_info:
        registry.hash(Unhashable())
    assert text in str(exc_info.value)


def test_hasher_registry_unhashable_error_is_a_type_not_registered_error() -> None:
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        HasherRegistry().hash(Unhashable())


def test_hasher_registry_hash_reraises_a_key_error_that_is_not_a_missing_type() -> None:
    class BrokenRegistry(HasherRegistry):
        def find_hasher(self, data_type: type) -> Any:  # noqa: ARG002
            msg = "broken"
            raise KeyError(msg)

    with pytest.raises(KeyError, match="broken") as exc_info:
        BrokenRegistry().hash(1)
    assert not isinstance(exc_info.value, TypeNotRegisteredError)


def test_hasher_registry_hash_deeply_nested_data_does_not_raise() -> None:
    assert len(get_default_registry().hash(nest([], 5000))) == 64


def test_hasher_registry_hash_deeply_nested_data_equal_data_same_hash() -> None:
    registry = get_default_registry()
    assert registry.hash(nest([], 5000)) == registry.hash(nest([], 5000))


class MixedStr(str, Enum):  # noqa: UP042
    A = "a"


def test_hasher_registry_hash_enum_mixed_with_str_without_enum_hasher() -> None:
    registry = HasherRegistry({str: StrHasher()})
    assert registry.hash(MixedStr.A) == hash_with(StrHasher(), MixedStr.A, registry)


def test_hasher_registry_find_hasher_fallback_does_not_build_the_error_message() -> None:
    registry = HasherRegistry()
    registry.register_fallback(is_marked, StrHasher())
    with patch.object(TypeRegistry, "_not_registered_msg") as msg:
        registry.find_hasher(Marked)
    msg.assert_not_called()


def test_hasher_registry_unregister_fallback_removes_the_fallback() -> None:
    registry = HasherRegistry()
    registry.register_fallback(is_marked, StrHasher())
    registry.unregister_fallback(is_marked)
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        registry.find_hasher(Marked)


def test_hasher_registry_unregister_fallback_keeps_the_other_fallbacks() -> None:
    hasher = ReprHasher()
    registry = HasherRegistry()
    registry.register_fallback(is_marked, StrHasher())
    registry.register_fallback(is_other, hasher)
    registry.unregister_fallback(is_marked)
    assert registry.find_hasher(Other) is hasher


def test_hasher_registry_unregister_fallback_missing_raises() -> None:
    with pytest.raises(KeyError, match="No fallback is registered"):
        HasherRegistry().unregister_fallback(is_marked)


def test_hasher_registry_clear_removes_the_hashers() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    registry.clear()
    assert not registry.has_hasher(list)


def test_hasher_registry_clear_removes_the_fallbacks() -> None:
    registry = HasherRegistry()
    registry.register_fallback(is_marked, StrHasher())
    registry.clear()
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        registry.find_hasher(Marked)


def test_hasher_registry_equal_same_fallbacks() -> None:
    hasher = StrHasher()
    registry1 = HasherRegistry()
    registry1.register_fallback(is_marked, hasher)
    registry2 = HasherRegistry()
    registry2.register_fallback(is_marked, hasher)
    assert registry1.equal(registry2)


def test_hasher_registry_equal_different_number_of_fallbacks() -> None:
    registry1 = HasherRegistry()
    registry1.register_fallback(is_marked, StrHasher())
    assert not registry1.equal(HasherRegistry())


def test_hasher_registry_equal_different_predicates() -> None:
    hasher = StrHasher()
    registry1 = HasherRegistry()
    registry1.register_fallback(is_marked, hasher)
    registry2 = HasherRegistry()
    registry2.register_fallback(is_other, hasher)
    assert not registry1.equal(registry2)


def test_hasher_registry_equal_different_fallback_hashers() -> None:
    registry1 = HasherRegistry()
    registry1.register_fallback(is_marked, StrHasher())
    registry2 = HasherRegistry()
    registry2.register_fallback(is_marked, ReprHasher())
    assert not registry1.equal(registry2)


def test_hasher_registry_equal_different_order_of_fallbacks() -> None:
    hasher = StrHasher()
    registry1 = HasherRegistry()
    registry1.register_fallback(is_marked, hasher)
    registry1.register_fallback(is_other, hasher)
    registry2 = HasherRegistry()
    registry2.register_fallback(is_other, hasher)
    registry2.register_fallback(is_marked, hasher)
    assert not registry1.equal(registry2)


def test_hasher_registry_equal_different_hashers() -> None:
    hasher = StrHasher()
    assert not HasherRegistry({list: hasher}).equal(HasherRegistry({tuple: hasher}))


def test_hasher_registry_equal_different_type() -> None:
    assert not HasherRegistry().equal(object())


def test_hasher_registry_is_tracked_for_forks() -> None:
    registry = HasherRegistry()
    assert registry in fork._TRACKED


def test_hasher_registry_copy_has_the_same_fallbacks() -> None:
    registry = HasherRegistry()
    registry.register_fallback(lambda tp: tp is complex, StrHasher())
    assert len(registry.copy().fallbacks()) == 1


def test_hasher_registry_copy_register_fallback_does_not_change_the_original() -> None:
    registry = HasherRegistry()
    registry.copy().register_fallback(lambda tp: tp is complex, StrHasher())
    assert len(registry.fallbacks()) == 0
