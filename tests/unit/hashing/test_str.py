from __future__ import annotations

from typing import Any

import pytest

from coola.hashing import HasherRegistry, ReprHasher, StrHasher, tagged
from tests.unit.hashing.helpers import hash_with, leaf_hash


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: StrHasher()})


###############################
#     Tests for StrHasher     #
###############################


def test_str_hasher_repr() -> None:
    assert repr(StrHasher()) == "StrHasher()"


def test_str_hasher_str() -> None:
    assert str(StrHasher()) == "StrHasher()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            1234,
            "44cb4c7822fa217c5c06c223c5b0aacc8c5c77c768df58591c091cb1500d815e",
            id="int",
        ),
        pytest.param(
            0,
            "aa89b5cee92948ed5b79726bbe6ee622716714cb89d6a929b74e576c8f0ca378",
            id="int_zero",
        ),
        pytest.param(
            -1,
            "ed8e02ad8b7c52faf9cfc853e54b3458a897bac0fa4cba0adfaa7b3cfe625851",
            id="int_negative",
        ),
        pytest.param(
            1.5,
            "d2c7ed87af69f0e55c61e9c0ac0f7f542447e494741cb8abc05168c9af2915c0",
            id="float",
        ),
        pytest.param(
            float("inf"),
            "237ebc3ac1cb090ca1513f9c741b1dcefae94a966e596e108ad9ebf562e98fef",
            id="float_inf",
        ),
        pytest.param(
            True,
            "a1246624c11215e07f0539a1d087c88b9a5f9c80a51424fac2ae1653cb91c9d6",
            id="bool_true",
        ),
        pytest.param(
            False,
            "6070e87c818c9d5527cc5b6d746bc073c252fe0f0ba6289e0e412e855005cc7a",
            id="bool_false",
        ),
        pytest.param(
            complex(1, 2),
            "e8f0895d7008e0922301b7a8802593794f7ab11f685fb7341fd705ebad5f3375",
            id="complex",
        ),
        pytest.param(
            complex(0, -1),
            "a52d76525628375f316e05c2b99b63f6448b087bb37981bb13ad815f3c271604",
            id="complex_imaginary",
        ),
        pytest.param(
            None,
            "a0fa4225f8e9bd67116bcc6996d3f48b888caade696d53d26cbdf44179e8ea61",
            id="none",
        ),
        pytest.param(
            "hello",
            "c548d0f1ffb93319621f7004d9155e45eebfb2726ef58d4bf899944c6d1c0dc2",
            id="str",
        ),
        pytest.param(
            [1, 2, 3],
            "8692ee439fcd517d39ce2e404a01f2fcc108a78ab7097301bb01ed452c96348d",
            id="list",
        ),
        pytest.param(
            {"a": 1},
            "23f399661da961d8049800fdfcdead0e209e54ac9987f1a8f18eacd24ddc6064",
            id="dict",
        ),
        pytest.param(
            (1, 2, 3),
            "6bb67c32d839f45ca6c2a9420c0f5550225505eb9cabec25b67fd16ebd0f7df7",
            id="tuple",
        ),
        pytest.param(
            b"bytes",
            "1ab8454f86ef06979217c00d3887954f7b9488c8c5485e803b03398d8a35f6a7",
            id="bytes",
        ),
    ],
)
def test_str_hasher_hash_parametrized(data: Any, expected: str, registry: HasherRegistry) -> None:
    assert hash_with(StrHasher(), data, registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "c548d0f1ffb93319", id="16"),
        pytest.param(32, "c548d0f1ffb93319621f7004d9155e45", id="32"),
        pytest.param(
            64, "c548d0f1ffb93319621f7004d9155e45eebfb2726ef58d4bf899944c6d1c0dc2", id="64-default"
        ),
    ],
)
def test_str_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = hash_with(StrHasher(), "hello", registry, length=length)
    assert result == expected


@pytest.mark.parametrize("length", [16, 32, 64])
def test_str_hasher_hash_length_length_is_respected(length: int, registry: HasherRegistry) -> None:
    result = hash_with(StrHasher(), "hello", registry, length=length)
    assert len(result) == length


def test_str_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(hash_with(StrHasher(), "hello", registry), str)


def test_str_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = StrHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(hasher, "hello", registry)


def test_str_hasher_hash_different_values_different_hashes(registry: HasherRegistry) -> None:
    hasher = StrHasher()
    assert hash_with(hasher, 1, registry) != hash_with(hasher, 2, registry)


def test_str_hasher_hash_bool_and_int_different_hashes(registry: HasherRegistry) -> None:
    # str(True) == 'True' and str(1) == '1', so they must not collide.
    hasher = StrHasher()
    assert hash_with(hasher, True, registry) != hash_with(hasher, 1, registry)


def test_str_hasher_hash_uses_str_not_repr_equals_the_str_hash(registry: HasherRegistry) -> None:
    # For objects where str() and repr() differ, StrHasher must use str().
    # str("hello") == 'hello' while repr("hello") == "'hello'".
    class MyObj:
        def __str__(self) -> str:
            return "str_val"

        def __repr__(self) -> str:
            return "repr_val"

    obj = MyObj()
    hasher = StrHasher()
    tag = f"str:{MyObj.__module__}.{MyObj.__qualname__}"
    assert hash_with(hasher, obj, registry) == leaf_hash(tagged(tag, str(obj)))


def test_str_hasher_hash_str_differs_from_repr_hasher_for_strings(
    registry: HasherRegistry,
) -> None:
    # StrHasher hashes str("hello") = "hello".
    # ReprHasher hashes repr("hello") = "'hello'" (with quotes).
    # The two must produce different results for string inputs.

    hasher_str = StrHasher()
    hasher_repr = ReprHasher()
    assert hash_with(hasher_str, "hello", registry) != hash_with(hasher_repr, "hello", registry)


def test_str_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # StrHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = StrHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(
        hasher, "hello", registry, ignore_unhashable=True
    )


def test_str_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = StrHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(hasher, "hello", empty_registry)


def test_str_hasher_repr_with_tag() -> None:
    assert repr(StrHasher("UUID")) == "StrHasher(tag='UUID')"


def test_str_hasher_hash_explicit_tag(registry: HasherRegistry) -> None:
    assert hash_with(StrHasher("UUID"), 1234, registry) == leaf_hash(tagged("UUID", "1234"))


def test_str_hasher_hash_explicit_tag_applies_to_subclasses(registry: HasherRegistry) -> None:
    class MyStr(str):
        pass

    hasher = StrHasher("Name")
    assert hash_with(hasher, MyStr("a"), registry) == hash_with(hasher, "a", registry)


def test_str_hasher_hash_user_class_does_not_share_the_tag_of_a_builtin_name(
    registry: HasherRegistry,
) -> None:
    class int:  # noqa: A001, N801
        def __str__(self) -> str:
            return "1"

    assert hash_with(StrHasher(), int(), registry) != hash_with(StrHasher(), 1, registry)


def test_str_hasher_hash_user_classes_with_the_same_name_in_different_scopes_differ(
    registry: HasherRegistry,
) -> None:
    class Same:
        def __str__(self) -> str:
            return "x"

    def make() -> object:
        class Same:
            def __str__(self) -> str:
                return "x"

        return Same()

    assert hash_with(StrHasher(), Same(), registry) != hash_with(StrHasher(), make(), registry)


def test_str_hasher_encode_differs_from_the_repr_hasher() -> None:
    assert StrHasher().encode("a") != ReprHasher().encode("a")
