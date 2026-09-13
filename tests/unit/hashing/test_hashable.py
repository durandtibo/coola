from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from coola.hashing import (
    HashableHasher,
    HasherRegistry,
    SupportsHash,
    get_default_registry,
)


class MyObj:
    r"""An object that implements the ``SupportsHash`` protocol by
    delegating to the registry for its inner value."""

    def __init__(self, value: object) -> None:
        self._value = value

    def hash(
        self,
        registry: HasherRegistry | None = None,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        if registry is None:
            registry = get_default_registry()
        return registry.hash(self._value, length=length, ignore_unhashable=ignore_unhashable)


class Unhashable:
    r"""A type with no registered hasher."""


@pytest.fixture
def registry() -> HasherRegistry:
    return get_default_registry()


####################################
#     Tests for HashableHasher     #
####################################


def test_hashable_hasher_repr() -> None:
    assert repr(HashableHasher()) == "HashableHasher()"


def test_hashable_hasher_str() -> None:
    assert str(HashableHasher()) == "HashableHasher()"


def test_hashable_hasher_hash_matches_object_hash(registry: HasherRegistry) -> None:
    obj = MyObj(42)
    assert HashableHasher().hash(obj, registry=registry) == obj.hash(registry=registry)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            42, "2f0039e93a27221fcf657fb877a1d4f60307106113e885096cb44a461cd0afbf", id="int"
        ),
        pytest.param(
            "abc", "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319", id="str"
        ),
        pytest.param([1, 2, 3], None, id="list"),
    ],
)
def test_hashable_hasher_hash_known_values(
    value: Any, expected: str | None, registry: HasherRegistry
) -> None:
    result = HashableHasher().hash(MyObj(value), registry=registry)
    if expected is not None:
        assert result == expected
    assert isinstance(result, str)
    assert len(result) == 64


def test_hashable_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(HashableHasher().hash(MyObj(42), registry=registry), str)


def test_hashable_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = HashableHasher()
    assert hasher.hash(MyObj(42), registry=registry) == hasher.hash(MyObj(42), registry=registry)


def test_hashable_hasher_hash_different_values_different_hashes(registry: HasherRegistry) -> None:
    hasher = HashableHasher()
    assert hasher.hash(MyObj(1), registry=registry) != hasher.hash(MyObj(2), registry=registry)


def test_hashable_hasher_hash_different_types_different_hashes(registry: HasherRegistry) -> None:
    # Two objects with the same nested value but different concrete types
    # are free to hash differently, since HashableHasher just
    # delegates to whatever the object's own hash() implements. Here a
    # class whose hash() mixes in its class name demonstrates that
    # HashableHasher does not collapse type identity on its own - that
    # guarantee is entirely up to the delegate's ``hash`` method.
    class OtherObj(MyObj):
        def hash(
            self,
            registry: HasherRegistry | None = None,
            length: int = 64,
            ignore_unhashable: bool = False,
        ) -> str:
            if registry is None:
                registry = get_default_registry()
            return registry.hash(
                (type(self).__qualname__, self._value),
                length=length,
                ignore_unhashable=ignore_unhashable,
            )

    hasher = HashableHasher()
    assert hasher.hash(MyObj(1), registry=registry) != hasher.hash(OtherObj(1), registry=registry)


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "57b43cf02666687a", id="16"),
        pytest.param(32, "755345b7ef54b0d592c17e04737ab3d9", id="32"),
        pytest.param(
            64, "2f0039e93a27221fcf657fb877a1d4f60307106113e885096cb44a461cd0afbf", id="64-default"
        ),
    ],
)
def test_hashable_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = HashableHasher().hash(MyObj(42), registry=registry, length=length)
    assert result == expected
    assert len(result) == length


def test_hashable_hasher_hash_default_length_is_64(registry: HasherRegistry) -> None:
    assert len(HashableHasher().hash(MyObj(42), registry=registry)) == 64


def test_hashable_hasher_hash_forwards_registry_argument() -> None:
    obj = Mock(spec_set=SupportsHash)
    obj.hash.return_value = "cafe"
    registry = HasherRegistry()

    result = HashableHasher().hash(obj, registry=registry)

    obj.hash.assert_called_once_with(registry=registry, length=64, ignore_unhashable=False)
    assert result == "cafe"


@pytest.mark.parametrize("length", [2, 16, 32, 64, 128])
def test_hashable_hasher_hash_forwards_length_argument(
    length: int, registry: HasherRegistry
) -> None:
    obj = Mock(spec_set=SupportsHash)
    obj.hash.return_value = "cafe"

    HashableHasher().hash(obj, registry=registry, length=length)

    obj.hash.assert_called_once_with(registry=registry, length=length, ignore_unhashable=False)


@pytest.mark.parametrize("ignore_unhashable", [True, False])
def test_hashable_hasher_hash_forwards_ignore_unhashable_argument(
    ignore_unhashable: bool, registry: HasherRegistry
) -> None:
    obj = Mock(spec_set=SupportsHash)
    obj.hash.return_value = "cafe"

    HashableHasher().hash(obj, registry=registry, ignore_unhashable=ignore_unhashable)

    obj.hash.assert_called_once_with(
        registry=registry, length=64, ignore_unhashable=ignore_unhashable
    )


def test_hashable_hasher_hash_propagates_unhashable_nested_value() -> None:
    empty_registry = HasherRegistry()
    obj = MyObj(Unhashable())
    with pytest.raises(KeyError, match=r"is not registered"):
        HashableHasher().hash(obj, registry=empty_registry)


def test_hashable_hasher_hash_ignore_unhashable_nested_value_returns_str() -> None:
    empty_registry = HasherRegistry()
    obj = MyObj(Unhashable())
    result = HashableHasher().hash(obj, registry=empty_registry, ignore_unhashable=True)
    assert isinstance(result, str)


def test_hashable_hasher_hash_propagates_object_own_errors() -> None:
    class Failing:
        def hash(
            self,
            registry: HasherRegistry | None = None,  # noqa: ARG002
            length: int = 64,  # noqa: ARG002
            ignore_unhashable: bool = False,  # noqa: ARG002
        ) -> str:
            msg = "boom"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"):
        HashableHasher().hash(Failing(), registry=HasherRegistry())


def test_hashable_hasher_hash_missing_hash_method_raises_error(registry: HasherRegistry) -> None:
    class NotHashable:
        pass

    with pytest.raises(AttributeError):
        HashableHasher().hash(NotHashable(), registry=registry)  # type: ignore[arg-type]


def test_hashable_hasher_hash_non_callable_hash_attribute_raises_error(
    registry: HasherRegistry,
) -> None:
    class NotCallableHash:
        hash = "not-a-method"

    with pytest.raises(TypeError):
        HashableHasher().hash(NotCallableHash(), registry=registry)  # type: ignore[arg-type]


def test_hashable_hasher_registered_in_registry_is_used_via_dispatch() -> None:
    # Verify end-to-end dispatch: registering HashableHasher for a
    # custom type lets HasherRegistry.hash() resolve and delegate to it
    # automatically, the same way it would for a built-in type.
    registry = HasherRegistry(
        {MyObj: HashableHasher(), int: get_default_registry().find_hasher(int)}
    )
    obj = MyObj(42)
    assert registry.hash(obj) == obj.hash(registry=registry)


##################################
#     Tests for SupportsHash     #
##################################


def test_supports_hash_isinstance_true_for_matching_object() -> None:
    assert isinstance(MyObj(42), SupportsHash)


def test_supports_hash_isinstance_false_for_missing_method() -> None:
    class NotHashable:
        pass

    assert not isinstance(NotHashable(), SupportsHash)


def test_supports_hash_protocol_method_body_is_a_stub() -> None:
    # The Protocol's own `hash` method is never meant to run - it only
    # declares the interface implementers must satisfy - but calling it
    # directly (unbound, on a compatible instance) exercises its `...`
    # body for coverage and confirms it is a no-op that returns `None`.
    assert SupportsHash.hash(MyObj(42)) is None  # type: ignore[reportAbstractUsage]


def test_supports_hash_isinstance_true_regardless_of_signature() -> None:
    # isinstance checks against a runtime_checkable Protocol only look
    # for the presence of the attribute, not its signature.
    class WrongSignature:
        def hash(self) -> str:
            return "x"

    assert isinstance(WrongSignature(), SupportsHash)
