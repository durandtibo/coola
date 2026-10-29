from __future__ import annotations

from collections import UserList
from collections.abc import Sequence

import pytest

from coola.registry import (
    AlreadyRegisteredError,
    KeyNotRegisteredError,
    Registry,
    TypeNotRegisteredError,
    TypeRegistry,
)

############################################
#     Tests for AlreadyRegisteredError     #
############################################


def test_already_registered_error_is_value_error() -> None:
    assert issubclass(AlreadyRegisteredError, ValueError)


def test_already_registered_error_is_not_a_runtime_error() -> None:
    assert not issubclass(AlreadyRegisteredError, RuntimeError)


def test_already_registered_error_str_is_not_quoted() -> None:
    assert str(AlreadyRegisteredError("A value is already registered")) == (
        "A value is already registered"
    )


def test_registry_register_duplicate_raises_already_registered_error() -> None:
    registry = Registry[str, int]({"key": 1})
    with pytest.raises(AlreadyRegisteredError, match="already registered"):
        registry.register("key", 2)


def test_registry_register_many_duplicate_raises_already_registered_error() -> None:
    registry = Registry[str, int]({"key": 1})
    with pytest.raises(AlreadyRegisteredError, match="already registered"):
        registry.register_many({"key": 2, "other": 3})


def test_type_registry_register_duplicate_raises_already_registered_error() -> None:
    registry = TypeRegistry[str]({int: "int"})
    with pytest.raises(AlreadyRegisteredError, match="already registered"):
        registry.register(int, "other")


def test_already_registered_error_is_not_a_key_error() -> None:
    # A missing key raises ``TypeNotRegisteredError`` (a ``KeyError``): the two must not be mixed.
    assert not issubclass(AlreadyRegisteredError, KeyError)


def test_type_not_registered_error_is_a_key_error() -> None:
    assert issubclass(TypeNotRegisteredError, KeyError)


###########################################
#     Tests for KeyNotRegisteredError     #
###########################################


def test_key_not_registered_error_is_key_error() -> None:
    assert issubclass(KeyNotRegisteredError, KeyError)


def test_key_not_registered_error_is_lookup_error() -> None:
    assert issubclass(KeyNotRegisteredError, LookupError)


def test_type_not_registered_error_is_key_not_registered_error() -> None:
    assert issubclass(TypeNotRegisteredError, KeyNotRegisteredError)


def test_key_not_registered_error_str_is_not_quoted() -> None:
    assert str(KeyNotRegisteredError("Key 'a'\nnot registered")) == "Key 'a'\nnot registered"


def test_registry_getitem_missing_raises_key_not_registered_error() -> None:
    with pytest.raises(KeyNotRegisteredError, match="Key 'missing' is not registered"):
        Registry[str, int]()["missing"]


def test_registry_getitem_missing_does_not_raise_type_not_registered_error() -> None:
    with pytest.raises(KeyNotRegisteredError, match="Key 'missing' is not registered") as info:
        Registry[str, int]()["missing"]
    assert not isinstance(info.value, TypeNotRegisteredError)


def test_registry_unregister_missing_raises_key_not_registered_error() -> None:
    with pytest.raises(KeyNotRegisteredError, match="Key 'missing' is not registered"):
        Registry[str, int]().unregister("missing")


def test_registry_unregister_missing_does_not_raise_type_not_registered_error() -> None:
    with pytest.raises(KeyNotRegisteredError, match="Key 'missing' is not registered") as info:
        Registry[str, int]().unregister("missing")
    assert not isinstance(info.value, TypeNotRegisteredError)


def test_registry_missing_key_can_still_be_caught_as_key_error() -> None:
    with pytest.raises(KeyError, match="is not registered"):
        Registry[str, int]()["missing"]


def test_type_registry_getitem_missing_raises_type_not_registered_error() -> None:
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        TypeRegistry[str]()[int]


def test_type_registry_unregister_missing_raises_type_not_registered_error() -> None:
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        TypeRegistry[str]().unregister(int)


def test_type_registry_missing_type_can_be_caught_as_key_not_registered_error() -> None:
    with pytest.raises(KeyNotRegisteredError, match="is not registered"):
        TypeRegistry[str]().resolve(int)


def test_type_registry_resolve_does_not_use_virtual_subclasses() -> None:
    # ``str`` is a virtual subclass of ``Sequence``: it must not be dispatched to the
    # handler registered for ``Sequence``.
    registry = TypeRegistry[str]({Sequence: "sequence", object: "object"})
    assert registry.resolve(str) == "object"


def test_type_registry_resolve_uses_real_subclasses() -> None:
    registry = TypeRegistry[str]({Sequence: "sequence", object: "object"})
    assert registry.resolve(UserList) == "sequence"  # a real subclass of Sequence
