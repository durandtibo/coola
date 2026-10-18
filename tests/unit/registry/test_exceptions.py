from __future__ import annotations

import pytest

from coola.registry import AlreadyRegisteredError, Registry, TypeNotRegisteredError, TypeRegistry

############################################
#     Tests for AlreadyRegisteredError     #
############################################


def test_already_registered_error_is_value_error_and_runtime_error() -> None:
    assert issubclass(AlreadyRegisteredError, ValueError)
    assert issubclass(AlreadyRegisteredError, RuntimeError)


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


def test_already_registered_error_can_still_be_caught_as_runtime_error() -> None:
    registry = Registry[str, int]({"key": 1})
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register("key", 2)


def test_already_registered_error_is_not_a_key_error() -> None:
    # A missing key raises ``TypeNotRegisteredError`` (a ``KeyError``): the two must not be mixed.
    assert not issubclass(AlreadyRegisteredError, KeyError)
    assert issubclass(TypeNotRegisteredError, KeyError)
