from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from coola.hashing import HasherRegistry, ReprHasher, StringHasher, get_default_registry
from coola.identifier.content import generate_content_id

if TYPE_CHECKING:
    from collections.abc import Generator

HEX_PATTERN = re.compile(r"^[0-9a-f]+$")


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the singleton registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


############################################
#     Tests for generate_content_id       #
############################################


def test_generate_content_id_returns_str() -> None:
    assert isinstance(generate_content_id({"key": "value"}), str)


def test_generate_content_id_default_length() -> None:
    assert len(generate_content_id({"key": "value"})) == 64


def test_generate_content_id_is_hex_string() -> None:
    assert HEX_PATTERN.match(generate_content_id({"key": "value"}))


def test_generate_content_id_custom_length() -> None:
    assert len(generate_content_id({"key": "value"}, length=16)) == 16


def test_generate_content_id_is_deterministic() -> None:
    data = {"source": "cats.txt", "page": 1}
    assert generate_content_id(data) == generate_content_id(data)


def test_generate_content_id_dict_key_order_independent() -> None:
    assert generate_content_id({"source": "cats.txt", "page": 1}) == generate_content_id(
        {"page": 1, "source": "cats.txt"}
    )


def test_generate_content_id_different_values_different_id() -> None:
    assert generate_content_id({"source": "cats.txt"}) != generate_content_id(
        {"source": "dogs.txt"}
    )


def test_generate_content_id_empty_dict() -> None:
    assert HEX_PATTERN.match(generate_content_id({}))


def test_generate_content_id_list_input() -> None:
    assert HEX_PATTERN.match(generate_content_id([1, 2, 3]))


def test_generate_content_id_tuple_and_list_same_id() -> None:
    assert generate_content_id([1, 2]) == generate_content_id((1, 2))


def test_generate_content_id_unhashable_type_raises_by_default() -> None:
    with pytest.raises(KeyError):
        generate_content_id(object())


def test_generate_content_id_ignore_unhashable_does_not_raise() -> None:
    assert HEX_PATTERN.match(generate_content_id(object(), ignore_unhashable=True))


def test_generate_content_id_uses_given_registry() -> None:
    registry = HasherRegistry({object: StringHasher()})
    assert generate_content_id("meow", registry=registry) == generate_content_id(
        "meow", registry=registry
    )


def test_generate_content_id_custom_registry_is_honored() -> None:
    custom_registry = HasherRegistry({object: ReprHasher()})
    assert generate_content_id("meow") != generate_content_id("meow", registry=custom_registry)


def test_generate_content_id_does_not_mutate_default_registry_when_custom_registry_given() -> None:
    custom_registry = HasherRegistry({object: StringHasher()})
    generate_content_id("meow", registry=custom_registry)
    assert not hasattr(get_default_registry, "_registry")
