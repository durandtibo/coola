from __future__ import annotations

import re

import pytest

from coola.hashing import HasherRegistry, ReprHasher, StringHasher
from coola.hashing import interface as hashing_interface
from coola.identifier.content import generate_stable_content_id

HEX_PATTERN = re.compile(r"^[0-9a-f]+$")


###################################################
#     Tests for generate_stable_content_id       #
###################################################


def test_generate_stable_content_id_returns_str() -> None:
    assert isinstance(generate_stable_content_id({"key": "value"}), str)


def test_generate_stable_content_id_default_length() -> None:
    assert len(generate_stable_content_id({"key": "value"})) == 64


def test_generate_stable_content_id_is_hex_string() -> None:
    assert HEX_PATTERN.match(generate_stable_content_id({"key": "value"}))


def test_generate_stable_content_id_custom_length() -> None:
    assert len(generate_stable_content_id({"key": "value"}, length=16)) == 16


def test_generate_stable_content_id_is_deterministic() -> None:
    data = {"source": "cats.txt", "page": 1}
    assert generate_stable_content_id(data) == generate_stable_content_id(data)


def test_generate_stable_content_id_dict_key_order_independent() -> None:
    assert generate_stable_content_id(
        {"source": "cats.txt", "page": 1}
    ) == generate_stable_content_id({"page": 1, "source": "cats.txt"})


def test_generate_stable_content_id_different_values_different_id() -> None:
    assert generate_stable_content_id({"source": "cats.txt"}) != generate_stable_content_id(
        {"source": "dogs.txt"}
    )


def test_generate_stable_content_id_empty_dict() -> None:
    assert HEX_PATTERN.match(generate_stable_content_id({}))


def test_generate_stable_content_id_list_input() -> None:
    assert HEX_PATTERN.match(generate_stable_content_id([1, 2, 3]))


def test_generate_stable_content_id_tuple_and_list_same_id() -> None:
    assert generate_stable_content_id([1, 2]) == generate_stable_content_id((1, 2))


def test_generate_stable_content_id_unhashable_type_raises_by_default() -> None:
    with pytest.raises(KeyError):
        generate_stable_content_id(object())


def test_generate_stable_content_id_ignore_unhashable_does_not_raise() -> None:
    assert HEX_PATTERN.match(generate_stable_content_id(object(), ignore_unhashable=True))


def test_generate_stable_content_id_uses_given_registry() -> None:
    registry = HasherRegistry({object: StringHasher()})
    assert generate_stable_content_id("meow", registry=registry) == generate_stable_content_id(
        "meow", registry=registry
    )


def test_generate_stable_content_id_custom_registry_is_honored() -> None:
    custom_registry = HasherRegistry({object: ReprHasher()})
    assert generate_stable_content_id("meow") != generate_stable_content_id(
        "meow", registry=custom_registry
    )


def test_generate_stable_content_id_does_not_mutate_default_registry_when_custom_registry_given() -> (
    None
):
    custom_registry = HasherRegistry({object: StringHasher()})
    generate_stable_content_id("meow", registry=custom_registry)
    assert hashing_interface._default_registry._instance is None
