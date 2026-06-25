from __future__ import annotations

from typing import Any

import pytest

from coola.hashing import DefaultHasher, HasherRegistry


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: DefaultHasher()})


###################################
#     Tests for DefaultHasher     #
###################################


def test_default_hasher_repr() -> None:
    assert repr(DefaultHasher()) == "DefaultHasher()"


def test_default_hasher_str() -> None:
    assert str(DefaultHasher()) == "DefaultHasher()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            42,
            "2f0039e93a27221fcf657fb877a1d4f60307106113e885096cb44a461cd0afbf",
            id="int",
        ),
        pytest.param(
            None,
            "047380c75d8da0e84df15f8218632f31ca7058142d75fa4ee225aea3e8b1da82",
            id="none",
        ),
        pytest.param(
            True,
            "b37c53228410790b2e6d4ab6eb00deb4e1e9b47e2075100b120e0abd777d0020",
            id="bool",
        ),
        pytest.param(
            3.14,
            "ab14d2d4a60939ddc6de342ae33b8a8da25564787546cbdfd6cf29164863ed4a",
            id="float",
        ),
        pytest.param(
            [1, 2, 3],
            "07d1c573fdfe5074241be58b56a31220ff7a9cb3dd94f1cba650eca27c0a421e",
            id="list",
        ),
        pytest.param(
            "hello",
            "324dcf027dd4a30a932c441f365a25e86b173defa4b8e58948253471b81b72cf",
            id="str",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            "03f322fbef054cc34a4647309a2ae0891dd31f90f817284732ec741b2260bff7",
            id="dict",
        ),
    ],
)
def test_default_hasher_hash_parametrized(
    data: Any, expected: str, registry: HasherRegistry
) -> None:
    assert DefaultHasher().hash(data, registry=registry) == expected


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(2, id="min"),
        pytest.param(16, id="16"),
        pytest.param(32, id="32"),
        pytest.param(64, id="64-default"),
        pytest.param(
            128,
            id="max",
        ),
    ],
)
def test_default_hasher_hash_length(length: int, registry: HasherRegistry) -> None:
    result = DefaultHasher().hash(42, registry=registry, length=length)
    assert len(result) == length


def test_default_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(DefaultHasher().hash(42, registry=registry), str)


def test_default_hasher_hash_same_value_same_hash(registry: HasherRegistry) -> None:
    hasher = DefaultHasher()
    assert hasher.hash(42, registry=registry) == hasher.hash(42, registry=registry)


def test_default_hasher_hash_different_values_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = DefaultHasher()
    assert hasher.hash(1, registry=registry) != hasher.hash(2, registry=registry)


def test_default_hasher_hash_different_lengths_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = DefaultHasher()
    assert hasher.hash(42, registry=registry, length=16) != hasher.hash(
        42, registry=registry, length=32
    )


def test_default_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    # DefaultHasher should produce the same hash regardless of the registry contents,
    # since it does not recurse into nested structures.
    empty_registry = HasherRegistry({})
    hasher = DefaultHasher()
    assert hasher.hash([1, 2, 3], registry=registry) == hasher.hash(
        [1, 2, 3], registry=empty_registry
    )


def test_default_hasher_hash_nested_structure_uses_str_repr(
    registry: HasherRegistry,
) -> None:
    # Hashing is based on str(), so nested structures are not recursed into.
    data = {"a": [1, 2], "b": {"c": 3}}
    expected = DefaultHasher().hash(str(data), registry=registry)
    assert DefaultHasher().hash(data, registry=registry) == expected
