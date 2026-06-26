from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from coola.hashing import DatetimeHasher, HasherRegistry


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: DatetimeHasher()})


####################################
#     Tests for DatetimeHasher     #
####################################


def test_datetime_hasher_repr() -> None:
    assert repr(DatetimeHasher()) == "DatetimeHasher()"


def test_datetime_hasher_str() -> None:
    assert str(DatetimeHasher()) == "DatetimeHasher()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            date(2021, 1, 1),
            "f2b4c6a9941206bb6fc3b4b9c1104d8c05264985c009e2e1c7c840aaeda00dac",
            id="date",
        ),
        pytest.param(
            date(2024, 12, 31),
            "1dc2d8ecaf7d9164fd224e3ff8637a7819c086ab506e02c334eea47ebb2f03db",
            id="date_end_of_year",
        ),
        pytest.param(
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "7b9000123bc9220f9759ddc4ae7e7780c16935c8ed6d9417b82c41500ebb3967",
            id="datetime_midnight",
        ),
        pytest.param(
            datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            "676dda2885fb7510aa8be2e0aff297d76cdec9661ee5f4383f964988edf25840",
            id="datetime_end_of_year",
        ),
        pytest.param(
            datetime(2021, 1, 1, 12, 30, 45, 123456, tzinfo=timezone.utc),
            "13f82bb8ce7a9ac5d81d8c0570e8b51b547ec83aac578142a84cc6d51cd0aee7",
            id="datetime_with_microseconds",
        ),
    ],
)
def test_datetime_hasher_hash_parametrized(
    data: date | datetime, expected: str, registry: HasherRegistry
) -> None:
    assert DatetimeHasher().hash(data, registry=registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "7a0778ccca4b20e9", id="16"),
        pytest.param(32, "1e9b4678d322a95de544bd97aa696363", id="32"),
        pytest.param(
            64, "f2b4c6a9941206bb6fc3b4b9c1104d8c05264985c009e2e1c7c840aaeda00dac", id="64-default"
        ),
    ],
)
def test_datetime_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = DatetimeHasher().hash(date(2021, 1, 1), registry=registry, length=length)
    assert result == expected
    assert len(result) == length


def test_datetime_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(DatetimeHasher().hash(date(2021, 1, 1), registry=registry), str)


def test_datetime_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    assert hasher.hash(date(2021, 1, 1), registry=registry) == hasher.hash(
        date(2021, 1, 1), registry=registry
    )


def test_datetime_hasher_hash_different_dates_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = DatetimeHasher()
    assert hasher.hash(date(2021, 1, 1), registry=registry) != hasher.hash(
        date(2021, 1, 2), registry=registry
    )


def test_datetime_hasher_hash_date_and_datetime_different_hashes(
    registry: HasherRegistry,
) -> None:
    # date(2021, 1, 1) -> '2021-01-01'
    # datetime(2021, 1, 1, 0, 0, 0) -> '2021-01-01T00:00:00'
    # isoformat() distinguishes them, so their hashes must differ.
    hasher = DatetimeHasher()
    assert hasher.hash(date(2021, 1, 1), registry=registry) != hasher.hash(
        datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc), registry=registry
    )


def test_datetime_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = DatetimeHasher()
    assert hasher.hash(date(2021, 1, 1), registry=registry) == hasher.hash(
        date(2021, 1, 1), registry=empty_registry
    )
