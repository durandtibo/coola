from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone

import pytest

from coola.hashing import DatetimeHasher, HasherRegistry, get_default_registry, hash_object
from tests.unit.hashing.helpers import hash_with


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
            "2f3e15e2b7f163a8206081e69471fa67b41bc72f009b31216f6f1872d9c22a5b",
            id="date",
        ),
        pytest.param(
            date(2024, 12, 31),
            "8ab7651876692f67abdc33911cd94bca9c2543e879196de2212d33451f79caa5",
            id="date_end_of_year",
        ),
        pytest.param(
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC),
            "2920da84f6bb2ba85ca45f0cd0b90c8b4a269966291676b9eab483d001e9e47d",
            id="datetime_midnight",
        ),
        pytest.param(
            datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC),
            "e2f69b579ca85471b042c8872a93177b258f17ad9b91747f0d5a5c7da629d34a",
            id="datetime_end_of_year",
        ),
        pytest.param(
            datetime(2021, 1, 1, 12, 30, 45, 123456, tzinfo=UTC),
            "95e351961dcf3179d17992dbd4cbcaf02514938b994051b08cc7ade49b03e634",
            id="datetime_with_microseconds",
        ),
    ],
)
def test_datetime_hasher_hash_parametrized(
    data: date | datetime, expected: str, registry: HasherRegistry
) -> None:
    assert hash_with(DatetimeHasher(), data, registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "2f3e15e2b7f163a8", id="16"),
        pytest.param(32, "2f3e15e2b7f163a8206081e69471fa67", id="32"),
        pytest.param(
            64, "2f3e15e2b7f163a8206081e69471fa67b41bc72f009b31216f6f1872d9c22a5b", id="64-default"
        ),
    ],
)
def test_datetime_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = hash_with(DatetimeHasher(), date(2021, 1, 1), registry, length=length)
    assert result == expected


@pytest.mark.parametrize("length", [16, 32, 64])
def test_datetime_hasher_hash_length_is_respected(length: int, registry: HasherRegistry) -> None:
    result = hash_with(DatetimeHasher(), date(2021, 1, 1), registry, length=length)
    assert len(result) == length


def test_datetime_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(hash_with(DatetimeHasher(), date(2021, 1, 1), registry), str)


def test_datetime_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    assert hash_with(hasher, date(2021, 1, 1), registry) == hash_with(
        hasher, date(2021, 1, 1), registry
    )


def test_datetime_hasher_hash_different_dates_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = DatetimeHasher()
    assert hash_with(hasher, date(2021, 1, 1), registry) != hash_with(
        hasher, date(2021, 1, 2), registry
    )


def test_datetime_hasher_hash_date_and_datetime_different_hashes(
    registry: HasherRegistry,
) -> None:
    # date(2021, 1, 1) -> '2021-01-01'
    # datetime(2021, 1, 1, 0, 0, 0) -> '2021-01-01T00:00:00'
    # isoformat() distinguishes them, so their hashes must differ.
    hasher = DatetimeHasher()
    assert hash_with(hasher, date(2021, 1, 1), registry) != hash_with(
        hasher, datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC), registry
    )


def test_datetime_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # DatetimeHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = DatetimeHasher()
    assert hash_with(hasher, date(2021, 1, 1), registry) == hash_with(
        hasher, date(2021, 1, 1), registry, ignore_unhashable=True
    )


def test_datetime_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = DatetimeHasher()
    assert hash_with(hasher, date(2021, 1, 1), registry) == hash_with(
        hasher, date(2021, 1, 1), empty_registry
    )


def test_datetime_hasher_hash_time_same_time(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    assert hash_with(hasher, time(12, 30), registry) == hash_with(hasher, time(12, 30), registry)


def test_datetime_hasher_hash_time_different_time(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    assert hash_with(hasher, time(12, 30), registry) != hash_with(hasher, time(12, 31), registry)


def test_datetime_hasher_hash_time_differs_from_datetime(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    assert hash_with(hasher, time(0, 0), registry) != hash_with(
        hasher,
        datetime(2024, 1, 1, tzinfo=UTC),
        registry=registry,
    )


def test_datetime_hasher_hash_same_instant_different_zone_is_equal_instants_are_equal() -> None:
    utc = datetime(2024, 1, 1, 12, tzinfo=UTC)
    paris = utc.astimezone(timezone(timedelta(hours=1)))
    assert utc == paris


def test_datetime_hasher_hash_same_instant_different_zone_is_equal_hash(
    registry: HasherRegistry,
) -> None:
    utc = datetime(2024, 1, 1, 12, tzinfo=UTC)
    paris = utc.astimezone(timezone(timedelta(hours=1)))
    assert hash_with(DatetimeHasher(), utc, registry) == hash_with(
        DatetimeHasher(), paris, registry
    )


def test_datetime_hasher_hash_different_instants_differ(registry: HasherRegistry) -> None:
    first = datetime(2024, 1, 1, 12, tzinfo=UTC)
    second = datetime(2024, 1, 1, 12, tzinfo=timezone(timedelta(hours=1)))
    assert hash_with(DatetimeHasher(), first, registry) != hash_with(
        DatetimeHasher(), second, registry
    )


def test_default_registry_hashes_timedelta_and_timezone_equal_timedeltas() -> None:
    registry = get_default_registry()
    assert registry.hash(timedelta(days=1)) == registry.hash(timedelta(hours=24))


def test_default_registry_hashes_timedelta_and_timezone_different_timedeltas() -> None:
    registry = get_default_registry()
    assert registry.hash(timedelta(days=1)) != registry.hash(timedelta(days=2))


def test_default_registry_hashes_timedelta_and_timezone_different_timezones() -> None:
    registry = get_default_registry()
    assert registry.hash(UTC) != registry.hash(timezone(timedelta(hours=1)))


@pytest.mark.parametrize(
    "value",
    [
        datetime.min.replace(tzinfo=timezone(timedelta(hours=1))),
        datetime.max.replace(tzinfo=timezone(timedelta(hours=-1))),
    ],
)
def test_datetime_hasher_hash_aware_datetime_near_limits(
    value: datetime, registry: HasherRegistry
) -> None:
    hasher = DatetimeHasher()
    assert hash_with(hasher, value, registry) == hash_with(hasher, value, registry)


def test_datetime_min_cannot_be_converted_to_utc() -> None:
    # Guards the premise of the tests below: converting to UTC overflows.
    with pytest.raises(OverflowError, match="date value out of range"):
        datetime.min.replace(tzinfo=timezone(timedelta(hours=1))).astimezone(UTC)


def test_datetime_max_cannot_be_converted_to_utc() -> None:
    with pytest.raises(OverflowError, match="date value out of range"):
        datetime.max.replace(tzinfo=timezone(timedelta(hours=-1))).astimezone(UTC)


def test_datetime_hasher_hash_aware_datetime_near_limits_different_values(
    registry: HasherRegistry,
) -> None:
    hasher = DatetimeHasher()
    plus_one = timezone(timedelta(hours=1))
    first = datetime.min.replace(tzinfo=plus_one)
    second = first + timedelta(microseconds=1)
    assert hash_with(hasher, first, registry) != hash_with(hasher, second, registry)


def test_datetime_hasher_hash_aware_datetime_near_limits_different_offsets(
    registry: HasherRegistry,
) -> None:
    hasher = DatetimeHasher()
    first = datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))
    second = datetime.min.replace(tzinfo=timezone(timedelta(hours=2)))
    assert hash_with(hasher, first, registry) != hash_with(hasher, second, registry)


def test_datetime_hasher_hash_aware_datetime_near_limits_differs_from_naive(
    registry: HasherRegistry,
) -> None:
    hasher = DatetimeHasher()
    aware = datetime.max.replace(tzinfo=timezone(timedelta(hours=-1)))
    assert hash_with(hasher, aware, registry) != hash_with(
        hasher, aware.replace(tzinfo=None), registry
    )


@pytest.mark.parametrize("length", [2, 16, 128])
def test_datetime_hasher_hash_aware_datetime_near_limits_length(
    length: int, registry: HasherRegistry
) -> None:
    value = datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))
    assert len(hash_with(DatetimeHasher(), value, registry, length=length)) == length


def test_default_registry_hashes_aware_datetime_near_limits() -> None:
    registry = get_default_registry()
    value = datetime.max.replace(tzinfo=timezone(timedelta(hours=-1)))
    assert registry.hash(value) == registry.hash(value)


def test_default_registry_hashes_nested_aware_datetime_near_limits() -> None:
    registry = get_default_registry()
    value = datetime.max.replace(tzinfo=timezone(timedelta(hours=-1)))
    assert registry.hash([value, {"k": value}])


def test_datetime_hasher_hash_equal_instants_away_from_limits_still_equal(
    registry: HasherRegistry,
) -> None:
    hasher = DatetimeHasher()
    first = datetime(2024, 1, 1, 12, tzinfo=UTC)
    second = datetime(2024, 1, 1, 13, tzinfo=timezone(timedelta(hours=1)))
    assert hash_with(hasher, first, registry) == hash_with(hasher, second, registry)


def test_datetime_hasher_hash_aware_time_equal_instants(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    first = time(12, tzinfo=UTC)
    second = time(13, tzinfo=timezone(timedelta(hours=1)))
    assert hash_with(hasher, first, registry) == hash_with(hasher, second, registry)


def test_datetime_hasher_hash_aware_time_different_instants(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    first = time(12, tzinfo=UTC)
    second = time(12, tzinfo=timezone(timedelta(hours=1)))
    assert hash_with(hasher, first, registry) != hash_with(hasher, second, registry)


def test_datetime_hasher_hash_aware_time_differs_from_naive(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    assert hash_with(hasher, time(12, tzinfo=UTC), registry) != hash_with(
        hasher, time(12), registry
    )


def test_datetime_hasher_hash_aware_time_microseconds(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    assert hash_with(hasher, time(12, 0, 0, 1, tzinfo=UTC), registry) != hash_with(
        hasher, time(12, tzinfo=UTC), registry
    )


def test_datetime_hasher_hash_aware_time_negative_offset(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    first = time(12, tzinfo=timezone(timedelta(hours=-5)))
    second = time(17, tzinfo=UTC)
    assert hash_with(hasher, first, registry) == hash_with(hasher, second, registry)


def test_datetime_hasher_hash_aware_time_before_midnight_in_utc(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    first = time(1, tzinfo=timezone(timedelta(hours=2)))
    second = time(23, tzinfo=UTC)
    assert hash_with(hasher, first, registry) != hash_with(hasher, second, registry)


def test_hash_object_equal_aware_times_have_the_same_hash() -> None:
    first = time(12, tzinfo=UTC)
    second = time(13, tzinfo=timezone(timedelta(hours=1)))
    assert first == second
    assert hash_object(first) == hash_object(second)


def test_datetime_hasher_hash_datetime_subclass_hashes_like_datetime(
    registry: HasherRegistry,
) -> None:
    class MyDatetime(datetime):
        pass

    hasher = DatetimeHasher()
    assert hash_with(hasher, MyDatetime(2024, 1, 2, tzinfo=UTC), registry) == hash_with(
        hasher, datetime(2024, 1, 2, tzinfo=UTC), registry
    )


def test_datetime_hasher_hash_date_differs_from_datetime(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    assert hash_with(hasher, date(2024, 1, 2), registry) != hash_with(
        hasher, datetime(2024, 1, 2, tzinfo=UTC), registry
    )
