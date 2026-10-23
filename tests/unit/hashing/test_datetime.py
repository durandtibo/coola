from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone

import pytest

from coola.hashing import DatetimeHasher, HasherRegistry, get_default_registry


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
            "1d7ccf2f2d2fa168388daafdef0be5efb6e74b780c83a649af3368a5a23911f1",
            id="date",
        ),
        pytest.param(
            date(2024, 12, 31),
            "ee4eff7d44d05ec8419af555ca241c8e81e66446f41de6acf0c7bd9da870541c",
            id="date_end_of_year",
        ),
        pytest.param(
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC),
            "1cc1275c7d33a7dc33384372f2b37341544cd099de233c9bcaa753fe08ea6621",
            id="datetime_midnight",
        ),
        pytest.param(
            datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC),
            "38eaaede7c78cc1b1173766b115890fe39b6cc532392863a6f6c0c8e3f36e84c",
            id="datetime_end_of_year",
        ),
        pytest.param(
            datetime(2021, 1, 1, 12, 30, 45, 123456, tzinfo=UTC),
            "4fa08fb0f0c7815492ed08e490f0b184f8d75a43eeba9023c7e072009facdd5d",
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
        pytest.param(16, "eab4ea3c8f815b6b", id="16"),
        pytest.param(32, "e8490efa4b99a2e38064fa10c867a0d4", id="32"),
        pytest.param(
            64, "1d7ccf2f2d2fa168388daafdef0be5efb6e74b780c83a649af3368a5a23911f1", id="64-default"
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
        datetime(2021, 1, 1, 0, 0, 0, tzinfo=UTC), registry=registry
    )


def test_datetime_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # DatetimeHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = DatetimeHasher()
    assert hasher.hash(date(2021, 1, 1), registry=registry) == hasher.hash(
        date(2021, 1, 1), registry=registry, ignore_unhashable=True
    )


def test_datetime_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = DatetimeHasher()
    assert hasher.hash(date(2021, 1, 1), registry=registry) == hasher.hash(
        date(2021, 1, 1), registry=empty_registry
    )


def test_datetime_hasher_hash_time(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    assert hasher.hash(time(12, 30), registry=registry) == hasher.hash(
        time(12, 30), registry=registry
    )
    assert hasher.hash(time(12, 30), registry=registry) != hasher.hash(
        time(12, 31), registry=registry
    )


def test_datetime_hasher_hash_time_differs_from_datetime(registry: HasherRegistry) -> None:
    hasher = DatetimeHasher()
    assert hasher.hash(time(0, 0), registry=registry) != hasher.hash(
        datetime(2024, 1, 1),  # noqa: DTZ001
        registry=registry,
    )


def test_datetime_hasher_hash_same_instant_different_zone_is_equal(
    registry: HasherRegistry,
) -> None:
    utc = datetime(2024, 1, 1, 12, tzinfo=UTC)
    paris = utc.astimezone(timezone(timedelta(hours=1)))
    assert utc == paris
    assert DatetimeHasher().hash(utc, registry=registry) == DatetimeHasher().hash(
        paris, registry=registry
    )


def test_datetime_hasher_hash_different_instants_differ(registry: HasherRegistry) -> None:
    first = datetime(2024, 1, 1, 12, tzinfo=UTC)
    second = datetime(2024, 1, 1, 12, tzinfo=timezone(timedelta(hours=1)))
    assert DatetimeHasher().hash(first, registry=registry) != DatetimeHasher().hash(
        second, registry=registry
    )


def test_default_registry_hashes_timedelta_and_timezone() -> None:
    registry = get_default_registry()
    assert registry.hash(timedelta(days=1)) == registry.hash(timedelta(hours=24))
    assert registry.hash(timedelta(days=1)) != registry.hash(timedelta(days=2))
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
    assert hasher.hash(value, registry=registry) == hasher.hash(value, registry=registry)


def test_datetime_near_limits_cannot_be_converted_to_utc() -> None:
    # Guards the premise of the tests below: converting to UTC overflows.
    with pytest.raises(OverflowError):
        datetime.min.replace(tzinfo=timezone(timedelta(hours=1))).astimezone(UTC)
    with pytest.raises(OverflowError):
        datetime.max.replace(tzinfo=timezone(timedelta(hours=-1))).astimezone(UTC)


def test_datetime_hasher_hash_aware_datetime_near_limits_different_values(
    registry: HasherRegistry,
) -> None:
    hasher = DatetimeHasher()
    plus_one = timezone(timedelta(hours=1))
    first = datetime.min.replace(tzinfo=plus_one)
    second = first + timedelta(microseconds=1)
    assert hasher.hash(first, registry=registry) != hasher.hash(second, registry=registry)


def test_datetime_hasher_hash_aware_datetime_near_limits_different_offsets(
    registry: HasherRegistry,
) -> None:
    hasher = DatetimeHasher()
    first = datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))
    second = datetime.min.replace(tzinfo=timezone(timedelta(hours=2)))
    assert hasher.hash(first, registry=registry) != hasher.hash(second, registry=registry)


def test_datetime_hasher_hash_aware_datetime_near_limits_differs_from_naive(
    registry: HasherRegistry,
) -> None:
    hasher = DatetimeHasher()
    aware = datetime.max.replace(tzinfo=timezone(timedelta(hours=-1)))
    assert hasher.hash(aware, registry=registry) != hasher.hash(
        aware.replace(tzinfo=None), registry=registry
    )


@pytest.mark.parametrize("length", [2, 16, 128])
def test_datetime_hasher_hash_aware_datetime_near_limits_length(
    length: int, registry: HasherRegistry
) -> None:
    value = datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))
    assert len(DatetimeHasher().hash(value, registry=registry, length=length)) == length


def test_default_registry_hashes_aware_datetime_near_limits() -> None:
    registry = get_default_registry()
    value = datetime.max.replace(tzinfo=timezone(timedelta(hours=-1)))
    assert registry.hash(value) == registry.hash(value)
    assert registry.hash([value, {"k": value}])


def test_datetime_hasher_hash_equal_instants_away_from_limits_still_equal(
    registry: HasherRegistry,
) -> None:
    hasher = DatetimeHasher()
    first = datetime(2024, 1, 1, 12, tzinfo=UTC)
    second = datetime(2024, 1, 1, 13, tzinfo=timezone(timedelta(hours=1)))
    assert hasher.hash(first, registry=registry) == hasher.hash(second, registry=registry)
