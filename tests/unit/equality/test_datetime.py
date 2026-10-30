from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from coola.equality import objects_are_allclose, objects_are_equal

if TYPE_CHECKING:
    from collections.abc import Callable

PLUS_ONE = timezone(timedelta(hours=1))

AWARE_UTC = datetime(2024, 1, 1, 12, tzinfo=UTC)
AWARE_PLUS_ONE_SAME_INSTANT = datetime(2024, 1, 1, 13, tzinfo=PLUS_ONE)
AWARE_PLUS_ONE_OTHER_INSTANT = datetime(2024, 1, 1, 12, tzinfo=PLUS_ONE)
NAIVE = datetime(2024, 1, 1, 12)  # noqa: DTZ001


def test_datetime_same_instant_is_equal_in_python() -> None:
    assert AWARE_UTC == AWARE_PLUS_ONE_SAME_INSTANT


@pytest.mark.parametrize("compare", [objects_are_equal, objects_are_allclose])
def test_datetime_same_instant_different_time_zone_is_equal(compare: Callable[..., bool]) -> None:
    assert compare(AWARE_UTC, AWARE_PLUS_ONE_SAME_INSTANT)


@pytest.mark.parametrize("compare", [objects_are_equal, objects_are_allclose])
def test_datetime_same_instant_different_time_zone_is_equal_reversed(
    compare: Callable[..., bool],
) -> None:
    assert compare(AWARE_PLUS_ONE_SAME_INSTANT, AWARE_UTC)


@pytest.mark.parametrize("compare", [objects_are_equal, objects_are_allclose])
def test_datetime_same_time_zone_is_equal(compare: Callable[..., bool]) -> None:
    assert compare(AWARE_UTC, datetime(2024, 1, 1, 12, tzinfo=UTC))


@pytest.mark.parametrize("compare", [objects_are_equal, objects_are_allclose])
def test_datetime_different_instant_is_not_equal(compare: Callable[..., bool]) -> None:
    # Same clock digits, but 12:00 UTC+1 is 11:00 UTC.
    assert not compare(AWARE_UTC, AWARE_PLUS_ONE_OTHER_INSTANT)


@pytest.mark.parametrize("compare", [objects_are_equal, objects_are_allclose])
def test_datetime_naive_and_aware_are_not_equal(compare: Callable[..., bool]) -> None:
    assert not compare(NAIVE, AWARE_UTC)


@pytest.mark.parametrize("compare", [objects_are_equal, objects_are_allclose])
def test_datetime_aware_and_naive_are_not_equal(compare: Callable[..., bool]) -> None:
    assert not compare(AWARE_UTC, NAIVE)


@pytest.mark.parametrize("compare", [objects_are_equal, objects_are_allclose])
def test_datetime_naive_is_equal_to_same_naive(compare: Callable[..., bool]) -> None:
    assert compare(NAIVE, datetime(2024, 1, 1, 12))  # noqa: DTZ001


@pytest.mark.parametrize("compare", [objects_are_equal, objects_are_allclose])
def test_datetime_and_date_are_not_equal(compare: Callable[..., bool]) -> None:
    assert not compare(date(2024, 1, 1), datetime(2024, 1, 1, tzinfo=UTC))


@pytest.mark.parametrize("compare", [objects_are_equal, objects_are_allclose])
def test_datetime_nested_in_containers(compare: Callable[..., bool]) -> None:
    assert compare({"when": [AWARE_UTC]}, {"when": [AWARE_PLUS_ONE_SAME_INSTANT]})


@pytest.mark.parametrize("compare", [objects_are_equal, objects_are_allclose])
def test_datetime_nested_in_containers_naive_and_aware(compare: Callable[..., bool]) -> None:
    assert not compare({"when": [AWARE_UTC]}, {"when": [NAIVE]})
