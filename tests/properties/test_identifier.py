r"""Property-based tests for ``coola.identifier``."""

from __future__ import annotations

import re
from typing import Any

from hypothesis import assume, given, strategies as st

from coola.identifier import (
    SnowflakeIdGenerator,
    extract_object_id_timestamp_s,
    extract_snowflake_timestamp_ms,
    extract_ulid_timestamp_ms,
    extract_uuid7_timestamp_ms,
    generate_checksummed_id,
    generate_object_id,
    generate_stable_content_id,
    generate_stable_uuid5,
    generate_ulid,
    generate_uuid7,
    verify_checksummed_id,
)
from coola.identifier.crockford import ALPHABET

ULID_PATTERN = re.compile(r"[0-7][0-9A-HJKMNP-TV-Z]{25}")
UUID7_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
OBJECT_ID_PATTERN = re.compile(r"[0-9a-f]{24}")

EPOCH_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z, the epoch of the Snowflake IDs

timestamps_ms = st.integers(0, 2**48 - 1)
timestamps_s = st.integers(0, 2**32 - 1)
snowflake_timestamps_ms = st.integers(EPOCH_MS, EPOCH_MS + 2**41 - 1)
worker_ids = st.integers(0, 1023)

####################################
#     Time-ordered identifiers     #
####################################


@given(timestamp_ms=timestamps_ms)
def test_ulid_round_trip(timestamp_ms: int) -> None:
    assert extract_ulid_timestamp_ms(generate_ulid(timestamp_ms=timestamp_ms)) == timestamp_ms


@given(timestamp_ms=timestamps_ms)
def test_uuid7_round_trip(timestamp_ms: int) -> None:
    assert extract_uuid7_timestamp_ms(generate_uuid7(timestamp_ms=timestamp_ms)) == timestamp_ms


@given(timestamp_s=timestamps_s)
def test_object_id_round_trip(timestamp_s: int) -> None:
    assert extract_object_id_timestamp_s(generate_object_id(timestamp_s=timestamp_s)) == timestamp_s


@given(timestamp_ms=snowflake_timestamps_ms, worker_id=worker_ids)
def test_snowflake_round_trip(timestamp_ms: int, worker_id: int) -> None:
    generator = SnowflakeIdGenerator(worker_id=worker_id)
    snowflake_id = generator.generate(timestamp_ms=timestamp_ms)
    assert extract_snowflake_timestamp_ms(snowflake_id) == timestamp_ms


@given(timestamp_ms=timestamps_ms)
def test_ulid_shape(timestamp_ms: int) -> None:
    assert ULID_PATTERN.fullmatch(generate_ulid(timestamp_ms=timestamp_ms))


@given(timestamp_ms=timestamps_ms)
def test_uuid7_shape(timestamp_ms: int) -> None:
    assert UUID7_PATTERN.fullmatch(generate_uuid7(timestamp_ms=timestamp_ms))


@given(timestamp_s=timestamps_s)
def test_object_id_shape(timestamp_s: int) -> None:
    assert OBJECT_ID_PATTERN.fullmatch(generate_object_id(timestamp_s=timestamp_s))


@given(timestamp_ms=snowflake_timestamps_ms, worker_id=worker_ids)
def test_snowflake_fits_in_63_bits(timestamp_ms: int, worker_id: int) -> None:
    generator = SnowflakeIdGenerator(worker_id=worker_id)
    assert 0 <= generator.generate(timestamp_ms=timestamp_ms) < 2**63


@given(first=timestamps_ms, second=timestamps_ms)
def test_ulid_sorts_like_the_timestamp(first: int, second: int) -> None:
    assume(first < second)
    assert generate_ulid(timestamp_ms=first) < generate_ulid(timestamp_ms=second)


@given(first=timestamps_ms, second=timestamps_ms)
def test_uuid7_sorts_like_the_timestamp(first: int, second: int) -> None:
    assume(first < second)
    assert generate_uuid7(timestamp_ms=first) < generate_uuid7(timestamp_ms=second)


@given(first=timestamps_s, second=timestamps_s)
def test_object_id_sorts_like_the_timestamp(first: int, second: int) -> None:
    assume(first < second)
    assert generate_object_id(timestamp_s=first) < generate_object_id(timestamp_s=second)


@given(first=snowflake_timestamps_ms, second=snowflake_timestamps_ms, worker_id=worker_ids)
def test_snowflake_sorts_like_the_timestamp(first: int, second: int, worker_id: int) -> None:
    assume(first < second)
    generator = SnowflakeIdGenerator(worker_id=worker_id)
    assert generator.generate(timestamp_ms=first) < generator.generate(timestamp_ms=second)


@given(count=st.integers(2, 50))
def test_ulid_with_the_current_time_is_strictly_increasing(count: int) -> None:
    ids = [generate_ulid() for _ in range(count)]
    assert ids == sorted(set(ids))


@given(count=st.integers(2, 50))
def test_uuid7_with_the_current_time_is_strictly_increasing(count: int) -> None:
    ids = [generate_uuid7() for _ in range(count)]
    assert ids == sorted(set(ids))


@given(count=st.integers(2, 50))
def test_object_id_with_the_current_time_is_strictly_increasing(count: int) -> None:
    ids = [generate_object_id() for _ in range(count)]
    assert ids == sorted(set(ids))


@given(count=st.integers(2, 50), worker_id=worker_ids)
def test_snowflake_with_the_current_time_is_strictly_increasing(count: int, worker_id: int) -> None:
    generator = SnowflakeIdGenerator(worker_id=worker_id)
    ids = [generator.generate() for _ in range(count)]
    assert ids == sorted(set(ids))


##############################
#     Stable identifiers     #
##############################

mappings = st.dictionaries(st.text(max_size=5), st.integers(), max_size=8)


@given(mapping=mappings)
def test_stable_content_id_ignores_the_insertion_order(mapping: dict[str, int]) -> None:
    reordered = dict(reversed(mapping.items()))
    assert generate_stable_content_id(mapping) == generate_stable_content_id(reordered)


@given(mapping=mappings)
def test_stable_uuid5_ignores_the_insertion_order(mapping: dict[str, int]) -> None:
    reordered = dict(reversed(mapping.items()))
    assert generate_stable_uuid5(mapping) == generate_stable_uuid5(reordered)


###################################
#     Checksummed identifiers     #
###################################

lengths = st.integers(1, 40)
group_sizes = st.integers(1, 10)
separators = st.sampled_from(["-", " ", "_", "."])


@given(length=lengths, group_size=group_sizes, sep=separators)
def test_checksummed_id_is_verified(length: int, group_size: int, sep: str) -> None:
    identifier = generate_checksummed_id(length=length, group_size=group_size, sep=sep)
    assert verify_checksummed_id(identifier, sep=sep)


@given(length=lengths, group_size=group_sizes, sep=separators)
def test_checksummed_id_only_has_the_alphabet_and_the_separator(
    length: int, group_size: int, sep: str
) -> None:
    identifier = generate_checksummed_id(length=length, group_size=group_size, sep=sep)
    assert set(identifier) <= set(ALPHABET + sep)


@given(length=lengths, group_size=group_sizes, data=st.data())
def test_checksummed_id_detects_a_single_substitution(
    length: int, group_size: int, data: st.DataObject
) -> None:
    identifier = generate_checksummed_id(length=length, group_size=group_size, sep="")
    index = data.draw(st.integers(0, len(identifier) - 1))
    char = data.draw(st.sampled_from(ALPHABET))
    assume(char != identifier[index])
    corrupted = identifier[:index] + char + identifier[index + 1 :]
    assert not verify_checksummed_id(corrupted, sep="")


@given(length=lengths, group_size=group_sizes, data=st.data())
def test_checksummed_id_detects_a_swap_of_adjacent_characters(
    length: int, group_size: int, data: st.DataObject
) -> None:
    identifier = generate_checksummed_id(length=length, group_size=group_size, sep="")
    index = data.draw(st.integers(0, len(identifier) - 2))
    assume(identifier[index] != identifier[index + 1])
    corrupted = (
        identifier[:index] + identifier[index + 1] + identifier[index] + identifier[index + 2 :]
    )
    assert not verify_checksummed_id(corrupted, sep="")


@given(identifier=st.one_of(st.none(), st.integers(), st.binary(), st.text()))
def test_verify_checksummed_id_never_raises(identifier: Any) -> None:
    assert verify_checksummed_id(identifier) in {True, False}
