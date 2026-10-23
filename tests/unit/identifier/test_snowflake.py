from __future__ import annotations

import contextlib
import itertools
import time
from unittest.mock import patch

import pytest

from coola.identifier import fork, snowflake
from coola.identifier.snowflake import (
    _EPOCH_MS,
    _MAX_SEQUENCE,
    _MAX_WORKER_ID,
    _TIMESTAMP_SHIFT,
    _WORKER_ID_SHIFT,
    SnowflakeIdGenerator,
    extract_snowflake_timestamp_ms,
    generate_snowflake_id,
)


def _decode(snowflake_id: int) -> tuple[int, int, int]:
    """Decode a Snowflake ID into ``(timestamp_ms, worker_id,
    sequence)``.

    Used to verify the bit-packing is actually correct, rather than just
    checking properties of the output as an opaque integer.
    """
    sequence = snowflake_id & _MAX_SEQUENCE
    worker_id = (snowflake_id >> _WORKER_ID_SHIFT) & _MAX_WORKER_ID
    timestamp_ms = (snowflake_id >> _TIMESTAMP_SHIFT) + _EPOCH_MS
    return timestamp_ms, worker_id, sequence


##########################################
#     Tests for SnowflakeIdGenerator     #
##########################################


def test_snowflake_id_generator_init_default_state_last_timestamp() -> None:
    generator = SnowflakeIdGenerator()
    assert generator._last_timestamp_ms == -1


def test_snowflake_id_generator_init_default_state_sequence() -> None:
    generator = SnowflakeIdGenerator()
    assert generator._sequence == 0


def test_snowflake_id_generator_init_resumes_state_last_timestamp() -> None:
    generator = SnowflakeIdGenerator(last_timestamp_ms=1_800_000_000_000, sequence=42)
    assert generator._last_timestamp_ms == 1_800_000_000_000


def test_snowflake_id_generator_init_resumes_state_sequence() -> None:
    generator = SnowflakeIdGenerator(last_timestamp_ms=1_800_000_000_000, sequence=42)
    assert generator._sequence == 42


def test_snowflake_id_generator_init_resumed_state_blocks_earlier_timestamp() -> None:
    generator = SnowflakeIdGenerator(last_timestamp_ms=_EPOCH_MS + (1 << 41) - 1)
    with pytest.raises(RuntimeError, match="clock moved backward"):
        generator.generate()


def test_snowflake_id_generator_init_last_timestamp_ms_too_small_raises() -> None:
    with pytest.raises(ValueError, match="last_timestamp_ms - epoch must fit in 41 bits"):
        SnowflakeIdGenerator(last_timestamp_ms=_EPOCH_MS - 1)


def test_snowflake_id_generator_init_last_timestamp_ms_too_large_raises() -> None:
    with pytest.raises(ValueError, match="last_timestamp_ms - epoch must fit in 41 bits"):
        SnowflakeIdGenerator(last_timestamp_ms=_EPOCH_MS + (1 << 41))


def test_snowflake_id_generator_init_last_timestamp_ms_min_is_valid() -> None:
    SnowflakeIdGenerator(last_timestamp_ms=_EPOCH_MS)


def test_snowflake_id_generator_init_last_timestamp_ms_max_is_valid() -> None:
    SnowflakeIdGenerator(last_timestamp_ms=_EPOCH_MS + (1 << 41) - 1)


def test_snowflake_id_generator_init_negative_sequence_raises() -> None:
    with pytest.raises(ValueError, match="sequence must fit in 12 bits"):
        SnowflakeIdGenerator(sequence=-1)


def test_snowflake_id_generator_init_sequence_too_large_raises() -> None:
    with pytest.raises(ValueError, match="sequence must fit in 12 bits"):
        SnowflakeIdGenerator(sequence=_MAX_SEQUENCE + 1)


def test_snowflake_id_generator_init_max_sequence_is_valid() -> None:
    SnowflakeIdGenerator(sequence=_MAX_SEQUENCE)


def test_snowflake_id_generator_generate_returns_int() -> None:
    assert isinstance(SnowflakeIdGenerator().generate(), int)


def test_snowflake_id_generator_generate_is_non_negative() -> None:
    assert SnowflakeIdGenerator().generate() >= 0


def test_snowflake_id_generator_generate_fits_in_64_bits() -> None:
    assert SnowflakeIdGenerator().generate() < 2**63


def test_snowflake_id_generator_generate_is_unique_across_calls() -> None:
    generator = SnowflakeIdGenerator()
    ids = {generator.generate() for _ in range(1000)}
    assert len(ids) == 1000


def test_snowflake_id_generator_generate_is_monotonically_increasing() -> None:
    generator = SnowflakeIdGenerator()
    ids = [generator.generate() for _ in range(1001)]
    assert all(current > previous for previous, current in itertools.pairwise(ids))


def _generate_from_two_generators() -> tuple[int, int, int]:
    # Two generators must not share sequence/timestamp state: minting
    # from one must not affect what the other returns next.
    generator1 = SnowflakeIdGenerator()
    generator2 = SnowflakeIdGenerator()
    with patch("time.time_ns", return_value=1_800_000_000_000 * 1_000_000):
        first_from_1 = generator1.generate()
        first_from_2 = generator2.generate()
        second_from_1 = generator1.generate()
    return first_from_1, first_from_2, second_from_1


def test_snowflake_id_generator_instances_are_independent_first_generator() -> None:
    first_from_1, _, _ = _generate_from_two_generators()
    assert _decode(first_from_1)[2] == 0


def test_snowflake_id_generator_instances_are_independent_second_generator() -> None:
    _, first_from_2, _ = _generate_from_two_generators()
    assert _decode(first_from_2)[2] == 0


def test_snowflake_id_generator_instances_are_independent_sequence() -> None:
    _, _, second_from_1 = _generate_from_two_generators()
    assert _decode(second_from_1)[2] == 1


def test_snowflake_id_generator_generate_same_millisecond_increments_sequence() -> None:
    generator = SnowflakeIdGenerator()
    with patch("time.time_ns", return_value=1_800_000_000_000 * 1_000_000):
        first = generator.generate()
        second = generator.generate()
    assert second == first + 1


def test_snowflake_id_generator_generate_default_worker_id_is_zero() -> None:
    _, worker_id, _ = _decode(SnowflakeIdGenerator().generate())
    assert worker_id == 0


def test_snowflake_id_generator_generate_encodes_given_worker_id() -> None:
    _, worker_id, _ = _decode(SnowflakeIdGenerator().generate(worker_id=7))
    assert worker_id == 7


def test_snowflake_id_generator_generate_different_worker_ids_differ() -> None:
    generator = SnowflakeIdGenerator()
    with patch("time.time_ns", return_value=1_800_000_000_000 * 1_000_000):
        assert generator.generate(worker_id=0) != generator.generate(worker_id=1)


def test_snowflake_id_generator_generate_negative_worker_id_raises() -> None:
    with pytest.raises(ValueError, match="worker_id must fit in 10 bits"):
        SnowflakeIdGenerator().generate(worker_id=-1)


def test_snowflake_id_generator_generate_worker_id_too_large_raises() -> None:
    with pytest.raises(ValueError, match="worker_id must fit in 10 bits"):
        SnowflakeIdGenerator().generate(worker_id=1024)


def test_snowflake_id_generator_generate_max_worker_id_is_valid() -> None:
    assert isinstance(SnowflakeIdGenerator().generate(worker_id=1023), int)


def test_snowflake_id_generator_generate_accepts_explicit_timestamp_ms() -> None:
    fixed_ms = 1_800_000_000_000
    snowflake_id = SnowflakeIdGenerator().generate(timestamp_ms=fixed_ms)
    decoded_ms, _, _ = _decode(snowflake_id)
    assert decoded_ms == fixed_ms


def test_snowflake_id_generator_generate_timestamp_ms_too_large_raises() -> None:
    generator = SnowflakeIdGenerator()
    with pytest.raises(ValueError, match="timestamp_ms"):
        generator.generate(timestamp_ms=_EPOCH_MS + (1 << 41))


def test_snowflake_id_generator_generate_timestamp_ms_too_small_raises() -> None:
    generator = SnowflakeIdGenerator()
    with pytest.raises(ValueError, match="timestamp_ms"):
        generator.generate(timestamp_ms=_EPOCH_MS - 1)


def test_snowflake_id_generator_generate_timestamp_ms_max_is_valid() -> None:
    fixed_ms = _EPOCH_MS + (1 << 41) - 1
    snowflake_id = SnowflakeIdGenerator().generate(timestamp_ms=fixed_ms)
    decoded_ms, _, _ = _decode(snowflake_id)
    assert decoded_ms == fixed_ms


def test_snowflake_id_generator_generate_timestamp_ms_min_is_valid() -> None:
    snowflake_id = SnowflakeIdGenerator().generate(timestamp_ms=_EPOCH_MS)
    decoded_ms, _, _ = _decode(snowflake_id)
    assert decoded_ms == _EPOCH_MS


def test_snowflake_id_generator_generate_explicit_timestamp_ms_keeps_the_state() -> None:
    generator = SnowflakeIdGenerator()
    generator.generate(timestamp_ms=time.time_ns() // 1_000_000 + 10 * 86_400_000)
    assert generator._last_timestamp_ms == -1


def test_snowflake_id_generator_generate_after_a_future_explicit_timestamp_ms() -> None:
    generator = SnowflakeIdGenerator()
    generator.generate(timestamp_ms=time.time_ns() // 1_000_000 + 10 * 86_400_000)
    assert generator.generate() > 0


def test_snowflake_id_generator_generate_explicit_timestamp_ms_in_the_past() -> None:
    generator = SnowflakeIdGenerator()
    generator.generate()
    snowflake_id = generator.generate(timestamp_ms=1_750_000_000_000)
    assert _decode(snowflake_id)[0] == 1_750_000_000_000


def test_snowflake_id_generator_generate_clock_moved_backward_raises() -> None:
    generator = SnowflakeIdGenerator(last_timestamp_ms=3_900_000_000_000)
    with pytest.raises(RuntimeError, match="clock moved backward"):
        generator.generate()


def test_snowflake_id_generator_generate_encodes_timestamp_roundtrip() -> None:
    fixed_ms = 1_800_000_000_000
    with patch("time.time_ns", return_value=fixed_ms * 1_000_000):
        snowflake_id = SnowflakeIdGenerator().generate()
    decoded_ms, _, _ = _decode(snowflake_id)
    assert decoded_ms == fixed_ms


def _generate_after_a_sequence_rollover() -> int:
    first_ms = 1_800_000_000_000
    next_ms = first_ms + 1
    generator = SnowflakeIdGenerator(last_timestamp_ms=first_ms, sequence=_MAX_SEQUENCE)
    # The main body sees `first_ms` again (sequence wraps to 0, forcing
    # the busy-wait loop), which itself observes `first_ms` once more
    # before the clock advances to `next_ms`.
    with patch(
        "time.time_ns",
        side_effect=[first_ms * 1_000_000, first_ms * 1_000_000, next_ms * 1_000_000],
    ):
        return generator.generate()


def test_snowflake_id_generator_generate_sequence_rollover_advances_to_next_millisecond() -> None:
    decoded_ms, _, _ = _decode(_generate_after_a_sequence_rollover())
    assert decoded_ms == 1_800_000_000_001


def test_snowflake_id_generator_generate_sequence_rollover_resets_the_sequence() -> None:
    _, _, sequence = _decode(_generate_after_a_sequence_rollover())
    assert sequence == 0


def test_snowflake_id_generator_generate_holds_lock_during_busy_wait() -> None:
    # Releasing the lock during the wait lets another thread mint the
    # (timestamp, sequence) pair that this call has already used.
    first_ms = 1_800_000_000_000
    next_ms = first_ms + 1
    generator = SnowflakeIdGenerator(last_timestamp_ms=first_ms, sequence=_MAX_SEQUENCE)

    lock_was_free_during_sleep = []

    def fake_sleep(_seconds: float) -> None:
        acquired = generator._lock.acquire(blocking=False)
        lock_was_free_during_sleep.append(acquired)
        if acquired:
            generator._lock.release()

    with (
        patch(
            "time.time_ns",
            side_effect=[first_ms * 1_000_000, first_ms * 1_000_000, next_ms * 1_000_000],
        ),
        patch("time.sleep", side_effect=fake_sleep),
    ):
        generator.generate()

    assert lock_was_free_during_sleep == [False, False]


def test_snowflake_id_generator_generate_clock_moved_backward_during_busy_wait_raises() -> None:
    first_ms = 1_800_000_000_000
    backward_ms = first_ms - 1
    generator = SnowflakeIdGenerator(last_timestamp_ms=first_ms, sequence=_MAX_SEQUENCE)
    # The main body sees `first_ms` again (sequence wraps to 0, forcing
    # the busy-wait loop), which itself then observes the clock having
    # moved backward.
    with (
        patch(
            "time.time_ns",
            side_effect=[first_ms * 1_000_000, backward_ms * 1_000_000],
        ),
        pytest.raises(RuntimeError, match="clock moved backward"),
    ):
        generator.generate()


def test_snowflake_id_generator_generate_sequence_wraps_within_same_millisecond() -> None:
    fixed_ms = 1_800_000_000_000
    generator = SnowflakeIdGenerator(last_timestamp_ms=fixed_ms, sequence=_MAX_SEQUENCE - 2)
    with patch("time.time_ns", return_value=fixed_ms * 1_000_000):
        ids = [generator.generate() for _ in range(2)]
    assert ids[-1] == ids[0] + 1


###########################################
#     Tests for generate_snowflake_id     #
###########################################


def test_generate_snowflake_id_returns_int() -> None:
    assert isinstance(generate_snowflake_id(), int)


def test_generate_snowflake_id_is_unique_across_calls() -> None:
    ids = {generate_snowflake_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_generate_snowflake_id_encodes_given_worker_id() -> None:
    _, worker_id, _ = _decode(generate_snowflake_id(worker_id=7))
    assert worker_id == 7


def test_generate_snowflake_id_negative_worker_id_raises() -> None:
    with pytest.raises(ValueError, match="worker_id must fit in 10 bits"):
        generate_snowflake_id(worker_id=-1)


def test_generate_snowflake_id_uses_shared_default_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Successive calls to the module-level function must observe each
    # other's state (they delegate to the same default generator),
    # unlike two independent `SnowflakeIdGenerator` instances. Swap in
    # a fresh instance for the duration of the test so the real
    # process-wide singleton is not left stamped with the patched
    # timestamp below, which would otherwise make any later call to
    # `generate_snowflake_id` in this process raise "clock moved
    # backward".
    fresh_generator = SnowflakeIdGenerator()
    monkeypatch.setattr(snowflake, "get_default_generator", lambda: fresh_generator)
    with patch("time.time_ns", return_value=1_900_000_000_000 * 1_000_000):
        first = generate_snowflake_id()
        second = generate_snowflake_id()
    assert second == first + 1


####################################################
#     Tests for extract_snowflake_timestamp_ms     #
####################################################


def test_extract_snowflake_timestamp_ms_roundtrip() -> None:
    fixed_ms = 1_800_000_000_000
    with patch("time.time_ns", return_value=fixed_ms * 1_000_000):
        snowflake_id = SnowflakeIdGenerator().generate()
    assert extract_snowflake_timestamp_ms(snowflake_id) == fixed_ms


def test_extract_snowflake_timestamp_ms_matches_decode_helper() -> None:
    snowflake_id = SnowflakeIdGenerator().generate()
    assert extract_snowflake_timestamp_ms(snowflake_id) == _decode(snowflake_id)[0]


def test_extract_snowflake_timestamp_ms_negative_raises() -> None:
    with pytest.raises(ValueError, match="snowflake_id must fit in"):
        extract_snowflake_timestamp_ms(-1)


def test_extract_snowflake_timestamp_ms_too_large_raises() -> None:
    with pytest.raises(ValueError, match="snowflake_id must fit in"):
        extract_snowflake_timestamp_ms(1 << 63)


def test_snowflake_generator_is_tracked_for_forks() -> None:
    generator = SnowflakeIdGenerator()
    assert generator in fork._TRACKED


def test_snowflake_generator_reset_after_fork_recreates_the_lock() -> None:
    generator = SnowflakeIdGenerator()
    lock = generator._lock
    generator._reset_after_fork()
    assert generator._lock is not lock


def test_snowflake_generator_reset_after_fork_does_not_block_if_the_lock_was_held() -> None:
    generator = SnowflakeIdGenerator()
    generator._lock.acquire()  # held by a thread that does not exist in the child
    generator._reset_after_fork()
    assert isinstance(generator.generate(), int)


def test_snowflake_generator_reset_after_fork_keeps_the_ids_monotonic() -> None:
    generator = SnowflakeIdGenerator()
    first = generator.generate()
    generator._reset_after_fork()
    assert generator.generate() > first


def _generate_with_a_backward_clock_in_the_busy_wait() -> SnowflakeIdGenerator:
    first_ms = 1_800_000_000_000
    generator = SnowflakeIdGenerator(last_timestamp_ms=first_ms, sequence=_MAX_SEQUENCE)
    with (
        patch(
            "time.time_ns",
            side_effect=[first_ms * 1_000_000, (first_ms - 1) * 1_000_000],
        ),
        contextlib.suppress(RuntimeError),
    ):
        generator.generate()
    return generator


def test_snowflake_id_generator_backward_clock_in_the_busy_wait_raises_error() -> None:
    first_ms = 1_800_000_000_000
    generator = SnowflakeIdGenerator(last_timestamp_ms=first_ms, sequence=_MAX_SEQUENCE)
    with (
        patch(
            "time.time_ns",
            side_effect=[first_ms * 1_000_000, (first_ms - 1) * 1_000_000],
        ),
        pytest.raises(RuntimeError, match="clock moved backward"),
    ):
        generator.generate()


def test_snowflake_id_generator_last_timestamp_is_unchanged_after_a_backward_clock() -> None:
    generator = _generate_with_a_backward_clock_in_the_busy_wait()
    assert generator._last_timestamp_ms == 1_800_000_000_000


def test_snowflake_id_generator_sequence_is_unchanged_after_a_backward_clock() -> None:
    generator = _generate_with_a_backward_clock_in_the_busy_wait()
    assert generator._sequence == _MAX_SEQUENCE


def _generate_after_a_backward_clock_in_the_busy_wait() -> tuple[set[int], int]:
    first_ms = 1_800_000_000_000
    generator = SnowflakeIdGenerator()
    with patch("time.time_ns", return_value=first_ms * 1_000_000):
        ids = {generator.generate() for _ in range(_MAX_SEQUENCE + 1)}
    # The sequence is exhausted for first_ms, and the clock moves backward during the wait.
    with (
        patch(
            "time.time_ns",
            side_effect=[first_ms * 1_000_000, (first_ms - 1) * 1_000_000],
        ),
        contextlib.suppress(RuntimeError),
    ):
        generator.generate()
    # The clock is back at first_ms and the next millisecond: no ID of first_ms is repeated.
    with patch(
        "time.time_ns",
        side_effect=[first_ms * 1_000_000, (first_ms + 1) * 1_000_000, (first_ms + 1) * 1_000_000],
    ):
        new_id = generator.generate()
    return ids, new_id


def test_snowflake_id_generator_exhausts_the_sequence_with_unique_ids() -> None:
    ids, _ = _generate_after_a_backward_clock_in_the_busy_wait()
    assert len(ids) == _MAX_SEQUENCE + 1


def test_snowflake_id_generator_does_not_repeat_an_id_after_a_backward_clock_in_the_busy_wait() -> (
    None
):
    ids, new_id = _generate_after_a_backward_clock_in_the_busy_wait()
    assert new_id not in ids


def test_snowflake_id_generator_increases_after_a_backward_clock_in_the_busy_wait() -> None:
    ids, new_id = _generate_after_a_backward_clock_in_the_busy_wait()
    assert new_id > max(ids)
