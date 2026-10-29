from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.identifier.snowflake import (
    _EPOCH_MS,
    _MAX_SEQUENCE,
    _MAX_WORKER_ID,
    _TIMESTAMP_SHIFT,
    _WORKER_ID_SHIFT,
    SnowflakeIdGenerator,
    extract_snowflake_timestamp_ms,
)

TIME_NS = "coola.identifier.sequence.time.time_ns"
NOW_MS = 1_800_000_000_000


def at(ms: int) -> int:
    return ms * 1_000_000


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


def test_snowflake_id_generator_worker_id_is_required() -> None:
    with pytest.raises(TypeError, match="worker_id"):
        SnowflakeIdGenerator()  # ty: ignore[missing-argument]


def test_snowflake_id_generator_worker_id_is_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        SnowflakeIdGenerator(3)  # ty: ignore[missing-argument, too-many-positional-arguments]


@pytest.mark.parametrize("worker_id", [True, 1.5, "1", None])
def test_snowflake_id_generator_worker_id_wrong_type_raises(worker_id: object) -> None:
    with pytest.raises(TypeError, match="worker_id must be an int"):
        SnowflakeIdGenerator(worker_id=worker_id)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("worker_id", [-1, 1024])
def test_snowflake_id_generator_worker_id_out_of_range_raises(worker_id: int) -> None:
    with pytest.raises(ValueError, match="worker_id must be in the range"):
        SnowflakeIdGenerator(worker_id=worker_id)


@pytest.mark.parametrize("worker_id", [0, 1023])
def test_snowflake_id_generator_worker_id_boundaries_are_valid(worker_id: int) -> None:
    assert _decode(SnowflakeIdGenerator(worker_id=worker_id).generate())[1] == worker_id


def test_snowflake_id_generator_generate_returns_int() -> None:
    assert isinstance(SnowflakeIdGenerator(worker_id=0).generate(), int)


def test_snowflake_id_generator_generate_is_non_negative() -> None:
    assert SnowflakeIdGenerator(worker_id=0).generate() >= 0


def test_snowflake_id_generator_generate_fits_in_63_bits() -> None:
    assert SnowflakeIdGenerator(worker_id=1023).generate() < 2**63


def test_snowflake_id_generator_generate_encodes_the_current_time() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with patch(TIME_NS, return_value=at(NOW_MS)):
        assert _decode(generator.generate()) == (NOW_MS, 5, 0)


def test_snowflake_id_generator_generate_fully_specified_literal() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with patch(TIME_NS, return_value=at(_EPOCH_MS + 1000)):
        assert generator.generate() == 4_194_304_000 + 20_480


def test_snowflake_id_generator_generate_increments_the_sequence_in_the_same_millisecond() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with patch(TIME_NS, return_value=at(NOW_MS)):
        generator.generate()
        assert _decode(generator.generate()) == (NOW_MS, 5, 1)


def test_snowflake_id_generator_generate_restarts_the_sequence_in_a_new_millisecond() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with patch(TIME_NS, return_value=at(NOW_MS)):
        generator.generate()
    with patch(TIME_NS, return_value=at(NOW_MS + 1)):
        assert _decode(generator.generate()) == (NOW_MS + 1, 5, 0)


def test_snowflake_id_generator_generate_increases() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with patch(TIME_NS, return_value=at(NOW_MS)):
        assert generator.generate() < generator.generate()


def test_snowflake_id_generator_generate_does_not_raise_if_the_clock_moves_backward() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with patch(TIME_NS, return_value=at(NOW_MS)):
        generator.generate()
    with patch(TIME_NS, return_value=at(NOW_MS - 10)):
        assert _decode(generator.generate()) == (NOW_MS, 5, 1)


def test_snowflake_id_generator_generate_borrows_the_next_millisecond_when_exhausted() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with patch(TIME_NS, return_value=at(NOW_MS)):
        ids = [generator.generate() for _ in range(_MAX_SEQUENCE + 2)]
    assert _decode(ids[-1]) == (NOW_MS + 1, 5, 0)


def test_snowflake_id_generator_generate_does_not_sleep_when_exhausted() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with (
        patch(TIME_NS, return_value=at(NOW_MS)),
        patch("time.sleep") as sleep,
    ):
        generator.generate()
        generator._sequence._counter = _MAX_SEQUENCE  # skip the calls that fill the millisecond
        generator.generate()
    sleep.assert_not_called()


def test_snowflake_id_generator_two_workers_never_collide_in_the_same_millisecond() -> None:
    first = SnowflakeIdGenerator(worker_id=1)
    second = SnowflakeIdGenerator(worker_id=2)
    with patch(TIME_NS, return_value=at(NOW_MS)):
        assert first.generate() != second.generate()


def test_snowflake_id_generator_generators_have_independent_sequences() -> None:
    first = SnowflakeIdGenerator(worker_id=1)
    second = SnowflakeIdGenerator(worker_id=1)
    with patch(TIME_NS, return_value=at(NOW_MS)):
        first.generate()
        first.generate()
        assert _decode(second.generate()) == (NOW_MS, 1, 0)


def test_snowflake_id_generator_generate_current_time_before_the_epoch_raises() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with (
        patch(TIME_NS, return_value=at(_EPOCH_MS - 1)),
        pytest.raises(ValueError, match="timestamp_ms must be in the range"),
    ):
        generator.generate()


def test_snowflake_id_generator_generate_current_time_after_the_field_raises() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with (
        patch(TIME_NS, return_value=at(_EPOCH_MS + 2**41)),
        pytest.raises(ValueError, match="timestamp_ms must be in the range"),
    ):
        generator.generate()


############################################
#     Tests for the explicit timestamp     #
############################################


def test_snowflake_id_generator_generate_explicit_timestamp() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    assert _decode(generator.generate(timestamp_ms=1_704_067_200_000))[:2] == (1_704_067_200_000, 5)


def test_snowflake_id_generator_generate_explicit_timestamp_minimum() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    assert _decode(generator.generate(timestamp_ms=_EPOCH_MS))[0] == _EPOCH_MS


def test_snowflake_id_generator_generate_explicit_timestamp_maximum() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    assert (
        _decode(generator.generate(timestamp_ms=_EPOCH_MS + 2**41 - 1))[0] == _EPOCH_MS + 2**41 - 1
    )


@pytest.mark.parametrize("timestamp_ms", [_EPOCH_MS - 1, _EPOCH_MS + 2**41])
def test_snowflake_id_generator_generate_explicit_timestamp_out_of_range_raises(
    timestamp_ms: int,
) -> None:
    with pytest.raises(ValueError, match="timestamp_ms must be in the range"):
        SnowflakeIdGenerator(worker_id=5).generate(timestamp_ms=timestamp_ms)


@pytest.mark.parametrize("timestamp_ms", [True, 1.5, "1"])
def test_snowflake_id_generator_generate_timestamp_ms_wrong_type_raises(
    timestamp_ms: object,
) -> None:
    with pytest.raises(TypeError, match="timestamp_ms must be an int"):
        SnowflakeIdGenerator(worker_id=5).generate(timestamp_ms=timestamp_ms)  # ty: ignore[invalid-argument-type]


def test_snowflake_id_generator_generate_timestamp_ms_is_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        SnowflakeIdGenerator(worker_id=5).generate(NOW_MS)  # ty: ignore[too-many-positional-arguments]


def test_snowflake_id_generator_generate_explicit_timestamp_does_not_touch_the_state() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with patch(TIME_NS, return_value=at(NOW_MS)):
        generator.generate()
    generator.generate(timestamp_ms=_EPOCH_MS + 1)
    with patch(TIME_NS, return_value=at(NOW_MS)):
        assert _decode(generator.generate()) == (NOW_MS, 5, 1)


def test_snowflake_id_generator_generate_explicit_timestamp_sequence_is_random() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    with patch("coola.identifier.snowflake.secrets.randbits", return_value=77):
        assert _decode(generator.generate(timestamp_ms=NOW_MS)) == (NOW_MS, 5, 77)


###############################
#     Tests for the forks     #
###############################


def test_snowflake_id_generator_reset_after_fork_makes_generate_raise() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    generator._reset_after_fork()
    with pytest.raises(RuntimeError, match="cannot be used in a forked process"):
        generator.generate()


def test_snowflake_id_generator_reset_after_fork_message_asks_for_a_worker_id() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    generator._reset_after_fork()
    with pytest.raises(RuntimeError, match="create one with its own worker_id"):
        generator.generate()


def test_snowflake_id_generator_reset_after_fork_makes_generate_with_a_timestamp_raise() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    generator._reset_after_fork()
    with pytest.raises(RuntimeError, match="cannot be used in a forked process"):
        generator.generate(timestamp_ms=NOW_MS)


def test_snowflake_id_generator_new_generator_works_after_a_fork() -> None:
    SnowflakeIdGenerator(worker_id=5)._reset_after_fork()
    assert isinstance(SnowflakeIdGenerator(worker_id=6).generate(), int)


####################################################
#     Tests for extract_snowflake_timestamp_ms     #
####################################################


def test_extract_snowflake_timestamp_ms_roundtrip() -> None:
    generator = SnowflakeIdGenerator(worker_id=5)
    assert extract_snowflake_timestamp_ms(generator.generate(timestamp_ms=NOW_MS)) == NOW_MS


def test_extract_snowflake_timestamp_ms_minimum() -> None:
    assert extract_snowflake_timestamp_ms(0) == _EPOCH_MS


def test_extract_snowflake_timestamp_ms_negative_raises() -> None:
    with pytest.raises(ValueError, match="snowflake_id must be in the range"):
        extract_snowflake_timestamp_ms(-1)


@pytest.mark.parametrize("snowflake_id", [True, 1.5, "1", None])
def test_extract_snowflake_timestamp_ms_wrong_type_raises(snowflake_id: object) -> None:
    with pytest.raises(TypeError, match="snowflake_id must be an int"):
        extract_snowflake_timestamp_ms(snowflake_id)  # ty: ignore[invalid-argument-type]


def test_extract_snowflake_timestamp_ms_rejects_a_64_bit_value() -> None:
    with pytest.raises(ValueError, match="snowflake_id must be in the range"):
        extract_snowflake_timestamp_ms(1 << 63)


def test_extract_snowflake_timestamp_ms_accepts_a_63_bit_value() -> None:
    assert extract_snowflake_timestamp_ms((1 << 63) - 1) == _EPOCH_MS + (1 << 41) - 1
