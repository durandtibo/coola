from __future__ import annotations

import contextlib
import logging

import pytest

from coola.tools.timing import TimingResult, timeblock

# ``timeblock`` logs at the INFO level: the tests set the level locally with ``caplog.at_level``
# instead of relying on a global log level.
LOGGER_NAME = "coola.tools.timing"


##################################
#     Tests for TimingResult     #
##################################


def test_timing_result_initial_started_at() -> None:
    assert TimingResult().started_at is None


def test_timing_result_initial_finished_at() -> None:
    assert TimingResult().finished_at is None


def test_timing_result_elapsed_is_none_when_not_started() -> None:
    assert TimingResult().elapsed is None


def test_timing_result_elapsed_is_none_when_only_started() -> None:
    assert TimingResult(started_at=1000.0).elapsed is None


def test_timing_result_elapsed_is_none_when_only_finished() -> None:
    assert TimingResult(finished_at=1002.0).elapsed is None


def test_timing_result_elapsed() -> None:
    assert TimingResult(started_at=1000.0, finished_at=1002.5).elapsed == 2.5


def test_timing_result_elapsed_zero() -> None:
    assert TimingResult(started_at=1000.0, finished_at=1000.0).elapsed == 0.0


def test_timing_result_repr() -> None:
    assert repr(TimingResult()).startswith("TimingResult(")


###############################
#     Tests for timeblock     #
###############################


def test_timeblock_logs_one_message(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME), timeblock():
        pass  # do nothing
    assert len(caplog.messages) == 1


def test_timeblock(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME), timeblock():
        pass  # do nothing
    assert caplog.messages[0].startswith("Total time: ")


def test_timeblock_custom_message(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME), timeblock("{time}"):
        pass  # do anything
    assert len(caplog.messages) == 1


def test_timeblock_custom_missing_time() -> None:
    with (
        pytest.raises(ValueError, match=r"{time} is missing in the message"),
        timeblock("message"),
    ):
        pass


def test_timeblock_exception_is_raised() -> None:
    msg = "test error"
    with pytest.raises(ValueError, match="test error"), timeblock():
        raise ValueError(msg)


def test_timeblock_exception_logs_one_message(caplog: pytest.LogCaptureFixture) -> None:
    msg = "test error"
    with (
        caplog.at_level(logging.INFO, logger=LOGGER_NAME),
        contextlib.suppress(ValueError),
        timeblock(),
    ):
        raise ValueError(msg)
    assert len(caplog.messages) == 1


def test_timeblock_exception_logs_total_time(caplog: pytest.LogCaptureFixture) -> None:
    msg = "test error"
    with (
        caplog.at_level(logging.INFO, logger=LOGGER_NAME),
        contextlib.suppress(ValueError),
        timeblock(),
    ):
        raise ValueError(msg)
    assert caplog.messages[0].startswith("Total time: ")


def test_timeblock_exception_logs_time(caplog: pytest.LogCaptureFixture) -> None:
    with (
        caplog.at_level(logging.INFO, logger=LOGGER_NAME),
        contextlib.suppress(RuntimeError),
        timeblock("Elapsed: {time}"),
    ):
        raise RuntimeError
    assert caplog.messages[0].startswith("Elapsed: ")


def test_timeblock_exception_logs_time_once(caplog: pytest.LogCaptureFixture) -> None:
    with (
        caplog.at_level(logging.INFO, logger=LOGGER_NAME),
        contextlib.suppress(RuntimeError),
        timeblock("Elapsed: {time}"),
    ):
        raise RuntimeError
    assert len(caplog.messages) == 1


def test_timeblock_with_timing_result_is_a_timing_result() -> None:
    with timeblock() as t:
        pass
    assert isinstance(t, TimingResult)


def test_timeblock_with_timing_result_started_at_in_the_block() -> None:
    with timeblock() as t:
        assert t.started_at is not None


def test_timeblock_with_timing_result_not_finished_in_the_block() -> None:
    with timeblock() as t:
        assert t.finished_at is None


def test_timeblock_with_timing_result_no_elapsed_in_the_block() -> None:
    with timeblock() as t:
        assert t.elapsed is None


def test_timeblock_with_timing_result_started_at_after_the_block() -> None:
    with timeblock() as t:
        pass
    assert t.started_at is not None


def test_timeblock_with_timing_result_finished_at_after_the_block() -> None:
    with timeblock() as t:
        pass
    assert t.finished_at is not None


def test_timeblock_with_timing_result_elapsed_after_the_block() -> None:
    with timeblock() as t:
        pass
    assert t.elapsed is not None


def test_timeblock_message_with_other_braces(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME), timeblock("{'step': 1} took {time} {}"):
        pass
    assert caplog.messages[0].startswith("{'step': 1} took ")


def test_timeblock_message_with_other_braces_ends_with_the_braces(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME), timeblock("{'step': 1} took {time} {}"):
        pass
    assert caplog.messages[0].endswith(" {}")
