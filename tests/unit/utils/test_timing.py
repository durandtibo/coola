from __future__ import annotations

import pytest

from coola.utils.timing import TimingResult, timeblock

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


def test_timeblock(caplog: pytest.LogCaptureFixture) -> None:
    with timeblock():
        pass  # do nothing
    assert len(caplog.messages) == 1
    assert caplog.messages[0].startswith("Total time: ")


def test_timeblock_custom_message(caplog: pytest.LogCaptureFixture) -> None:
    with timeblock("{time}"):
        pass  # do anything
    assert len(caplog.messages) == 1


def test_timeblock_custom_missing_time() -> None:
    with (
        pytest.raises(RuntimeError, match=r"{time} is missing in the message"),
        timeblock("message"),
    ):
        pass


def test_timeblock_exception_is_raised(caplog: pytest.LogCaptureFixture) -> None:
    msg = "test error"
    with pytest.raises(ValueError, match="test error"), timeblock():
        raise ValueError(msg)
    assert len(caplog.messages) == 1
    assert caplog.messages[0].startswith("Total time: ")


def test_timeblock_exception_logs_time(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(RuntimeError), timeblock("Elapsed: {time}"):
        raise RuntimeError
    assert len(caplog.messages) == 1
    assert caplog.messages[0].startswith("Elapsed: ")
