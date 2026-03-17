from __future__ import annotations

import pytest

from coola.utils.timing import timeblock

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
