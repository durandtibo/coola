from __future__ import annotations

from coola.identifier import fork
from coola.identifier.fork import track_fork


class Resettable:
    def __init__(self) -> None:
        self.resets = 0
        track_fork(self)

    def _reset_after_fork(self) -> None:
        self.resets += 1


##############################
#     Tests for track_fork   #
##############################


def test_track_fork_resets_the_tracked_objects() -> None:
    first, second = Resettable(), Resettable()
    fork._after_fork_in_child()
    assert (first.resets, second.resets) == (1, 1)


def test_track_fork_resets_at_each_fork() -> None:
    obj = Resettable()
    fork._after_fork_in_child()
    fork._after_fork_in_child()
    assert obj.resets == 2


def test_track_fork_does_not_keep_the_object_alive() -> None:
    obj = Resettable()
    assert obj in fork._TRACKED
    size = len(fork._TRACKED)
    del obj
    assert len(fork._TRACKED) == size - 1


def test_track_fork_same_object_is_tracked_once() -> None:
    obj = Resettable()
    track_fork(obj)
    fork._after_fork_in_child()
    assert obj.resets == 1
