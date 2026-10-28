from __future__ import annotations

import os

import pytest

from coola.identifier import generate_object_id, objectid
from coola.identifier.sequence import TickSequence
from tests.integration.helpers import run_in_children

pytestmark = [
    pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires os.fork"),
    # Python (multi-threaded process) and JAX warn about ``os.fork``: the children only use the
    # standard library and exit right away.
    pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:os.fork\\(\\) was called:RuntimeWarning"),
]


###############################
#     Tests for the forks     #
###############################


def test_object_ids_of_forked_children_do_not_repeat_the_ids_of_the_parent() -> None:
    parent = generate_object_id()
    children = run_in_children(4, generate_object_id)
    assert len({parent, *children}) == 5


def test_tick_sequence_is_reset_in_the_forked_child() -> None:
    generator = TickSequence(bits=80)
    generator.next()

    def child() -> str:
        # The state is reset: the first value of the child is a fresh one
        return str(generator._tick)

    assert run_in_children(1, child) == ["-1"]
    assert generator._tick != -1  # the parent is not reset


def test_object_id_works_in_the_child_if_the_lock_is_held_during_the_fork() -> None:
    sequence = objectid._SEQUENCE
    sequence._lock.acquire()  # as if another thread held it when the fork happened
    try:
        children = run_in_children(1, generate_object_id)
    finally:
        sequence._lock.release()
    assert len(children[0]) == 24
