from __future__ import annotations

import os

import pytest

from coola.random import get_rng_state, manual_seed, set_rng_state
from tests.helpers import draw_python_random
from tests.integration.helpers import run_in_children

pytestmark = [
    pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires os.fork"),
    # The children only use the standard library and exit right away.
    pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:os.fork\\(\\) was called:RuntimeWarning"),
]


###############################
#     Tests for the forks     #
###############################


def test_forked_child_restores_a_state_of_the_parent() -> None:
    manual_seed(0)
    state = get_rng_state()
    expected = draw_python_random()

    def child() -> str:
        set_rng_state(state)
        return repr(draw_python_random())

    assert run_in_children(1, child) == [repr(expected)]


def test_forked_child_can_read_the_state() -> None:
    def child() -> str:
        return ",".join(get_rng_state())

    assert run_in_children(1, child) == [",".join(get_rng_state())]
