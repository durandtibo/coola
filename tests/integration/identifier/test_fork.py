from __future__ import annotations

import os

import pytest

from coola.identifier import ObjectIdGenerator, generate_object_id
from coola.identifier.monotonic import MonotonicRandom

pytestmark = [
    pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires os.fork"),
    # Python (multi-threaded process) and JAX warn about ``os.fork``: the children only use the
    # standard library and exit right away.
    pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:os.fork\\(\\) was called:RuntimeWarning"),
]


def run_in_children(num_children: int, function: object) -> list[str]:
    """Run ``function`` in ``num_children`` forked children and return
    the string each one returns."""
    results = []
    for _ in range(num_children):
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover  (measured in the child process only)
            os.close(read_fd)
            try:
                os.write(write_fd, function().encode())  # type: ignore[operator]
            finally:
                os._exit(0)
        os.close(write_fd)
        with os.fdopen(read_fd) as file:
            results.append(file.read())
        os.waitpid(pid, 0)
    return results


###############################
#     Tests for the forks     #
###############################


def test_object_ids_of_forked_children_have_distinct_process_values() -> None:
    generate_object_id()  # create the default generator in the parent before the forks
    parent = generate_object_id()
    children = run_in_children(4, generate_object_id)
    process_values = {parent[8:18], *(child[8:18] for child in children)}
    # the 5-byte process value (hex chars 8 to 18) is different in the parent and each child
    assert len(process_values) == 5
    assert len(set(children)) == 4


def test_object_id_generator_forked_children_do_not_repeat_the_ids_of_the_parent() -> None:
    generator = ObjectIdGenerator()
    generator.generate()
    children = run_in_children(3, generator.generate)
    assert len({generator.generate(), *children}) == 4


def test_monotonic_random_is_reset_in_the_forked_child() -> None:
    generator = MonotonicRandom(bits=80)
    generator.next()

    def child() -> str:
        # The state is reset: the first value of the child is a fresh one
        return str(generator._last_timestamp_ms)

    assert run_in_children(1, child) == ["-1"]
    assert generator._last_timestamp_ms != -1  # the parent is not reset


def test_object_id_generator_works_in_the_child_if_the_lock_is_held_during_the_fork() -> None:
    generator = ObjectIdGenerator()
    generator._lock.acquire()  # as if another thread held it when the fork happened
    try:
        children = run_in_children(1, generator.generate)
    finally:
        generator._lock.release()
    assert len(children[0]) == 24
