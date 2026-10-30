from __future__ import annotations

import threading

from coola.utils.singleton import LazySingleton
from tests.integration.helpers import collect_errors, run_threads

###################################
#     Tests for LazySingleton     #
###################################


def test_lazy_singleton_is_thread_safe() -> None:
    build_count = 0
    build_lock = threading.Lock()

    def factory() -> object:
        nonlocal build_count
        # Simulate slow construction to widen the race window so concurrent
        # callers are likely to overlap inside `get()`.
        threading.Event().wait(0.01)
        with build_lock:
            build_count += 1
        return object()

    singleton = LazySingleton(factory)
    results: list[object] = [None] * 32
    errors: list[BaseException] = []

    def target(index: int) -> None:
        with collect_errors(errors):
            results[index] = singleton.get()

    run_threads([threading.Thread(target=target, args=(i,)) for i in range(32)])

    assert not errors
    assert build_count == 1
    assert all(result is results[0] for result in results)


def test_lazy_singleton_get_from_another_thread_waits_for_the_factory() -> None:
    started = threading.Event()
    release = threading.Event()
    results: list[int | None] = [None, None]
    errors: list[BaseException] = []

    def factory() -> int:
        started.set()
        release.wait(timeout=5)
        return 42

    singleton = LazySingleton(factory)

    def first() -> None:
        with collect_errors(errors):
            results[0] = singleton.get()

    def second() -> None:
        # The factory is running in the first thread: this call waits for it.
        started.wait(timeout=5)
        with collect_errors(errors):
            results[1] = singleton.get()

    def release_later() -> None:
        started.wait(timeout=5)
        threading.Event().wait(0.1)
        release.set()

    run_threads(
        [
            threading.Thread(target=first),
            threading.Thread(target=second),
            threading.Thread(target=release_later),
        ]
    )

    assert not errors
    assert results == [42, 42]
