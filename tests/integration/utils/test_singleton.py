from __future__ import annotations

import threading

from coola.utils.singleton import LazySingleton
from tests.integration.helpers import run_threads

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
        try:
            results[index] = singleton.get()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    run_threads([threading.Thread(target=target, args=(i,)) for i in range(32)])

    assert not errors
    assert build_count == 1
    assert all(result is results[0] for result in results)
