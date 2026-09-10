from __future__ import annotations

import threading
from typing import Self

from coola.utils.singleton import LazySingleton

##################################
#     Tests for LazySingleton     #
##################################


def test_lazy_singleton_get_creates_instance() -> None:
    singleton = LazySingleton(object)
    assert isinstance(singleton.get(), object)


def test_lazy_singleton_get_returns_same_instance() -> None:
    singleton = LazySingleton(object)
    assert singleton.get() is singleton.get()


def test_lazy_singleton_factory_called_once() -> None:
    counter = {"n": 0}

    def factory() -> object:
        counter["n"] += 1
        return object()

    singleton = LazySingleton(factory)
    singleton.get()
    singleton.get()
    singleton.get()
    assert counter["n"] == 1


def test_lazy_singleton_factory_not_called_at_init() -> None:
    counter = {"n": 0}

    def factory() -> object:
        counter["n"] += 1
        return object()

    LazySingleton(factory)
    assert counter["n"] == 0


def test_lazy_singleton_reset_clears_cached_instance() -> None:
    singleton = LazySingleton(object)
    instance1 = singleton.get()
    singleton.reset()
    instance2 = singleton.get()
    assert instance1 is not instance2


def test_lazy_singleton_reset_before_get_is_noop() -> None:
    singleton = LazySingleton(object)
    singleton.reset()
    assert isinstance(singleton.get(), object)


def test_lazy_singleton_concurrent_get_builds_instance_once() -> None:
    counter = {"n": 0}
    lock = threading.Lock()

    def factory() -> object:
        # Simulate work so concurrent callers overlap before the winner stores
        # the instance, exercising the double-checked locking.
        with lock:
            counter["n"] += 1
        return object()

    singleton = LazySingleton(factory)
    results: list[object] = [None] * 32  # type: ignore[list-item]
    barrier = threading.Barrier(32)

    def worker(index: int) -> None:
        barrier.wait()
        results[index] = singleton.get()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert counter["n"] == 1
    assert all(result is results[0] for result in results)


def test_lazy_singleton_skips_factory_when_instance_set_while_waiting_for_lock() -> None:
    """Deterministically exercise the losing side of the double-checked
    lock: the outer check sees no instance, but by the time the lock is
    acquired another caller has already built and stored one, so the
    inner check must be False and the factory must not run again."""
    counter = {"n": 0}

    def factory() -> object:
        counter["n"] += 1
        return object()

    singleton = LazySingleton(factory)

    class _RaceLock:
        r"""Lock stand-in that simulates a concurrent winner storing the
        instance between the outer check and the lock acquisition."""

        def __init__(self) -> None:
            self._real_lock = threading.Lock()

        def __enter__(self) -> Self:
            self._real_lock.acquire()
            if singleton._instance is None:
                singleton._instance = object()
            return self

        def __exit__(self, *args: object) -> None:
            self._real_lock.release()

    singleton._lock = _RaceLock()

    result = singleton.get()

    assert counter["n"] == 0
    assert result is singleton._instance
