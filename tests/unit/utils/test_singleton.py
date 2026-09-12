from __future__ import annotations

import threading

import pytest

from coola.utils.singleton import LazySingleton

###################################
#     Tests for LazySingleton     #
###################################


def test_lazy_singleton_get_returns_value_from_factory() -> None:
    singleton = LazySingleton(lambda: 42)
    assert singleton.get() == 42


def test_lazy_singleton_does_not_call_factory_until_get() -> None:
    calls = []

    def factory() -> int:
        calls.append(1)
        return 42

    LazySingleton(factory)
    assert calls == []


def test_lazy_singleton_calls_factory_only_once() -> None:
    calls = []

    def factory() -> int:
        calls.append(1)
        return 42

    singleton = LazySingleton(factory)
    singleton.get()
    singleton.get()
    singleton.get()
    assert len(calls) == 1


def test_lazy_singleton_returns_same_instance() -> None:
    singleton = LazySingleton(object)
    assert singleton.get() is singleton.get()


def test_lazy_singleton_returns_mutated_instance() -> None:
    singleton = LazySingleton(list)
    instance1 = singleton.get()
    instance1.append(1)
    instance2 = singleton.get()
    assert instance1 is instance2
    assert instance2 == [1]


def test_lazy_singleton_independent_instances_have_independent_state() -> None:
    singleton1 = LazySingleton(list)
    singleton2 = LazySingleton(list)
    singleton1.get().append(1)
    assert singleton1.get() == [1]
    assert singleton2.get() == []


def test_lazy_singleton_falsy_value_is_cached_correctly() -> None:
    calls = []

    def factory() -> int:
        calls.append(1)
        return 0

    singleton = LazySingleton(factory)
    assert singleton.get() == 0
    assert singleton.get() == 0
    assert len(calls) == 1


def test_lazy_singleton_none_value_calls_factory_every_time() -> None:
    # A factory returning None cannot be cached because the implementation
    # uses `self._instance is None` as the "not built yet" sentinel.
    calls = []

    def factory() -> None:
        calls.append(1)
        return

    singleton = LazySingleton(factory)
    singleton.get()
    singleton.get()
    assert len(calls) == 2


def test_lazy_singleton_propagates_factory_exception() -> None:
    def factory() -> int:
        msg = "boom"
        raise RuntimeError(msg)

    singleton = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="boom"):
        singleton.get()


def test_lazy_singleton_recovers_after_factory_exception() -> None:
    calls = []

    def factory() -> int:
        calls.append(1)
        if len(calls) == 1:
            msg = "boom"
            raise RuntimeError(msg)
        return 42

    singleton = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="boom"):
        singleton.get()
    assert singleton.get() == 42
    assert len(calls) == 2


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

    threads = [threading.Thread(target=target, args=(i,)) for i in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert build_count == 1
    assert all(result is results[0] for result in results)
