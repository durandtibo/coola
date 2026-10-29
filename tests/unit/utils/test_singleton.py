from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast

import pytest

from coola.utils import fork
from coola.utils.singleton import LazySingleton

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable

###################################
#     Tests for LazySingleton     #
###################################


def test_lazy_singleton_get_returns_value_from_factory() -> None:
    singleton = LazySingleton(lambda: 42)
    assert singleton.get() == 42


def test_lazy_singleton_get_returns_none_from_factory() -> None:
    assert LazySingleton(lambda: None).get() is None


def test_lazy_singleton_calls_factory_only_once_when_it_returns_none() -> None:
    calls = []
    singleton_none = LazySingleton(lambda: calls.append(1))
    singleton_none.get()
    singleton_none.get()
    assert len(calls) == 1


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


def test_lazy_singleton_get_in_factory_raises_instead_of_blocking() -> None:
    holder: dict[str, LazySingleton[int]] = {}

    def factory() -> int:
        return holder["singleton"].get()

    holder["singleton"] = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="requested while it is being built"):
        holder["singleton"].get()


def test_lazy_singleton_reentrant_call_raises_error() -> None:
    holder: dict[str, LazySingleton[int]] = {}

    def factory() -> int:
        return holder["singleton"].get()

    holder["singleton"] = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="requested while it is being built"):
        holder["singleton"].get()


def test_lazy_singleton_can_be_built_after_a_reentrant_call_failed() -> None:
    holder: dict[str, LazySingleton[int]] = {}
    calls = []

    def factory() -> int:
        calls.append(1)
        if len(calls) == 1:
            return holder["singleton"].get()
        return 42

    holder["singleton"] = LazySingleton(factory)
    with contextlib.suppress(RuntimeError):
        holder["singleton"].get()
    assert holder["singleton"].get() == 42


def test_lazy_singleton_returns_same_instance() -> None:
    singleton = LazySingleton(object)
    assert singleton.get() is singleton.get()


def test_lazy_singleton_returns_the_same_instance() -> None:
    singleton = LazySingleton(list)
    instance1 = singleton.get()
    instance1.append(1)
    assert instance1 is singleton.get()


def test_lazy_singleton_returns_mutated_instance() -> None:
    singleton = LazySingleton(list)
    singleton.get().append(1)
    assert singleton.get() == [1]


def test_lazy_singleton_mutated_instance_has_the_mutation() -> None:
    singleton1 = LazySingleton(list)
    singleton1.get().append(1)
    assert singleton1.get() == [1]


def test_lazy_singleton_independent_instances_have_independent_state() -> None:
    singleton1 = LazySingleton(list)
    singleton2 = LazySingleton(list)
    singleton1.get().append(1)
    assert singleton2.get() == []


def test_lazy_singleton_falsy_value_is_returned() -> None:
    singleton = LazySingleton(lambda: 0)
    assert singleton.get() == 0


def test_lazy_singleton_falsy_value_is_returned_again() -> None:
    singleton = LazySingleton(lambda: 0)
    singleton.get()
    assert singleton.get() == 0


def test_lazy_singleton_falsy_value_is_cached_correctly() -> None:
    calls = []

    def factory() -> int:
        calls.append(1)
        return 0

    singleton = LazySingleton(factory)
    singleton.get()
    singleton.get()
    assert len(calls) == 1


def test_lazy_singleton_propagates_factory_exception() -> None:
    def factory() -> int:
        msg = "boom"
        raise RuntimeError(msg)

    singleton = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="boom"):
        singleton.get()


def test_lazy_singleton_factory_exception_is_raised() -> None:
    def factory() -> int:
        msg = "boom"
        raise RuntimeError(msg)

    singleton = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="boom"):
        singleton.get()


def make_failing_once_factory(calls: list[int]) -> Callable[[], int]:
    def factory() -> int:
        calls.append(1)
        if len(calls) == 1:
            msg = "boom"
            raise RuntimeError(msg)
        return 42

    return factory


def test_lazy_singleton_recovers_after_factory_exception() -> None:
    singleton = LazySingleton(make_failing_once_factory([]))
    with contextlib.suppress(RuntimeError):
        singleton.get()
    assert singleton.get() == 42


def test_lazy_singleton_calls_the_factory_again_after_factory_exception() -> None:
    calls: list[int] = []
    singleton = LazySingleton(make_failing_once_factory(calls))
    with contextlib.suppress(RuntimeError):
        singleton.get()
    singleton.get()
    assert len(calls) == 2


class SettingLock:
    def __init__(self, singleton: LazySingleton[str]) -> None:
        self._singleton = singleton

    def __enter__(self) -> None:
        self._singleton._instance = "built-by-another-caller"

    def __exit__(self, *args: object) -> None:
        return None


def test_lazy_singleton_get_skips_factory_when_instance_set_during_lock_acquisition() -> None:
    # Covers the double-checked-locking branch: if the instance was built by
    # another caller while this one was waiting to acquire the lock, the
    # factory must not be called again once the lock is held.
    singleton = LazySingleton(lambda: "built-by-factory")
    singleton._lock = cast("threading.Lock", SettingLock(singleton))
    assert singleton.get() == "built-by-another-caller"


def test_lazy_singleton_get_does_not_call_the_factory_when_instance_set_during_lock_acquisition() -> (
    None
):
    calls = []

    def factory() -> str:
        calls.append(1)
        return "built-by-factory"

    singleton = LazySingleton(factory)
    singleton._lock = cast("threading.Lock", SettingLock(singleton))
    singleton.get()
    assert calls == []


def test_lazy_singleton_is_built_false_initially() -> None:
    assert not LazySingleton(lambda: 42).is_built()


def test_lazy_singleton_is_built_true_after_get() -> None:
    singleton = LazySingleton(lambda: None)
    singleton.get()
    assert singleton.is_built()


def test_lazy_singleton_reset_rebuilds_the_value() -> None:
    calls = []
    singleton = LazySingleton(lambda: calls.append(1) or len(calls))
    singleton.get()
    singleton.reset()
    assert singleton.get() == 2


def test_lazy_singleton_reset_from_the_factory_raises() -> None:
    singleton = LazySingleton(lambda: singleton.reset())
    with pytest.raises(RuntimeError, match="reset while its value is being built"):
        singleton.get()


def test_lazy_singleton_reset_from_the_factory_does_not_build_the_value() -> None:
    singleton = LazySingleton(lambda: singleton.reset())
    with contextlib.suppress(RuntimeError):
        singleton.get()
    assert not singleton.is_built()


def test_lazy_singleton_set_replaces_the_value() -> None:
    singleton = LazySingleton(lambda: 1)
    singleton.get()
    singleton.set(2)
    assert singleton.get() == 2


def test_lazy_singleton_set_before_get_does_not_call_the_factory() -> None:
    calls = []
    singleton = LazySingleton[Any](lambda: calls.append(1))
    singleton.set(2)
    singleton.get()
    assert calls == []


def test_lazy_singleton_set_makes_the_singleton_built() -> None:
    singleton = LazySingleton(lambda: 1)
    singleton.set(2)
    assert singleton.is_built()


def test_lazy_singleton_set_does_not_change_the_previous_value() -> None:
    previous = [1]
    singleton = LazySingleton(lambda: previous)
    singleton.get()
    singleton.set([2])
    assert previous == [1]


def test_lazy_singleton_reset_after_set_rebuilds_the_value() -> None:
    singleton = LazySingleton(lambda: 1)
    singleton.set(2)
    singleton.reset()
    assert singleton.get() == 1


def test_lazy_singleton_set_from_the_factory_raises() -> None:
    singleton = LazySingleton[Any](lambda: singleton.set(1))
    with pytest.raises(RuntimeError, match="set while its value is being built"):
        singleton.get()


def test_lazy_singleton_set_from_the_factory_does_not_build_the_value() -> None:
    singleton = LazySingleton[Any](lambda: singleton.set(1))
    with contextlib.suppress(RuntimeError):
        singleton.get()
    assert not singleton.is_built()


def test_lazy_singleton_is_tracked_for_forks() -> None:
    singleton = LazySingleton(lambda: 1)
    assert singleton in fork._TRACKED


def test_lazy_singleton_reset_after_fork_recreates_the_lock() -> None:
    singleton = LazySingleton(lambda: 1)
    lock = singleton._lock
    singleton._reset_after_fork()
    assert singleton._lock is not lock


def test_lazy_singleton_reset_after_fork_does_not_block_if_the_lock_was_held() -> None:
    singleton = LazySingleton(lambda: 1)
    singleton._lock.acquire()  # held by a thread that does not exist in the child
    singleton._reset_after_fork()
    assert singleton.get() == 1


def test_lazy_singleton_reset_after_fork_forgets_the_building_thread() -> None:
    singleton = LazySingleton(lambda: 1)
    singleton._building_thread = 1
    singleton._reset_after_fork()
    assert singleton._building_thread is None


def test_lazy_singleton_reset_after_fork_keeps_the_value() -> None:
    singleton = LazySingleton(object)
    value = singleton.get()
    singleton._reset_after_fork()
    assert singleton.get() is value
