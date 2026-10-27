from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Self

import pytest

from coola.hashing import HasherRegistry, StrHasher
from coola.io import base as io_base
from coola.registry import Registry
from coola.utils.singleton import LazySingleton
from tests.integration.helpers import run_in_children

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = [
    pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires os.fork"),
    # Python (multi-threaded process) and JAX warn about ``os.fork``: the children only use the
    # standard library and exit right away.
    pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:os.fork\\(\\) was called:RuntimeWarning"),
]


class HeldLock:
    r"""Hold a lock in a thread of this process until ``release`` is
    called, like a thread that is inside a registry operation when the
    process forks."""

    def __init__(self, get_lock: Callable[[], object]) -> None:
        self._get_lock = get_lock
        self._acquired = threading.Event()
        self._released = threading.Event()
        self._thread = threading.Thread(target=self._hold)

    def _hold(self) -> None:
        with self._get_lock():  # ty: ignore[invalid-context-manager]
            self._acquired.set()
            self._released.wait()

    def __enter__(self) -> Self:
        self._thread.start()
        self._acquired.wait()
        return self

    def __exit__(self, *args: object) -> None:
        self._released.set()
        self._thread.join()


def hold(registry: object, name: str) -> HeldLock:
    # The lock is read at the time of the call: a reset replaces it.
    return HeldLock(lambda: getattr(registry, name))


#####################################
#     Tests for the held locks      #
#####################################


def test_registry_is_usable_in_a_child_forked_while_its_lock_is_held() -> None:
    """Test that a child forked while another thread holds the lock of a
    registry can use the registry."""
    registry = Registry[str, int]()

    def use_registry() -> str:
        registry.register("key", 1)
        return str(registry.get("key"))

    with hold(registry, "_lock"):
        assert run_in_children(1, use_registry) == ["1"]


def test_registry_keeps_its_content_in_a_child_forked_while_its_lock_is_held() -> None:
    registry = Registry[str, int]({"key": 2})
    with hold(registry, "_lock"):
        assert run_in_children(1, lambda: str(registry.get("key"))) == ["2"]


def test_hasher_registry_is_usable_in_a_child_forked_while_its_lock_is_held() -> None:
    registry = HasherRegistry()

    def register_hasher() -> str:
        registry.register(str, StrHasher())
        return "done"

    with hold(registry, "_lock"):
        assert run_in_children(1, register_hasher) == ["done"]


def test_hasher_registry_can_register_a_fallback_in_a_child_forked_while_its_lock_is_held() -> None:
    registry = HasherRegistry()

    def register_fallback() -> str:
        registry.register_fallback(lambda tp: tp is complex, StrHasher())
        return "done"

    with hold(registry, "_lock"):
        assert run_in_children(1, register_fallback) == ["done"]


def test_lazy_singleton_is_usable_in_a_child_forked_while_its_lock_is_held() -> None:
    singleton = LazySingleton(lambda: 42)
    with hold(singleton, "_lock"):
        assert run_in_children(1, lambda: str(singleton.get())) == ["42"]


def test_save_locks_are_usable_in_a_child_forked_while_the_guard_is_held(tmp_path: Path) -> None:
    def get_lock() -> str:
        io_base._get_save_lock(tmp_path.joinpath("data.txt"))
        return "done"

    with HeldLock(lambda: io_base._SAVE_LOCKS_GUARD):
        assert run_in_children(1, get_lock) == ["done"]


def test_parent_registry_still_works_after_the_children_exit() -> None:
    registry = Registry[str, int]()
    with hold(registry, "_lock"):
        run_in_children(1, lambda: "done")
    registry.register("key", 1)
    assert registry.get("key") == 1
