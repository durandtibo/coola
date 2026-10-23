from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

from coola.io import BaseFileSaver
from coola.io.base import _acquire_file_lock, _break_stale_lock, _keep_lock_fresh, _save_lock
from tests.integration.helpers import run_python, run_threads

if TYPE_CHECKING:
    from pathlib import Path

########################################
#     Tests for BaseFileSaver.save     #
########################################


class SimpleFileSaver(BaseFileSaver[str]):
    r"""A file saver that writes the string representation of the data
    to save."""

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return type(other) is type(self)

    def _save_file(self, to_save: Any, path: Path) -> None:
        path.write_text(str(to_save))


class SlowFileSaver(BaseFileSaver[str]):
    r"""A file saver like ``SimpleFileSaver`` but that sleeps while
    writing, to widen the window during which a concurrent ``save`` call
    could interleave with this one."""

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return type(other) is type(self)

    def _save_file(self, to_save: Any, path: Path) -> None:
        path.write_text(str(to_save))
        time.sleep(0.01)


def test_base_file_saver_save_concurrent_exist_ok_true_does_not_interleave(
    tmp_path: Path,
) -> None:
    # Regression test: concurrent ``save(..., exist_ok=True)`` calls to the
    # same path must not interleave their write/commit steps. Before the
    # per-path lock was added, each thread wrote its own tmp file and then
    # raced to commit, so the final content could in principle come from a
    # tmp file that had already been unlinked by another thread, or the
    # commit steps could otherwise interleave; the fix serializes the whole
    # write+commit sequence per path so the result is always exactly one
    # writer's full content, never a mix or a crash.
    path = tmp_path.joinpath("data.txt")
    saver = SlowFileSaver()
    values = [f"value-{i}" for i in range(8)]
    errors: list[BaseException] = []

    def worker(value: str) -> None:
        try:
            saver.save(value, path, exist_ok=True)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    run_threads([threading.Thread(target=worker, args=(value,)) for value in values])

    assert not errors
    assert path.is_file()
    assert path.read_text() in values
    # No leftover tmp files from any writer.
    assert list(tmp_path.iterdir()) == [path]


def test_base_file_saver_save_concurrent_exist_ok_true_serializes_writers(
    tmp_path: Path,
) -> None:
    # Directly verify the per-path lock prevents two ``save`` calls to the
    # same path from ever being inside the write/commit section at the same
    # time.
    path = tmp_path.joinpath("data.txt")
    saver = SlowFileSaver()
    active = 0
    max_active = 0
    guard = threading.Lock()

    original_save_file = saver._save_file

    def tracked_save_file(to_save: Any, path: Path) -> None:
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        try:
            original_save_file(to_save, path)
        finally:
            with guard:
                active -= 1

    saver._save_file = tracked_save_file
    run_threads(
        [
            threading.Thread(
                target=saver.save, args=(f"value-{i}",), kwargs={"path": path, "exist_ok": True}
            )
            for i in range(6)
        ]
    )

    assert max_active == 1


def test_acquire_file_lock_blocks_while_another_holder_owns_the_lock_file(
    tmp_path: Path,
) -> None:
    # Simulate a concurrent writer (possibly in another process) holding the
    # lock file: ``_acquire_file_lock`` must not return until it is removed.
    path = tmp_path.joinpath("data.txt")
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    acquired: list[Path] = []

    def waiter() -> None:
        acquired.append(_acquire_file_lock(path, poll_interval=0.001))

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.05)
    assert not acquired  # still waiting, the lock file is held
    lock_path.unlink()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert acquired == [lock_path]
    lock_path.unlink()


def test_base_file_saver_save_blocks_while_lock_file_is_held(tmp_path: Path) -> None:
    # Regression test for cross-process serialization: even though the
    # in-process ``threading.Lock`` used by ``_get_save_lock`` cannot see
    # holders in other processes, ``save`` must still wait on the lock file
    # left behind by such a holder before writing/committing.
    path = tmp_path.joinpath("data.txt")
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    saver = SimpleFileSaver()
    done = threading.Event()

    def worker() -> None:
        saver.save("value", path, exist_ok=True)
        done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.05)
    assert not done.is_set()
    assert not path.is_file()
    lock_path.unlink()
    thread.join(timeout=5)
    assert done.is_set()
    assert path.is_file()
    assert path.read_text() == "value"
    assert not lock_path.is_file()


_HOLD_LOCK_SCRIPT = """
import sys
import time
from pathlib import Path

from coola.io.base import _acquire_file_lock

_acquire_file_lock(Path(sys.argv[1]))
print("locked", flush=True)
time.sleep(60)
"""


def test_acquire_file_lock_recovers_after_holder_process_is_killed(tmp_path: Path) -> None:
    # Regression test for stale lock-file recovery: a process killed while
    # holding the lock never releases it, so the lock file is left behind.
    # A later writer must be able to break it once it is old enough.
    path = tmp_path.joinpath("data.txt")
    lock_path = tmp_path.joinpath("data.txt.lock")
    child = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", _HOLD_LOCK_SCRIPT, str(path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        assert lock_path.is_file()
    finally:
        child.kill()
        child.wait()
        if child.stdout is not None:
            child.stdout.close()

    # The killed process cannot release the lock, so it is still there and
    # a writer that does not consider it stale times out.
    assert lock_path.is_file()
    with pytest.raises(TimeoutError, match="timed out"):
        _acquire_file_lock(path, timeout=0.05, poll_interval=0.001, stale_after=3600)

    # Once the lock is older than ``stale_after`` it is broken.
    time.sleep(0.06)
    assert _acquire_file_lock(path, timeout=5, stale_after=0.05) == lock_path
    lock_path.unlink()


#####################################
#     Tests for the lock heartbeat  #
#####################################


def test_save_lock_keeps_a_slow_save_from_being_stale(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    lock_path = tmp_path.joinpath("data.txt.lock")
    with _save_lock(path, stale_after=0.2):
        old = time.time() - 1000
        os.utime(lock_path, (old, old))
        # The heartbeat thread refreshes the lock while the save is running.
        time.sleep(0.25)
        assert time.time() - lock_path.stat().st_mtime < 0.2
        assert not _break_stale_lock(lock_path, stale_after=0.2)


def test_save_lock_heartbeat_thread_stops(tmp_path: Path) -> None:
    before = {t.name for t in threading.enumerate()}
    with _save_lock(tmp_path.joinpath("data.txt")):
        pass
    after = {t.name for t in threading.enumerate()}
    assert after == before


def test_keep_lock_fresh_updates_the_modification_time(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("abc")
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    stop = threading.Event()

    def stop_later() -> None:
        time.sleep(0.05)
        stop.set()

    run_threads(
        [
            threading.Thread(target=_keep_lock_fresh, args=(lock_path, "abc", 0.01, stop)),
            threading.Thread(target=stop_later),
        ]
    )
    assert time.time() - lock_path.stat().st_mtime < 10


def test_keep_lock_fresh_stops_if_the_lock_has_another_owner(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("other")
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    # The thread returns by itself because the lock is not owned by the token.
    run_threads(
        [
            threading.Thread(
                target=_keep_lock_fresh, args=(lock_path, "abc", 0.01, threading.Event())
            )
        ]
    )
    assert time.time() - lock_path.stat().st_mtime > 100


def test_keep_lock_fresh_stops_if_the_lock_is_missing(tmp_path: Path) -> None:
    # The thread returns by itself because the lock does not exist.
    run_threads(
        [
            threading.Thread(
                target=_keep_lock_fresh,
                args=(tmp_path.joinpath("missing.lock"), "abc", 0.01, threading.Event()),
            )
        ]
    )


############################################
#     Tests for the import of coola.io     #
############################################


def test_importing_coola_io_does_not_modify_the_default_equality_registry() -> None:
    code = (
        "import coola.io; "
        "from coola.equality.tester import interface; "
        "from coola.utils.singleton import _UNSET; "
        "print(interface._default_registry._instance is _UNSET)"
    )
    assert run_python(code) == ["True"]


def test_default_equality_registry_has_the_io_testers_after_importing_coola_io() -> None:
    code = (
        "import coola.io; "
        "from coola.equality.tester import get_default_registry; "
        "registry = get_default_registry(); "
        "print(registry.has_equality_tester(coola.io.BaseLoader), "
        "registry.has_equality_tester(coola.io.BaseSaver))"
    )
    assert run_python(code) == ["True", "True"]


#########################################################
#     Tests for the save lock across several processes  #
#########################################################

_LOCKED_WRITER_SCRIPT = """
import sys
import time
from pathlib import Path

from coola.io.base import _save_lock

path, log, name, hold = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
with _save_lock(path, stale_after=float(sys.argv[5])):
    with log.open("a") as f:
        f.write(f"{name} start\\n")
    time.sleep(hold)
    with log.open("a") as f:
        f.write(f"{name} end\\n")
"""


def test_save_lock_serializes_two_processes_even_if_the_save_outlasts_stale_after(
    tmp_path: Path,
) -> None:
    # Each process holds the lock for 0.4s, which is longer than ``stale_after``
    # (0.15s): the heartbeat must keep the lock alive, otherwise the second
    # process breaks it and both write at the same time.
    path = tmp_path.joinpath("data.txt")
    log = tmp_path.joinpath("log.txt")
    children = [
        subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", _LOCKED_WRITER_SCRIPT, str(path), str(log), name, "0.4", "0.15"],
        )
        for name in ("a", "b")
    ]
    for child in children:
        assert child.wait(timeout=60) == 0

    events = [line.split() for line in log.read_text().splitlines()]
    assert len(events) == 4
    assert [event for _, event in events] == ["start", "end", "start", "end"]
    assert events[0][0] == events[1][0]
    assert events[2][0] == events[3][0]
    assert not tmp_path.joinpath("data.txt.lock").is_file()
