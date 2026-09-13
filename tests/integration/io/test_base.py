from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from coola.io import BaseFileSaver
from coola.io.base import _acquire_file_lock
from tests.integration.helpers import run_threads

if TYPE_CHECKING:
    from pathlib import Path

#######################################
#     Tests for BaseFileSaver.save     #
#######################################


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
