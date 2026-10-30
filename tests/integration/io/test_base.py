from __future__ import annotations

import multiprocessing
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

from coola.io import BaseFileSaver
from tests.integration.helpers import collect_errors, run_python, run_threads

if TYPE_CHECKING:
    from pathlib import Path

# The children are forked, not spawned: they do not pay the import of coola again.
_FORK: Any = multiprocessing.get_context("fork") if hasattr(os, "fork") else None
requires_fork = pytest.mark.skipif(_FORK is None, reason="Requires os.fork")
# JAX (when another test imported it) warns about ``os.fork``: the children only use the standard
# library.
pytestmark = [
    pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:os.fork\\(\\) was called:RuntimeWarning"),
]

########################################
#     Tests for BaseFileSaver.save     #
########################################


class SimpleFileSaver(BaseFileSaver[str]):
    r"""A file saver that writes the string representation of the data
    to save."""

    def _save_file(self, to_save: Any, path: Path) -> None:
        path.write_text(str(to_save))


class SlowFileSaver(BaseFileSaver[str]):
    r"""A file saver like ``SimpleFileSaver`` but that sleeps while
    writing, to widen the window during which a concurrent ``save`` call
    could interleave with this one."""

    def _save_file(self, to_save: Any, path: Path) -> None:
        path.write_text(str(to_save))
        time.sleep(0.01)


def test_base_file_saver_save_concurrent_exist_ok_true_does_not_interleave(
    tmp_path: Path,
) -> None:
    """Test that concurrent overwrites of one path from threads leave
    one writer's complete content."""
    # Each thread writes its own hidden tmp file and commits it atomically, so
    # the result is always exactly one writer's full content, never a mix.
    path = tmp_path.joinpath("data.txt")
    saver = SlowFileSaver()
    values = [f"value-{i}" for i in range(8)]
    errors: list[BaseException] = []

    def worker(value: str) -> None:
        with collect_errors(errors):
            saver.save(value, path, exist_ok=True)

    run_threads([threading.Thread(target=worker, args=(value,)) for value in values])

    assert not errors
    assert path.is_file()
    assert path.read_text() in values
    # No leftover tmp files from any writer.
    assert list(tmp_path.iterdir()) == [path]


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


########################################################
#     Tests for the atomic commit across processes     #
########################################################

_NUM_PROCESSES = 8
_CONTENT_SIZE = 200_000


def _create_file(path: Path, value: str, barrier: Any, queue: Any) -> None:  # pragma: no cover
    r"""Create ``path`` with ``exist_ok=False`` and report the
    outcome."""
    barrier.wait()
    try:
        SimpleFileSaver().save(value * _CONTENT_SIZE, path)
    except FileExistsError:
        queue.put("exists")
    else:
        queue.put("created")


def _overwrite_file(path: Path, value: str, barrier: Any, queue: Any) -> None:  # pragma: no cover
    r"""Overwrite ``path`` with ``exist_ok=True`` and report the
    outcome."""
    barrier.wait()
    errors: list[BaseException] = []
    with collect_errors(errors):
        SimpleFileSaver().save(value * _CONTENT_SIZE, path, exist_ok=True)
    queue.put("error" if errors else "done")


def _run_processes(target: Any, path: Path) -> list[str]:
    r"""Run ``target`` in several forked processes that start together
    and return the outcome reported by each."""
    barrier = _FORK.Barrier(_NUM_PROCESSES)
    queue = _FORK.Queue()
    values = [str(i) for i in range(_NUM_PROCESSES)]
    processes = [_FORK.Process(target=target, args=(path, v, barrier, queue)) for v in values]
    for process in processes:
        process.start()
    outcomes = [queue.get(timeout=60) for _ in processes]
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    return outcomes


@requires_fork
def test_processes_create_the_same_file_exactly_one_wins(tmp_path: Path) -> None:
    """Test that processes that create the same file at the same time
    have exactly one winner and that the others see
    ``FileExistsError``."""
    outcomes = _run_processes(_create_file, tmp_path.joinpath("data.txt"))
    assert sorted(outcomes) == ["created"] + ["exists"] * (_NUM_PROCESSES - 1)


@requires_fork
def test_processes_create_the_same_file_content_is_complete(tmp_path: Path) -> None:
    """Test that the file created by racing processes is the complete
    content of one of them."""
    path = tmp_path.joinpath("data.txt")
    _run_processes(_create_file, path)
    assert path.read_text() in [str(i) * _CONTENT_SIZE for i in range(_NUM_PROCESSES)]


@requires_fork
def test_processes_create_the_same_file_leaves_only_the_file(tmp_path: Path) -> None:
    """Test that racing processes leave neither a temporary file nor a
    lock file."""
    _run_processes(_create_file, tmp_path.joinpath("data.txt"))
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]


@requires_fork
def test_processes_overwrite_the_same_file_all_succeed(tmp_path: Path) -> None:
    """Test that processes that overwrite the same file at the same time
    all succeed."""
    assert (
        _run_processes(_overwrite_file, tmp_path.joinpath("data.txt")) == ["done"] * _NUM_PROCESSES
    )


@requires_fork
def test_processes_overwrite_the_same_file_content_is_complete(tmp_path: Path) -> None:
    """Test that the file overwritten by racing processes is the
    complete content of one of them."""
    path = tmp_path.joinpath("data.txt")
    _run_processes(_overwrite_file, path)
    assert path.read_text() in [str(i) * _CONTENT_SIZE for i in range(_NUM_PROCESSES)]


@requires_fork
def test_processes_overwrite_the_same_file_leaves_only_the_file(tmp_path: Path) -> None:
    """Test that overwriting processes leave one file in the
    directory."""
    _run_processes(_overwrite_file, tmp_path.joinpath("data.txt"))
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]


########################################################
#     Tests for the atomic commit across threads       #
########################################################


def test_threads_create_the_same_file_exactly_one_wins(tmp_path: Path) -> None:
    """Test that threads that create the same file have exactly one
    winner and that the others see ``FileExistsError``."""
    path = tmp_path.joinpath("data.txt")
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker(value: str) -> None:
        try:
            SimpleFileSaver().save(value, path)
        except FileExistsError:
            outcome = "exists"
        else:
            outcome = "created"
        with lock:
            outcomes.append(outcome)

    run_threads([threading.Thread(target=worker, args=(str(i),)) for i in range(8)])

    assert sorted(outcomes) == ["created"] + ["exists"] * 7


def test_threads_save_the_same_path_without_a_lock_file(tmp_path: Path) -> None:
    """Test that concurrent saves create no lock file."""
    path = tmp_path.joinpath("data.txt")
    run_threads(
        [
            threading.Thread(
                target=SimpleFileSaver().save, args=(str(i), path), kwargs={"exist_ok": True}
            )
            for i in range(8)
        ]
    )
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]


########################################
#     Tests for the real file system   #
########################################


def test_save_fsync_with_the_real_fsync_leaves_one_file(tmp_path: Path) -> None:
    """Test that ``fsync=True`` succeeds with the real ``os.fsync``."""
    SimpleFileSaver().save("hello", tmp_path.joinpath("data.txt"), fsync=True)
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="Requires non-root")
def test_save_in_a_read_only_directory_raises_permission_error(tmp_path: Path) -> None:
    """Test that a read-only target directory raises
    ``PermissionError``."""
    tmp_path.chmod(0o500)
    try:
        with pytest.raises(PermissionError, match="Permission denied"):
            SimpleFileSaver().save("hello", tmp_path.joinpath("data.txt"))
    finally:
        tmp_path.chmod(0o700)
