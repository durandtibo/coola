from __future__ import annotations

import contextlib
import errno
import os
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from coola.equality import objects_are_equal
from coola.equality.tester import (
    EqualityTesterRegistry,
    EqualNanEqualityTester,
    get_default_registry,
    interface as equality_interface,
)
from coola.factory import OBJECT_TARGET
from coola.io import (
    BaseFileSaver,
    BaseLoader,
    BaseSaver,
    JsonLoader,
    JsonSaver,
    TextLoader,
    base as io_base,
    is_loader_config,
    is_saver_config,
    resolve_loader,
    resolve_saver,
    save_json,
    save_pickle,
    save_text,
)
from coola.io.base import (
    _SAVE_LOCKS,
    _SAVE_LOCKS_RESETTER,
    _acquire_file_lock,
    _break_stale_lock,
    _get_save_lock,
    _keep_lock_fresh,
    _lock_owner_is_dead,
    _make_lock_token,
    _make_parent_dir,
    _release_file_lock,
    _save_lock,
    _short_name,
    _tmp_path,
)
from coola.utils import fork

######################################
#     Tests for is_loader_config     #
######################################

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def test_is_loader_config_true() -> None:
    assert is_loader_config({OBJECT_TARGET: "coola.io.JsonLoader"})


def test_is_loader_config_false() -> None:
    assert not is_loader_config({OBJECT_TARGET: "coola.io.JsonSaver"})


#####################################
#     Tests for is_saver_config     #
#####################################


def test_is_saver_config_true() -> None:
    assert is_saver_config({OBJECT_TARGET: "coola.io.JsonSaver"})


def test_is_saver_config_false() -> None:
    assert not is_saver_config({OBJECT_TARGET: "coola.io.JsonLoader"})


####################################
#     Tests for resolve_loader     #
####################################


def test_resolve_loader_object() -> None:
    loader = JsonLoader()
    assert resolve_loader(loader) is loader


def test_resolve_loader_dict() -> None:
    assert isinstance(resolve_loader({OBJECT_TARGET: "coola.io.JsonLoader"}), JsonLoader)


def test_resolve_loader_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Received object is not a BaseLoader instance"):
        resolve_loader({OBJECT_TARGET: "coola.io.JsonSaver"})


###################################
#     Tests for resolve_saver     #
###################################


def test_resolve_saver_object() -> None:
    saver = JsonSaver()
    assert resolve_saver(saver) is saver


def test_resolve_saver_dict() -> None:
    assert isinstance(resolve_saver({OBJECT_TARGET: "coola.io.JsonSaver"}), JsonSaver)


def test_resolve_saver_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Received object is not a BaseSaver instance"):
        resolve_saver({OBJECT_TARGET: "coola.io.JsonLoader"})


def test_equality_tester_registry_has_equality_tester_loader() -> None:
    assert get_default_registry().has_equality_tester(BaseLoader)


def test_equality_tester_registry_has_equality_tester_saver() -> None:
    assert get_default_registry().has_equality_tester(BaseSaver)


########################################
#     Tests for BaseFileSaver.save     #
########################################


class FailingFileSaver(BaseFileSaver[Any]):
    r"""A file saver whose ``_save_file`` always fails, to test that the
    temporary file is cleaned up when the write fails."""

    def __init__(self, delete_tmp_file: bool = False) -> None:
        # If ``True``, the temp file is removed before raising, to
        # simulate ``_save_file`` implementations that clean up after
        # themselves or never created the file in the first place.
        self.delete_tmp_file = delete_tmp_file
        self.tmp_path: Path | None = None

    def _save_file(self, to_save: Any, path: Path) -> None:  # noqa: ARG002
        self.tmp_path = path
        path.touch()
        if self.delete_tmp_file:
            path.unlink()
        msg = "failed to save"
        raise RuntimeError(msg)


def test_base_file_saver_save_removes_tmp_file_on_failure_raises(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    saver = FailingFileSaver()
    with pytest.raises(RuntimeError, match="failed to save"):
        saver.save("hello", path)


def test_base_file_saver_save_removes_tmp_file_on_failure_state_target_is_not_created(
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath("data.txt")
    saver = FailingFileSaver()
    with contextlib.suppress(RuntimeError):
        saver.save("hello", path)
    assert not path.is_file()


def test_base_file_saver_save_removes_tmp_file_on_failure_state_tmp_file_is_set(
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath("data.txt")
    saver = FailingFileSaver()
    with contextlib.suppress(RuntimeError):
        saver.save("hello", path)
    assert saver.tmp_path is not None


def test_base_file_saver_save_removes_tmp_file_on_failure_state_tmp_file_is_removed(
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath("data.txt")
    saver = FailingFileSaver()
    with contextlib.suppress(RuntimeError):
        saver.save("hello", path)
    assert not saver.tmp_path.is_file()


def test_base_file_saver_save_removes_tmp_file_on_failure_state_directory_is_empty(
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath("data.txt")
    saver = FailingFileSaver()
    with contextlib.suppress(RuntimeError):
        saver.save("hello", path)
    assert list(tmp_path.iterdir()) == []


def test_base_file_saver_save_missing_tmp_file_on_failure_raises(tmp_path: Path) -> None:
    # ``unlink(missing_ok=True)`` must not raise even if the tmp file
    # does not exist anymore when the failure is handled.
    path = tmp_path.joinpath("data.txt")
    saver = FailingFileSaver(delete_tmp_file=True)
    with pytest.raises(RuntimeError, match="failed to save"):
        saver.save("hello", path)


def test_base_file_saver_save_missing_tmp_file_on_failure_state_target_is_not_created(
    tmp_path: Path,
) -> None:
    # ``unlink(missing_ok=True)`` must not raise even if the tmp file
    # does not exist anymore when the failure is handled.
    path = tmp_path.joinpath("data.txt")
    saver = FailingFileSaver(delete_tmp_file=True)
    with contextlib.suppress(RuntimeError):
        saver.save("hello", path)
    assert not path.is_file()


def test_base_file_saver_save_missing_tmp_file_on_failure_state_directory_is_empty(
    tmp_path: Path,
) -> None:
    # ``unlink(missing_ok=True)`` must not raise even if the tmp file
    # does not exist anymore when the failure is handled.
    path = tmp_path.joinpath("data.txt")
    saver = FailingFileSaver(delete_tmp_file=True)
    with contextlib.suppress(RuntimeError):
        saver.save("hello", path)
    assert list(tmp_path.iterdir()) == []


class SimpleFileSaver(BaseFileSaver[str]):
    r"""A file saver that writes the string representation of the data
    to save."""

    def _save_file(self, to_save: Any, path: Path) -> None:
        path.write_text(str(to_save))


def test_base_file_saver_save_exist_ok_overwrites_content_target_is_a_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()
    saver.save("hello", path)
    saver.save("world", path, exist_ok=True)
    assert path.is_file()


def test_base_file_saver_save_exist_ok_overwrites_content_content(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()
    saver.save("hello", path)
    saver.save("world", path, exist_ok=True)
    assert path.read_text() == "world"


def test_base_file_saver_save_removes_tmp_file_on_commit_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failure during the final commit step (``Path.replace``) must
    # still trigger the cleanup of the temporary file, not just a
    # failure in ``_save_file``.
    path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()

    def failing_replace(self: Path, target: Path) -> Path:  # noqa: ARG001
        msg = "failed to commit"
        raise RuntimeError(msg)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(RuntimeError, match="failed to commit"):
        saver.save("hello", path, exist_ok=True)


def test_base_file_saver_save_removes_tmp_file_on_commit_failure_state_target_is_not_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failure during the final commit step (``Path.replace``) must
    # still trigger the cleanup of the temporary file, not just a
    # failure in ``_save_file``.
    path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()

    def failing_replace(self: Path, target: Path) -> Path:  # noqa: ARG001
        msg = "failed to commit"
        raise RuntimeError(msg)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with contextlib.suppress(RuntimeError):
        saver.save("hello", path, exist_ok=True)
    assert not path.is_file()


def test_base_file_saver_save_removes_tmp_file_on_commit_failure_state_directory_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failure during the final commit step (``Path.replace``) must
    # still trigger the cleanup of the temporary file, not just a
    # failure in ``_save_file``.
    path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()

    def failing_replace(self: Path, target: Path) -> Path:  # noqa: ARG001
        msg = "failed to commit"
        raise RuntimeError(msg)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with contextlib.suppress(RuntimeError):
        saver.save("hello", path, exist_ok=True)
    assert list(tmp_path.iterdir()) == []


def test_base_file_saver_save_removes_tmp_file_on_link_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failure during the final commit step (``os.link``, used when
    # ``exist_ok=False``) must still trigger the cleanup of the
    # temporary file, not just a failure in ``_save_file``.
    path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()

    def failing_link(src: str, dst: str) -> None:  # noqa: ARG001
        msg = "failed to commit"
        raise RuntimeError(msg)

    monkeypatch.setattr(os, "link", failing_link)
    with pytest.raises(RuntimeError, match="failed to commit"):
        saver.save("hello", path)


def test_base_file_saver_save_removes_tmp_file_on_link_failure_state_target_is_not_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failure during the final commit step (``os.link``, used when
    # ``exist_ok=False``) must still trigger the cleanup of the
    # temporary file, not just a failure in ``_save_file``.
    path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()

    def failing_link(src: str, dst: str) -> None:  # noqa: ARG001
        msg = "failed to commit"
        raise RuntimeError(msg)

    monkeypatch.setattr(os, "link", failing_link)
    with contextlib.suppress(RuntimeError):
        saver.save("hello", path)
    assert not path.is_file()


def test_base_file_saver_save_removes_tmp_file_on_link_failure_state_directory_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failure during the final commit step (``os.link``, used when
    # ``exist_ok=False``) must still trigger the cleanup of the
    # temporary file, not just a failure in ``_save_file``.
    path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()

    def failing_link(src: str, dst: str) -> None:  # noqa: ARG001
        msg = "failed to commit"
        raise RuntimeError(msg)

    monkeypatch.setattr(os, "link", failing_link)
    with contextlib.suppress(RuntimeError):
        saver.save("hello", path)
    assert list(tmp_path.iterdir()) == []


def test_base_file_saver_save_original_exception_survives_unlink_failure_in_except(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If cleaning up the temp file in the outer ``except`` handler itself
    # raises (e.g. a permission error), the original exception being
    # handled must still be the one that propagates, not the cleanup
    # failure.
    path = tmp_path.joinpath("data.txt")
    saver = FailingFileSaver()

    def failing_unlink(self: Path, missing_ok: bool = False) -> None:
        # Only the temp file cleanup should fail; the lock file cleanup
        # (a different path, suffixed ``.lock``) must be left alone.
        if self.suffix == ".lock":
            try:
                os.unlink(self)  # noqa: PTH108
            except FileNotFoundError:
                if not missing_ok:
                    raise
            return
        msg = "failed to clean up"
        raise PermissionError(msg)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with pytest.raises(RuntimeError, match="failed to save"):
        saver.save("hello", path)


def test_base_file_saver_save_original_exception_survives_unlink_failure_in_finally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If cleaning up the temp file in the ``finally`` block after a failed
    # ``os.link`` itself raises, the original ``os.link`` exception must
    # still be the one that propagates, not the cleanup failure.
    path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()

    def failing_link(src: str, dst: str) -> None:  # noqa: ARG001
        msg = "failed to commit"
        raise RuntimeError(msg)

    def failing_unlink(self: Path, missing_ok: bool = False) -> None:
        # Only the temp file cleanup should fail; the lock file cleanup
        # (a different path, suffixed ``.lock``) must be left alone.
        if self.suffix == ".lock":
            try:
                os.unlink(self)  # noqa: PTH108
            except FileNotFoundError:
                if not missing_ok:
                    raise
            return
        msg = "failed to clean up"
        raise PermissionError(msg)

    monkeypatch.setattr(os, "link", failing_link)
    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with pytest.raises(RuntimeError, match="failed to commit"):
        saver.save("hello", path)


def test_base_file_saver_save_exist_ok_false_fails_atomically_on_concurrent_create_raises(
    tmp_path: Path,
) -> None:
    # If the target path is created after the initial existence check but
    # before the commit step, ``exist_ok=False`` must still raise instead
    # of silently overwriting it.
    target_path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()
    original_save_file = saver._save_file

    def save_file_then_create_target(to_save: Any, path: Path) -> None:
        original_save_file(to_save, path)
        target_path.write_text("concurrent")

    saver._save_file = save_file_then_create_target
    with pytest.raises(FileExistsError):
        saver.save("hello", target_path)


def test_base_file_saver_save_exist_ok_false_fails_atomically_on_concurrent_create_state(
    tmp_path: Path,
) -> None:
    # If the target path is created after the initial existence check but
    # before the commit step, ``exist_ok=False`` must still raise instead
    # of silently overwriting it.
    target_path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()
    original_save_file = saver._save_file

    def save_file_then_create_target(to_save: Any, path: Path) -> None:
        original_save_file(to_save, path)
        target_path.write_text("concurrent")

    saver._save_file = save_file_then_create_target
    with contextlib.suppress(FileExistsError):
        saver.save("hello", target_path)
    assert target_path.read_text() == "concurrent"


def test_get_save_lock_reuses_existing_lock_for_same_path(tmp_path: Path) -> None:
    # Keep a strong reference to the first lock so the ``WeakValueDictionary``
    # entry survives until the second call, exercising the branch where an
    # existing lock is reused instead of a new one being created.
    path = tmp_path.joinpath("data.txt")
    lock = _get_save_lock(path)
    assert _get_save_lock(path) is lock


def test_save_locks_resetter_is_tracked_for_forks() -> None:
    assert _SAVE_LOCKS_RESETTER in fork._TRACKED


def test_save_locks_resetter_reset_after_fork_recreates_the_guard() -> None:
    guard = io_base._SAVE_LOCKS_GUARD
    _SAVE_LOCKS_RESETTER._reset_after_fork()
    assert io_base._SAVE_LOCKS_GUARD is not guard


def test_save_locks_resetter_reset_after_fork_does_not_block_if_the_guard_was_held(
    tmp_path: Path,
) -> None:
    io_base._SAVE_LOCKS_GUARD.acquire()  # held by a thread that does not exist in the child
    _SAVE_LOCKS_RESETTER._reset_after_fork()
    path = tmp_path.joinpath("data.txt")
    lock = _get_save_lock(path)
    assert _get_save_lock(path) is lock


def test_save_locks_resetter_reset_after_fork_clears_the_locks(tmp_path: Path) -> None:
    lock = _get_save_lock(tmp_path.joinpath("data.txt"))
    lock.acquire()  # held by a thread that does not exist in the child
    _SAVE_LOCKS_RESETTER._reset_after_fork()
    assert len(_SAVE_LOCKS) == 0


def test_acquire_file_lock_has_no_lock_file_initially(tmp_path: Path) -> None:
    assert not tmp_path.joinpath("data.txt.lock").is_file()


def test_acquire_file_lock_returns_the_lock_path(tmp_path: Path) -> None:
    # Releasing is the caller's responsibility (``_save_lock`` does this);
    # ``_acquire_file_lock`` itself only creates the lock file.
    path = tmp_path.joinpath("data.txt")
    assert _acquire_file_lock(path) == tmp_path.joinpath("data.txt.lock")


def test_acquire_file_lock_creates_the_lock_file(tmp_path: Path) -> None:
    _acquire_file_lock(tmp_path.joinpath("data.txt"))
    assert tmp_path.joinpath("data.txt.lock").is_file()


def test_acquire_file_lock_times_out_when_never_released(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            _acquire_file_lock(path, timeout=0.01, poll_interval=0.001)
    finally:
        lock_path.unlink()


def test_base_file_saver_save_removes_lock_file_after_success(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    SimpleFileSaver().save("value", path)
    assert not tmp_path.joinpath("data.txt.lock").is_file()


def test_base_file_saver_save_removes_lock_file_after_failure_raises(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    with pytest.raises(RuntimeError, match="failed to save"):
        FailingFileSaver().save("value", path)


def test_base_file_saver_save_removes_lock_file_after_failure_state(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    with contextlib.suppress(RuntimeError):
        FailingFileSaver().save("value", path)
    assert not tmp_path.joinpath("data.txt.lock").is_file()


def test_base_file_saver_save_exist_ok_true_silently_overwrites_concurrent_create(
    tmp_path: Path,
) -> None:
    # Unlike ``exist_ok=False``, the ``exist_ok=True`` path has no TOCTOU
    # guard: a file created concurrently between the initial check and the
    # commit step is silently overwritten by the plain ``Path.replace``
    # commit, instead of raising.
    target_path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()
    original_save_file = saver._save_file

    def save_file_then_create_target(to_save: Any, path: Path) -> None:
        original_save_file(to_save, path)
        target_path.write_text("concurrent")

    saver._save_file = save_file_then_create_target
    saver.save("hello", target_path, exist_ok=True)
    assert target_path.read_text() == "hello"


def _make_stale_lock(tmp_path: Path) -> Path:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    return lock_path


def test_acquire_file_lock_breaks_stale_lock(tmp_path: Path) -> None:
    lock_path = _make_stale_lock(tmp_path)
    assert _acquire_file_lock(tmp_path.joinpath("data.txt"), timeout=0.5, stale_after=10) == (
        lock_path
    )


def test_acquire_file_lock_recreates_the_stale_lock(tmp_path: Path) -> None:
    lock_path = _make_stale_lock(tmp_path)
    _acquire_file_lock(tmp_path.joinpath("data.txt"), timeout=0.5, stale_after=10)
    assert lock_path.is_file()


def test_acquire_file_lock_refreshes_the_stale_lock(tmp_path: Path) -> None:
    lock_path = _make_stale_lock(tmp_path)
    _acquire_file_lock(tmp_path.joinpath("data.txt"), timeout=0.5, stale_after=10)
    assert time.time() - lock_path.stat().st_mtime < 10


def test_acquire_file_lock_removes_the_stale_file(tmp_path: Path) -> None:
    _make_stale_lock(tmp_path)
    _acquire_file_lock(tmp_path.joinpath("data.txt"), timeout=0.5, stale_after=10)
    assert list(tmp_path.glob("*.stale")) == []


def test_acquire_file_lock_keeps_fresh_lock(tmp_path: Path) -> None:
    tmp_path.joinpath("data.txt.lock").touch()
    with pytest.raises(TimeoutError, match="timed out"):
        _acquire_file_lock(
            tmp_path.joinpath("data.txt"), timeout=0.01, poll_interval=0.001, stale_after=100
        )


def test_acquire_file_lock_does_not_remove_fresh_lock(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    with contextlib.suppress(TimeoutError):
        _acquire_file_lock(
            tmp_path.joinpath("data.txt"), timeout=0.01, poll_interval=0.001, stale_after=100
        )
    assert lock_path.is_file()


def test_break_stale_lock_returns_false_when_lock_is_missing(tmp_path: Path) -> None:
    assert not _break_stale_lock(tmp_path.joinpath("data.txt.lock"), stale_after=10)


def test_break_stale_lock_returns_false_when_lock_disappears_before_rename(
    tmp_path: Path,
) -> None:
    # Another waiter may break the same stale lock between our ``stat`` and
    # ``rename`` calls; the loser must not raise.
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    with patch.object(Path, "rename", side_effect=FileNotFoundError):
        assert not _break_stale_lock(lock_path, stale_after=10)


def test_break_stale_lock_removes_old_lock_returns_true(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    assert _break_stale_lock(lock_path, stale_after=10)


def test_break_stale_lock_removes_old_lock_directory_is_empty(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    _break_stale_lock(lock_path, stale_after=10)
    assert list(tmp_path.iterdir()) == []


def test_break_stale_lock_restores_a_lock_renewed_before_the_rename(tmp_path: Path) -> None:
    # Another waiter broke the stale lock and took a new one between the
    # check of the age and the rename: the live lock must be put back.
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("old")
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    original_rename = Path.rename

    def rename_after_renewal(self: Path, target: Path) -> Path:
        self.unlink()
        self.write_text("live")
        return original_rename(self, target)

    with patch.object(Path, "rename", rename_after_renewal):
        assert not _break_stale_lock(lock_path, stale_after=10)


def test_break_stale_lock_keeps_the_live_lock_after_a_renewal(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("old")
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    original_rename = Path.rename

    def rename_after_renewal(self: Path, target: Path) -> Path:
        self.unlink()
        self.write_text("live")
        return original_rename(self, target)

    with patch.object(Path, "rename", rename_after_renewal):
        _break_stale_lock(lock_path, stale_after=10)
    assert lock_path.read_text() == "live"


##################################################
#     Tests for the equality tester registry     #
##################################################


@pytest.fixture
def reset_default_registry() -> Iterator[EqualityTesterRegistry]:
    # The testers are part of the default registry: they do not depend on a registration
    # made when ``coola.io`` was imported, so they survive a reset of the singleton.
    equality_interface._default_registry.reset()
    try:
        yield get_default_registry()
    finally:
        equality_interface._default_registry.reset()


def test_equality_tester_registry_has_the_loader_tester_after_the_singleton_is_reset(
    reset_default_registry: EqualityTesterRegistry,
) -> None:
    assert isinstance(
        reset_default_registry.find_equality_tester(BaseLoader), EqualNanEqualityTester
    )


def test_equality_tester_registry_has_the_saver_tester_after_the_singleton_is_reset(
    reset_default_registry: EqualityTesterRegistry,
) -> None:
    assert isinstance(
        reset_default_registry.find_equality_tester(BaseSaver), EqualNanEqualityTester
    )


def test_equality_tester_registry_uses_the_loader_tester_for_the_subclasses(
    reset_default_registry: EqualityTesterRegistry,
) -> None:
    registry = reset_default_registry
    assert registry.find_equality_tester(JsonLoader) is registry.find_equality_tester(BaseLoader)


def test_loaders_and_savers_are_compared_with_equal_equal_loaders() -> None:
    assert objects_are_equal(JsonLoader(), JsonLoader())


def test_loaders_and_savers_are_compared_with_equal_different_loaders() -> None:
    assert not objects_are_equal(JsonLoader(), TextLoader())


def test_loaders_and_savers_are_compared_with_equal_equal_savers_in_a_dict() -> None:
    assert objects_are_equal({"saver": JsonSaver(indent=2)}, {"saver": JsonSaver(indent=2)})


def test_loaders_and_savers_are_compared_with_equal_different_savers() -> None:
    assert not objects_are_equal(JsonSaver(indent=2), JsonSaver(indent=4))


############################################
#     Tests for the hard link fallback     #
############################################


def unsupported_link(src: str, dst: str) -> None:  # noqa: ARG001
    raise OSError(errno.ENOTSUP, "Operation not supported")


def test_base_file_saver_save_falls_back_to_rename_if_hard_links_are_not_supported_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.joinpath("data.txt")
    monkeypatch.setattr(os, "link", unsupported_link)
    SimpleFileSaver().save("hello", path)
    assert path.read_text() == "hello"


def test_base_file_saver_save_falls_back_to_rename_if_hard_links_are_not_supported_no_other_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.joinpath("data.txt")
    monkeypatch.setattr(os, "link", unsupported_link)
    SimpleFileSaver().save("hello", path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.txt"]  # no tmp or lock file


def test_base_file_saver_save_fallback_raises_if_the_file_exists_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.joinpath("data.txt")

    class CreatingFileSaver(SimpleFileSaver):
        def _save_file(self, to_save: str, path: Path) -> None:
            super()._save_file(to_save, path)
            # The target is created between the initial check and the commit.
            tmp_path.joinpath("data.txt").write_text("other")

    monkeypatch.setattr(os, "link", unsupported_link)
    with pytest.raises(FileExistsError, match="already exists"):
        CreatingFileSaver().save("hello", path)


def test_base_file_saver_save_fallback_raises_if_the_file_exists_state_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.joinpath("data.txt")

    class CreatingFileSaver(SimpleFileSaver):
        def _save_file(self, to_save: str, path: Path) -> None:
            super()._save_file(to_save, path)
            # The target is created between the initial check and the commit.
            tmp_path.joinpath("data.txt").write_text("other")

    monkeypatch.setattr(os, "link", unsupported_link)
    with contextlib.suppress(FileExistsError):
        CreatingFileSaver().save("hello", path)
    assert path.read_text() == "other"


def test_base_file_saver_save_fallback_raises_if_the_file_exists_state_directory_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.joinpath("data.txt")

    class CreatingFileSaver(SimpleFileSaver):
        def _save_file(self, to_save: str, path: Path) -> None:
            super()._save_file(to_save, path)
            # The target is created between the initial check and the commit.
            tmp_path.joinpath("data.txt").write_text("other")

    monkeypatch.setattr(os, "link", unsupported_link)
    with contextlib.suppress(FileExistsError):
        CreatingFileSaver().save("hello", path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.txt"]


def test_base_file_saver_save_link_file_exists_error_is_not_swallowed_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.joinpath("data.txt")

    def exists_link(src: str, dst: str) -> None:  # noqa: ARG001
        raise FileExistsError

    monkeypatch.setattr(os, "link", exists_link)
    with pytest.raises(FileExistsError):
        SimpleFileSaver().save("hello", path)


def test_base_file_saver_save_link_file_exists_error_is_not_swallowed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.joinpath("data.txt")

    def exists_link(src: str, dst: str) -> None:  # noqa: ARG001
        raise FileExistsError

    monkeypatch.setattr(os, "link", exists_link)
    with contextlib.suppress(FileExistsError):
        SimpleFileSaver().save("hello", path)
    assert not path.exists()


def test_base_file_saver_save_fallback_failure_removes_the_tmp_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.joinpath("data.txt")
    monkeypatch.setattr(os, "link", unsupported_link)

    def failing_replace(self: Path, target: Path) -> Path:  # noqa: ARG001
        msg = "no space left"
        raise OSError(errno.ENOSPC, msg)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(OSError, match="no space left"):
        SimpleFileSaver().save("hello", path)


def test_base_file_saver_save_fallback_failure_removes_the_tmp_file_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.joinpath("data.txt")
    monkeypatch.setattr(os, "link", unsupported_link)

    def failing_replace(self: Path, target: Path) -> Path:  # noqa: ARG001
        msg = "no space left"
        raise OSError(errno.ENOSPC, msg)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with contextlib.suppress(OSError):
        SimpleFileSaver().save("hello", path)
    assert list(tmp_path.iterdir()) == []


###########################
#     Tests for fsync     #
###########################


def test_base_file_saver_save_does_not_fsync_by_default(tmp_path: Path) -> None:
    with patch("coola.io.base.os.fsync") as mock_fsync:
        SimpleFileSaver().save("hello", tmp_path.joinpath("data.txt"))
    mock_fsync.assert_not_called()


@pytest.mark.parametrize("exist_ok", [False, True])
def test_base_file_saver_save_fsync_saves_the_file(tmp_path: Path, exist_ok: bool) -> None:
    path = tmp_path.joinpath("data.txt")
    with patch("coola.io.base.os.fsync"):
        SimpleFileSaver().save("hello", path, exist_ok=exist_ok, fsync=True)
    assert path.read_text() == "hello"


@pytest.mark.parametrize("exist_ok", [False, True])
def test_base_file_saver_save_fsync_flushes_the_file_and_the_directory(
    tmp_path: Path, exist_ok: bool
) -> None:
    path = tmp_path.joinpath("data.txt")
    with patch("coola.io.base.os.fsync") as mock_fsync:
        SimpleFileSaver().save("hello", path, exist_ok=exist_ok, fsync=True)
    assert mock_fsync.call_count == 2  # the file, then the directory


def test_base_file_saver_save_fsync_flushes_the_file_before_the_commit(
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath("data.txt")
    events: list[str] = []
    real_link = os.link

    def recording_link(src: str, dst: str) -> None:
        events.append("commit")
        real_link(src, dst)

    with (
        patch("coola.io.base.os.fsync", side_effect=lambda fd: events.append("fsync")),  # noqa: ARG005
        patch("coola.io.base.os.link", recording_link),
    ):
        SimpleFileSaver().save("hello", path, fsync=True)
    assert events == ["fsync", "commit", "fsync"]


@pytest.fixture
def failing_directory_open(tmp_path: Path) -> Callable[..., int]:
    real_open = os.open

    def failing_directory_open(p: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(p) == tmp_path:
            msg = "cannot open a directory"
            raise PermissionError(msg)
        return real_open(p, flags, *args, **kwargs)

    return failing_directory_open


def test_base_file_saver_save_fsync_directory_error_is_ignored(
    tmp_path: Path, failing_directory_open: Callable[..., int]
) -> None:
    path = tmp_path.joinpath("data.txt")
    with patch("coola.io.base.os.open", failing_directory_open):
        SimpleFileSaver().save("hello", path, fsync=True)
    assert path.read_text() == "hello"


def test_base_file_saver_save_fails_with_fsync(tmp_path: Path) -> None:
    with (
        patch("coola.io.base.os.fsync"),
        pytest.raises(RuntimeError, match="failed to save"),
    ):
        FailingFileSaver().save("hello", tmp_path.joinpath("data.txt"), fsync=True)


def test_base_file_saver_save_fsync_not_called_if_saving_fails(tmp_path: Path) -> None:
    with (
        patch("coola.io.base.os.fsync") as mock_fsync,
        contextlib.suppress(RuntimeError),
    ):
        FailingFileSaver().save("hello", tmp_path.joinpath("data.txt"), fsync=True)
    mock_fsync.assert_not_called()


@pytest.mark.parametrize(
    ("save", "name", "data"),
    [
        pytest.param(save_json, "data.json", {"a": 1}, id="json"),
        pytest.param(save_pickle, "data.pkl", {"a": 1}, id="pickle"),
        pytest.param(save_text, "data.txt", "hello", id="text"),
    ],
)
def test_save_functions_forward_fsync(save: Any, name: str, data: Any, tmp_path: Path) -> None:
    with patch("coola.io.base.os.fsync") as mock_fsync:
        save(data, tmp_path.joinpath(name), fsync=True)
    assert mock_fsync.call_count == 2


####################################################
#     Tests for allowed_prefixes of the config     #
####################################################


def test_is_loader_config_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonLoader"}
    assert is_loader_config(config, allowed_prefixes=["coola.io"])


def test_is_loader_config_not_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonLoader"}
    with pytest.raises(ImportError, match="not allowed"):
        is_loader_config(config, allowed_prefixes=["os"])


def test_is_saver_config_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonSaver"}
    assert is_saver_config(config, allowed_prefixes=["coola.io"])


def test_is_saver_config_not_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonSaver"}
    with pytest.raises(ImportError, match="not allowed"):
        is_saver_config(config, allowed_prefixes=["os"])


def test_is_loader_config_module_not_imported() -> None:
    assert not is_loader_config({OBJECT_TARGET: "coola_missing_module.Loader"})


def test_is_loader_config_import_target() -> None:
    with pytest.raises(ImportError, match="coola_missing_module"):
        is_loader_config({OBJECT_TARGET: "coola_missing_module.Loader"}, import_target=True)


def test_is_saver_config_module_not_imported() -> None:
    assert not is_saver_config({OBJECT_TARGET: "coola_missing_module.Saver"})


def test_is_saver_config_import_target() -> None:
    with pytest.raises(ImportError, match="coola_missing_module"):
        is_saver_config({OBJECT_TARGET: "coola_missing_module.Saver"}, import_target=True)


def test_resolve_loader_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonLoader"}
    assert isinstance(resolve_loader(config, allowed_prefixes=["coola.io"]), JsonLoader)


def test_resolve_loader_not_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonLoader"}
    with pytest.raises(ImportError, match="not allowed"):
        resolve_loader(config, allowed_prefixes=["os"])


def test_resolve_saver_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonSaver"}
    assert isinstance(resolve_saver(config, allowed_prefixes=["coola.io"]), JsonSaver)


def test_resolve_saver_not_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonSaver"}
    with pytest.raises(ImportError, match="not allowed"):
        resolve_saver(config, allowed_prefixes=["os"])


#############################
#     Tests for the lock    #
#############################


def test_acquire_file_lock_writes_the_token(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    lock_path = _acquire_file_lock(path, token="abc")  # noqa: S106
    assert lock_path.read_text() == "abc"
    lock_path.unlink()


def test_release_file_lock_removes_own_lock(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("abc")
    _release_file_lock(lock_path, "abc")
    assert not lock_path.exists()


def test_release_file_lock_keeps_lock_of_another_owner_keeps_the_lock(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("other")
    _release_file_lock(lock_path, "abc")
    assert lock_path.read_text() == "other"


def test_release_file_lock_keeps_lock_of_another_owner_logs_a_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("other")
    _release_file_lock(lock_path, "abc")
    assert "owned by another process" in caplog.text


def test_release_file_lock_missing_lock(tmp_path: Path) -> None:
    _release_file_lock(tmp_path.joinpath("data.txt.lock"), "abc")


def test_save_lock_creates_the_lock(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    with _save_lock(tmp_path.joinpath("data.txt")):
        assert lock_path.is_file()


def test_save_lock_creates_and_removes_the_lock(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    with _save_lock(tmp_path.joinpath("data.txt")):
        pass
    assert not lock_path.exists()


def test_save_lock_does_not_remove_a_lock_taken_by_another_process(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    lock_path = tmp_path.joinpath("data.txt.lock")
    with _save_lock(path):
        # Another process considered the lock stale and took it.
        lock_path.unlink()
        lock_path.write_text("another-process")
    assert lock_path.read_text() == "another-process"


class _StopAfter:
    r"""A stand-in for ``threading.Event`` whose ``wait`` returns the
    given values, so ``_keep_lock_fresh`` can be run without a
    thread."""

    def __init__(self, *waits: bool) -> None:
        self._waits = list(waits)

    def wait(self, timeout: float) -> bool:  # noqa: ARG002
        return self._waits.pop(0)


def test_keep_lock_fresh_updates_the_modification_time(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("abc")
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    _keep_lock_fresh(lock_path, "abc", 0.01, _StopAfter(False, False, True))
    assert time.time() - lock_path.stat().st_mtime < 10


def test_keep_lock_fresh_does_nothing_if_stopped(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("abc")
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    _keep_lock_fresh(lock_path, "abc", 0.01, _StopAfter(True))
    assert time.time() - lock_path.stat().st_mtime > 100


def test_keep_lock_fresh_returns_if_the_lock_has_another_owner(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("other")
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    # Only one wait: the function must return without waiting again.
    _keep_lock_fresh(lock_path, "abc", 0.01, _StopAfter(False))
    assert time.time() - lock_path.stat().st_mtime > 100


def test_keep_lock_fresh_returns_if_the_lock_is_missing(tmp_path: Path) -> None:
    _keep_lock_fresh(tmp_path.joinpath("missing.lock"), "abc", 0.01, _StopAfter(False))


def test_stale_lock_of_a_dead_process_is_still_broken(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("dead-process")
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    with _save_lock(path, stale_after=10):
        assert lock_path.read_text() != "dead-process"


###################################
#     Tests for the tmp path      #
###################################


def test_tmp_path_is_hidden_and_keeps_the_extension_parent(tmp_path: Path) -> None:
    result = _tmp_path(tmp_path.joinpath("data.json"))
    assert result.parent == tmp_path


def test_tmp_path_is_hidden_and_keeps_the_extension_hidden_name(tmp_path: Path) -> None:
    result = _tmp_path(tmp_path.joinpath("data.json"))
    assert result.name.startswith(".data-")


def test_tmp_path_is_hidden_and_keeps_the_extension_extension(tmp_path: Path) -> None:
    result = _tmp_path(tmp_path.joinpath("data.json"))
    assert result.suffix == ".json"


def test_tmp_path_without_extension_hidden_name(tmp_path: Path) -> None:
    result = _tmp_path(tmp_path.joinpath("data"))
    assert result.name.startswith(".data-")


def test_tmp_path_without_extension_no_extension(tmp_path: Path) -> None:
    result = _tmp_path(tmp_path.joinpath("data"))
    assert result.suffix == ""


def test_tmp_path_is_unique(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    assert _tmp_path(path) != _tmp_path(path)


def test_save_does_not_write_a_visible_file_while_saving_no_visible_file(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    class Saver(BaseFileSaver[str]):
        def _save_file(self, to_save: str, path: Path) -> None:
            path.write_text(to_save)
            seen.append(
                [
                    p.name
                    for p in tmp_path.iterdir()
                    if not p.name.startswith(".") and p.suffix != ".lock"
                ]
            )

    Saver().save("abc", tmp_path.joinpath("data.json"))
    assert seen == [[]]


def test_save_does_not_write_a_visible_file_while_saving_content(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    class Saver(BaseFileSaver[str]):
        def _save_file(self, to_save: str, path: Path) -> None:
            path.write_text(to_save)
            seen.append(
                [
                    p.name
                    for p in tmp_path.iterdir()
                    if not p.name.startswith(".") and p.suffix != ".lock"
                ]
            )

    Saver().save("abc", tmp_path.joinpath("data.json"))
    assert tmp_path.joinpath("data.json").read_text() == "abc"


#####################################
#     Tests for dead lock owners     #
#####################################


def test_make_lock_token_starts_with_host_and_pid() -> None:
    assert _make_lock_token().startswith(f"{socket.gethostname()}:{os.getpid()}:")


def test_lock_owner_is_dead_false_for_current_process(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text(_make_lock_token())
    assert not _lock_owner_is_dead(lock_path)


def test_lock_owner_is_dead_true_if_process_does_not_exist(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text(f"{socket.gethostname()}:1234:abc")
    with patch("os.kill", side_effect=ProcessLookupError):
        assert _lock_owner_is_dead(lock_path)


def test_lock_owner_is_dead_false_for_another_host(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("another-host-xyz:1234:abc")
    with patch("os.kill", side_effect=ProcessLookupError):
        assert not _lock_owner_is_dead(lock_path)


@pytest.mark.parametrize("content", ["", "abc", f"{socket.gethostname()}:notapid:abc"])
def test_lock_owner_is_dead_false_for_unknown_token(tmp_path: Path, content: str) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text(content)
    assert not _lock_owner_is_dead(lock_path)


def test_lock_owner_is_dead_false_if_lock_is_missing(tmp_path: Path) -> None:
    assert not _lock_owner_is_dead(tmp_path.joinpath("data.txt.lock"))


def test_break_stale_lock_removes_recent_lock_of_dead_owner(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text(f"{socket.gethostname()}:1234:abc")
    with patch("os.kill", side_effect=ProcessLookupError):
        assert _break_stale_lock(lock_path, stale_after=1000)


def test_acquire_file_lock_replaces_lock_of_dead_owner(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    path.with_name("data.txt.lock").write_text(f"{socket.gethostname()}:1234:abc")
    with patch("os.kill", side_effect=ProcessLookupError):
        lock_path = _acquire_file_lock(path, timeout=1, token="new")  # noqa: S106
    assert lock_path.read_text() == "new"


def test_save_exist_ok_keeps_the_mode(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    path.write_text("{}")
    path.chmod(0o600)
    JsonSaver().save({"a": 1}, path, exist_ok=True)
    assert path.stat().st_mode & 0o777 == 0o600


def test_save_exist_ok_follows_a_symlink(tmp_path: Path) -> None:
    target = tmp_path.joinpath("target.json")
    target.write_text("{}")
    link = tmp_path.joinpath("link.json")
    link.symlink_to(target)
    JsonSaver().save({"a": 1}, link, exist_ok=True)
    assert link.is_symlink()


def test_save_exist_ok_writes_the_target_of_a_symlink(tmp_path: Path) -> None:
    target = tmp_path.joinpath("target.json")
    target.write_text("{}")
    link = tmp_path.joinpath("link.json")
    link.symlink_to(target)
    JsonSaver().save({"a": 1}, link, exist_ok=True)
    assert JsonLoader().load(target) == {"a": 1}


def test_save_parent_is_a_file(tmp_path: Path) -> None:
    parent = tmp_path.joinpath("file")
    parent.write_text("abc")
    with pytest.raises(NotADirectoryError, match="is not a directory"):
        JsonSaver().save({"a": 1}, parent.joinpath("data.json"))


def test_get_save_lock_same_lock_for_two_spellings_of_a_path(tmp_path: Path) -> None:
    lock = _get_save_lock(tmp_path.joinpath("data.txt"))
    assert _get_save_lock(tmp_path.joinpath("sub", "..", "data.txt")) is lock


################################
#     Tests for long names     #
################################


def test_save_long_file_name(tmp_path: Path) -> None:
    path = tmp_path.joinpath("a" * 240 + ".json")
    JsonSaver().save({"a": 1}, path)
    assert JsonLoader().load(path) == {"a": 1}


def test_save_long_file_name_exist_ok_false(tmp_path: Path) -> None:
    path = tmp_path.joinpath("a" * 240 + ".json")
    JsonSaver().save({"a": 1}, path, exist_ok=False)
    assert JsonLoader().load(path) == {"a": 1}


def test_short_name_long_suffix_is_dropped() -> None:
    assert "." not in _short_name("a" * 200 + "." + "b" * 20)


def test_break_stale_lock_returns_false_when_renamed_lock_disappears(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    with patch.object(Path, "read_text", side_effect=[lock_path.read_text(), FileNotFoundError]):
        assert not _break_stale_lock(lock_path, stale_after=10)


def test_make_parent_dir_reraises_if_the_parent_is_a_directory(tmp_path: Path) -> None:
    with (
        patch.object(Path, "mkdir", side_effect=FileExistsError),
        pytest.raises(FileExistsError),
    ):
        _make_parent_dir(tmp_path.joinpath("data.txt"))


def test_base_file_saver_save_lock_true_takes_the_lock(tmp_path: Path) -> None:
    with patch("coola.io.base._save_lock") as save_lock:
        SimpleFileSaver().save("hello", tmp_path.joinpath("data.txt"))
    save_lock.assert_called_once()


def test_base_file_saver_save_lock_false_does_not_take_the_lock(tmp_path: Path) -> None:
    with patch("coola.io.base._save_lock") as save_lock:
        SimpleFileSaver().save("hello", tmp_path.joinpath("data.txt"), lock=False)
    save_lock.assert_not_called()


def test_base_file_saver_save_lock_false_saves_the_data(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    SimpleFileSaver().save("hello", path, lock=False)
    assert path.read_text() == "hello"


def test_base_file_saver_save_lock_false_leaves_only_the_file(tmp_path: Path) -> None:
    SimpleFileSaver().save("hello", tmp_path.joinpath("data.txt"), lock=False)
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]


def test_base_file_saver_save_lock_false_exist_ok_false_raises(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    SimpleFileSaver().save("hello", path, lock=False)
    with pytest.raises(FileExistsError, match="already exists"):
        SimpleFileSaver().save("world", path, lock=False)


def test_base_file_saver_save_lock_false_exist_ok_true_overwrites(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    SimpleFileSaver().save("hello", path, lock=False)
    SimpleFileSaver().save("world", path, exist_ok=True, lock=False)
    assert path.read_text() == "world"


def test_base_file_saver_save_lock_false_removes_tmp_file_on_failure(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="failed to save"):
        FailingFileSaver().save("hello", tmp_path.joinpath("data.txt"), lock=False)
    assert list(tmp_path.iterdir()) == []


def test_resolve_saver_wrong_target_is_not_called(tmp_path: Path) -> None:
    path = tmp_path.joinpath("marker")
    with pytest.raises(TypeError, match="no return type hint"):
        resolve_saver({OBJECT_TARGET: "os.mkdir", "path": str(path)}, strict=True)
    assert not path.exists()


def test_resolve_loader_wrong_target_is_not_called(tmp_path: Path) -> None:
    path = tmp_path.joinpath("marker")
    with pytest.raises(TypeError, match="no return type hint"):
        resolve_loader({OBJECT_TARGET: "os.mkdir", "path": str(path)}, strict=True)
    assert not path.exists()


def test_resolve_saver_wrong_class_target_message() -> None:
    with pytest.raises(TypeError, match="it was not called"):
        resolve_saver({OBJECT_TARGET: "coola.io.JsonLoader"})


def test_resolve_loader_wrong_class_target_message() -> None:
    with pytest.raises(TypeError, match="it was not called"):
        resolve_loader({OBJECT_TARGET: "coola.io.JsonSaver"})


####################################################
#     Tests for BaseLoader, BaseSaver (shared)     #
####################################################


class PathTypeLoader(BaseLoader[type]):
    r"""A loader that returns the type of the path given to ``_load``."""

    def _load(self, path: Path) -> type:
        return type(path)


class ConfigurableLoader(BaseLoader[Any]):
    def __init__(self, encoding: str = "utf-8", scale: float = 1.0) -> None:
        self._encoding = encoding
        self._scale = scale

    def _display_fields(self) -> dict[str, Any]:
        return {"encoding": self._encoding, "scale": self._scale}

    def _load(self, path: Path) -> Any:  # noqa: ARG002
        return None


class ConfigurableSaver(BaseFileSaver[Any]):
    def __init__(self, indent: int = 2) -> None:
        self._indent = indent

    def _display_fields(self) -> dict[str, Any]:
        return {"indent": self._indent}

    def _save_file(self, to_save: Any, path: Path) -> None:
        path.write_text(str(to_save))


def test_base_loader_load_passes_a_path_for_a_path(tmp_path: Path) -> None:
    assert PathTypeLoader().load(tmp_path) is type(tmp_path)


def test_base_loader_load_passes_a_path_for_a_str(tmp_path: Path) -> None:
    # Regression: a ``load`` that calls ``Path`` methods failed for a ``str``.
    assert issubclass(PathTypeLoader().load(str(tmp_path)), Path)


def test_base_loader_load_passes_a_path_for_a_file_uri(tmp_path: Path) -> None:
    assert issubclass(PathTypeLoader().load(tmp_path.as_uri()), Path)


def test_base_loader_load_expands_a_file_uri(tmp_path: Path) -> None:
    class Loader(BaseLoader[Path]):
        def _load(self, path: Path) -> Path:
            return path

    assert Loader().load(tmp_path.joinpath("data.txt").as_uri()) == tmp_path.joinpath("data.txt")


def test_base_loader_equal_same_configuration() -> None:
    assert ConfigurableLoader(scale=2.0).equal(ConfigurableLoader(scale=2.0))


def test_base_loader_equal_different_field() -> None:
    assert not ConfigurableLoader(scale=2.0).equal(ConfigurableLoader(scale=3.0))


def test_base_loader_equal_different_type() -> None:
    assert not ConfigurableLoader().equal(PathTypeLoader())


def test_base_loader_equal_subclass_is_not_equal_to_base() -> None:
    class Child(ConfigurableLoader): ...

    assert not ConfigurableLoader().equal(Child())


def test_base_loader_equal_not_a_loader() -> None:
    assert not ConfigurableLoader().equal("abc")


def test_base_loader_equal_no_fields() -> None:
    assert PathTypeLoader().equal(PathTypeLoader())


def test_base_loader_equal_nan_false() -> None:
    assert not ConfigurableLoader(scale=float("nan")).equal(ConfigurableLoader(scale=float("nan")))


def test_base_loader_equal_nan_true() -> None:
    assert ConfigurableLoader(scale=float("nan")).equal(
        ConfigurableLoader(scale=float("nan")), equal_nan=True
    )


def test_base_loader_repr() -> None:
    assert repr(ConfigurableLoader(scale=2.0)) == "ConfigurableLoader(encoding='utf-8', scale=2.0)"


def test_base_loader_repr_no_fields() -> None:
    assert repr(PathTypeLoader()) == "PathTypeLoader()"


def test_base_loader_str() -> None:
    assert str(ConfigurableLoader(scale=2.0)) == "ConfigurableLoader(encoding=utf-8, scale=2.0)"


def test_base_saver_equal_same_configuration() -> None:
    assert ConfigurableSaver(indent=4).equal(ConfigurableSaver(indent=4))


def test_base_saver_equal_different_field() -> None:
    assert not ConfigurableSaver(indent=4).equal(ConfigurableSaver(indent=2))


def test_base_saver_equal_different_type() -> None:
    assert not ConfigurableSaver().equal(SimpleFileSaver())


def test_base_saver_equal_subclass_is_not_equal_to_base() -> None:
    class Child(ConfigurableSaver): ...

    assert not ConfigurableSaver().equal(Child())


def test_base_saver_repr() -> None:
    assert repr(ConfigurableSaver(indent=4)) == "ConfigurableSaver(indent=4)"


def test_base_saver_repr_no_fields() -> None:
    assert repr(SimpleFileSaver()) == "SimpleFileSaver()"


def test_base_saver_str() -> None:
    assert str(ConfigurableSaver(indent=4)) == "ConfigurableSaver(indent=4)"
