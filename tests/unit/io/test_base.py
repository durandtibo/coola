from __future__ import annotations

import errno
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from coola.equality import objects_are_equal
from coola.equality.tester import (
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
    is_loader_config,
    is_saver_config,
    resolve_loader,
    resolve_saver,
    save_json,
    save_pickle,
    save_text,
)
from coola.io.base import (
    _acquire_file_lock,
    _break_stale_lock,
    _get_save_lock,
    _keep_lock_fresh,
    _release_file_lock,
    _save_lock,
    _tmp_path,
)

######################################
#     Tests for is_loader_config     #
######################################


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


def test_equality_tester_registry_has_equality_tester() -> None:
    assert get_default_registry().has_equality_tester(BaseLoader)
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

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return type(other) is type(self)

    def _save_file(self, to_save: Any, path: Path) -> None:  # noqa: ARG002
        self.tmp_path = path
        path.touch()
        if self.delete_tmp_file:
            path.unlink()
        msg = "failed to save"
        raise RuntimeError(msg)


def test_base_file_saver_save_removes_tmp_file_on_failure(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    saver = FailingFileSaver()
    with pytest.raises(RuntimeError, match="failed to save"):
        saver.save("hello", path)
    assert not path.is_file()
    assert saver.tmp_path is not None
    assert not saver.tmp_path.is_file()
    assert list(tmp_path.iterdir()) == []


def test_base_file_saver_save_missing_tmp_file_on_failure(tmp_path: Path) -> None:
    # ``unlink(missing_ok=True)`` must not raise even if the tmp file
    # does not exist anymore when the failure is handled.
    path = tmp_path.joinpath("data.txt")
    saver = FailingFileSaver(delete_tmp_file=True)
    with pytest.raises(RuntimeError, match="failed to save"):
        saver.save("hello", path)
    assert not path.is_file()
    assert list(tmp_path.iterdir()) == []


class SimpleFileSaver(BaseFileSaver[str]):
    r"""A file saver that writes the string representation of the data
    to save."""

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return type(other) is type(self)

    def _save_file(self, to_save: Any, path: Path) -> None:
        path.write_text(str(to_save))


def test_base_file_saver_save_exist_ok_overwrites_content(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    saver = SimpleFileSaver()
    saver.save("hello", path)
    saver.save("world", path, exist_ok=True)
    assert path.is_file()
    assert path.read_text() == "world"


def test_base_file_saver_save_removes_tmp_file_on_commit_failure(
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
    assert not path.is_file()
    assert list(tmp_path.iterdir()) == []


def test_base_file_saver_save_removes_tmp_file_on_link_failure(
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
    assert not path.is_file()
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


def test_base_file_saver_save_exist_ok_false_fails_atomically_on_concurrent_create(
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
    assert target_path.read_text() == "concurrent"


def test_get_save_lock_reuses_existing_lock_for_same_path(tmp_path: Path) -> None:
    # Keep a strong reference to the first lock so the ``WeakValueDictionary``
    # entry survives until the second call, exercising the branch where an
    # existing lock is reused instead of a new one being created.
    path = tmp_path.joinpath("data.txt")
    lock = _get_save_lock(path)
    assert _get_save_lock(path) is lock


def test_acquire_file_lock_creates_and_removes_lock_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    lock_path = tmp_path.joinpath("data.txt.lock")
    assert not lock_path.is_file()
    acquired = _acquire_file_lock(path)
    assert acquired == lock_path
    assert lock_path.is_file()
    # Releasing is the caller's responsibility (``_save_lock`` does this);
    # ``_acquire_file_lock`` itself only creates the lock file.
    lock_path.unlink()


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


def test_base_file_saver_save_removes_lock_file_after_failure(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    with pytest.raises(RuntimeError, match="failed to save"):
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


def test_acquire_file_lock_breaks_stale_lock(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    assert _acquire_file_lock(path, timeout=0.5, stale_after=10) == lock_path
    assert lock_path.is_file()
    assert time.time() - lock_path.stat().st_mtime < 10
    assert list(tmp_path.glob("*.stale")) == []
    lock_path.unlink()


def test_acquire_file_lock_keeps_fresh_lock(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    with pytest.raises(TimeoutError, match="timed out"):
        _acquire_file_lock(path, timeout=0.01, poll_interval=0.001, stale_after=100)
    assert lock_path.is_file()
    lock_path.unlink()


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


def test_break_stale_lock_removes_old_lock(tmp_path: Path) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.touch()
    old = time.time() - 1000
    os.utime(lock_path, (old, old))
    assert _break_stale_lock(lock_path, stale_after=10)
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


def test_equality_tester_registry_has_io_testers_after_the_singleton_is_reset() -> None:
    # The testers are part of the default registry: they do not depend on a registration
    # made when ``coola.io`` was imported, so they survive a reset of the singleton.
    equality_interface._default_registry._instance = None
    try:
        registry = get_default_registry()
        assert isinstance(registry.find_equality_tester(BaseLoader), EqualNanEqualityTester)
        assert isinstance(registry.find_equality_tester(BaseSaver), EqualNanEqualityTester)
        assert registry.find_equality_tester(JsonLoader) is registry.find_equality_tester(
            BaseLoader
        )
    finally:
        equality_interface._default_registry._instance = None


def test_loaders_and_savers_are_compared_with_equal() -> None:
    assert objects_are_equal(JsonLoader(), JsonLoader())
    assert not objects_are_equal(JsonLoader(), TextLoader())
    assert objects_are_equal({"saver": JsonSaver(indent=2)}, {"saver": JsonSaver(indent=2)})
    assert not objects_are_equal(JsonSaver(indent=2), JsonSaver(indent=4))


############################################
#     Tests for the hard link fallback     #
############################################


def unsupported_link(src: str, dst: str) -> None:  # noqa: ARG001
    raise OSError(errno.ENOTSUP, "Operation not supported")


def test_base_file_saver_save_falls_back_to_rename_if_hard_links_are_not_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.joinpath("data.txt")
    monkeypatch.setattr(os, "link", unsupported_link)
    SimpleFileSaver().save("hello", path)
    assert path.read_text() == "hello"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.txt"]  # no tmp or lock file


def test_base_file_saver_save_fallback_raises_if_the_file_exists(
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
    assert path.read_text() == "other"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.txt"]


def test_base_file_saver_save_link_file_exists_error_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path.joinpath("data.txt")

    def exists_link(src: str, dst: str) -> None:  # noqa: ARG001
        raise FileExistsError

    monkeypatch.setattr(os, "link", exists_link)
    with pytest.raises(FileExistsError):
        SimpleFileSaver().save("hello", path)
    assert not path.exists()


def test_base_file_saver_save_fallback_failure_removes_the_tmp_file(
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
    assert list(tmp_path.iterdir()) == []


###########################
#     Tests for fsync     #
###########################


def test_base_file_saver_save_does_not_fsync_by_default(tmp_path: Path) -> None:
    with patch("coola.io.base.os.fsync") as mock_fsync:
        SimpleFileSaver().save("hello", tmp_path.joinpath("data.txt"))
    mock_fsync.assert_not_called()


@pytest.mark.parametrize("exist_ok", [False, True])
def test_base_file_saver_save_fsync_flushes_the_file_and_the_directory(
    tmp_path: Path, exist_ok: bool
) -> None:
    path = tmp_path.joinpath("data.txt")
    with patch("coola.io.base.os.fsync") as mock_fsync:
        SimpleFileSaver().save("hello", path, exist_ok=exist_ok, fsync=True)
    assert path.read_text() == "hello"
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


def test_base_file_saver_save_fsync_directory_error_is_ignored(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    real_open = os.open

    def failing_directory_open(p: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(p) == tmp_path:
            msg = "cannot open a directory"
            raise PermissionError(msg)
        return real_open(p, flags, *args, **kwargs)

    with patch("coola.io.base.os.open", failing_directory_open):
        SimpleFileSaver().save("hello", path, fsync=True)
    assert path.read_text() == "hello"


def test_base_file_saver_save_fsync_not_called_if_saving_fails(tmp_path: Path) -> None:
    with (
        patch("coola.io.base.os.fsync") as mock_fsync,
        pytest.raises(RuntimeError, match="failed to save"),
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
    with pytest.raises(ImportError, match="not allowed"):
        is_loader_config(config, allowed_prefixes=["os"])


def test_is_saver_config_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonSaver"}
    assert is_saver_config(config, allowed_prefixes=["coola.io"])
    with pytest.raises(ImportError, match="not allowed"):
        is_saver_config(config, allowed_prefixes=["os"])


def test_resolve_loader_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonLoader"}
    assert isinstance(resolve_loader(config, allowed_prefixes=["coola.io"]), JsonLoader)
    with pytest.raises(ImportError, match="target object does not exist"):
        resolve_loader(config, allowed_prefixes=["os"])


def test_resolve_saver_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonSaver"}
    assert isinstance(resolve_saver(config, allowed_prefixes=["coola.io"]), JsonSaver)
    with pytest.raises(ImportError, match="target object does not exist"):
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


def test_release_file_lock_keeps_lock_of_another_owner(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lock_path = tmp_path.joinpath("data.txt.lock")
    lock_path.write_text("other")
    _release_file_lock(lock_path, "abc")
    assert lock_path.read_text() == "other"
    assert "owned by another process" in caplog.text


def test_release_file_lock_missing_lock(tmp_path: Path) -> None:
    _release_file_lock(tmp_path.joinpath("data.txt.lock"), "abc")


def test_save_lock_creates_and_removes_the_lock(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    lock_path = tmp_path.joinpath("data.txt.lock")
    with _save_lock(path):
        assert lock_path.is_file()
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


def test_tmp_path_is_hidden_and_keeps_the_extension(tmp_path: Path) -> None:
    result = _tmp_path(tmp_path.joinpath("data.json"))
    assert result.parent == tmp_path
    assert result.name.startswith(".data-")
    assert result.suffix == ".json"


def test_tmp_path_without_extension(tmp_path: Path) -> None:
    result = _tmp_path(tmp_path.joinpath("data"))
    assert result.name.startswith(".data-")
    assert result.suffix == ""


def test_tmp_path_is_unique(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    assert _tmp_path(path) != _tmp_path(path)


def test_save_does_not_write_a_visible_file_while_saving(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    class Saver(BaseFileSaver[str]):
        def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
            return True

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
    assert tmp_path.joinpath("data.json").read_text() == "abc"


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
