from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from coola.equality.tester import get_default_registry
from coola.factory import OBJECT_TARGET
from coola.io import (
    BaseFileSaver,
    BaseLoader,
    BaseSaver,
    JsonLoader,
    JsonSaver,
    is_loader_config,
    is_saver_config,
    resolve_loader,
    resolve_saver,
)
from coola.io.base import _acquire_file_lock, _get_save_lock

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


#######################################
#     Tests for BaseFileSaver.save     #
#######################################


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
            _acquire_file_lock(path, timeout=0.05, poll_interval=0.001)
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
