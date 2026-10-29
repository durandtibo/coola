from __future__ import annotations

import contextlib
import errno
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from coola.factory import OBJECT_TARGET
from coola.io import (
    BaseFileSaver,
    JsonLoader,
    JsonSaver,
    resolve_loader,
    resolve_saver,
    save_json,
    save_pickle,
    save_text,
)
from coola.io.base import (
    _make_parent_dir,
    _short_name,
    _tmp_path,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
    assert saver.tmp_path is not None
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


def test_base_file_saver_save_exist_ok_false_fails_atomically_on_concurrent_create_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    monkeypatch.setattr(saver, "_save_file", save_file_then_create_target)
    with pytest.raises(FileExistsError, match="File exists"):
        saver.save("hello", target_path)


def test_base_file_saver_save_exist_ok_false_fails_atomically_on_concurrent_create_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    monkeypatch.setattr(saver, "_save_file", save_file_then_create_target)
    with contextlib.suppress(FileExistsError):
        saver.save("hello", target_path)
    assert target_path.read_text() == "concurrent"


def test_base_file_saver_save_exist_ok_true_silently_overwrites_concurrent_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    monkeypatch.setattr(saver, "_save_file", save_file_then_create_target)
    saver.save("hello", target_path, exist_ok=True)
    assert target_path.read_text() == "hello"


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
        msg = "exists"
        raise FileExistsError(msg)

    monkeypatch.setattr(os, "link", exists_link)
    with pytest.raises(FileExistsError, match="exists"):
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


##############################################
#     Tests for overwrite and directories     #
##############################################


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


def test_make_parent_dir_reraises_if_the_parent_is_a_directory(tmp_path: Path) -> None:
    with (
        patch.object(Path, "mkdir", side_effect=FileExistsError("exists")),
        pytest.raises(FileExistsError, match="exists"),
    ):
        _make_parent_dir(tmp_path.joinpath("data.txt"))


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


#####################################
#     Tests for the removed lock    #
#####################################


def test_base_file_saver_save_lock_option_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'lock'"):
        SimpleFileSaver().save("hello", tmp_path.joinpath("data.txt"), lock=False)  # ty: ignore[unknown-argument]


def test_save_json_lock_option_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="lock"):
        save_json({}, tmp_path.joinpath("data.json"), lock=False)


def test_base_file_saver_save_leaves_only_the_file(tmp_path: Path) -> None:
    SimpleFileSaver().save("hello", tmp_path.joinpath("data.txt"))
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]


def test_base_file_saver_save_failure_leaves_the_directory_empty(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="failed to save"):
        FailingFileSaver().save("hello", tmp_path.joinpath("data.txt"))
    assert list(tmp_path.iterdir()) == []


def test_base_file_saver_save_fsync_opens_the_file_for_writing(tmp_path: Path) -> None:
    # Regression: the file was opened read-only, which ``fsync`` rejects on Windows.
    flags: list[int] = []
    real_open = os.open

    def spy(path: Any, flag: int, *args: Any) -> int:
        flags.append(flag)
        return real_open(path, flag, *args)

    with patch("coola.io.base.os.open", spy), patch("coola.io.base.os.fsync"):
        SimpleFileSaver().save("hello", tmp_path.joinpath("data.txt"), fsync=True)
    assert flags[0] & os.O_RDWR
