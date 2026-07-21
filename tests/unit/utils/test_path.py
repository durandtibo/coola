from __future__ import annotations

from pathlib import Path

import pytest

from coola.utils.path import sanitize_path, working_directory

###################################
#     Tests for sanitize_path     #
###################################


def test_sanitize_path_empty_str() -> None:
    assert sanitize_path("") == Path.cwd()


def test_sanitize_path_str() -> None:
    assert sanitize_path("something") == Path.cwd().joinpath("something")


def test_sanitize_path_path(tmp_path: Path) -> None:
    assert sanitize_path(tmp_path) == tmp_path


def test_sanitize_path_resolve() -> None:
    assert sanitize_path(Path("something/./../")) == Path.cwd()


def test_sanitize_path_uri() -> None:
    assert sanitize_path("file:///my/path/something/./../") == Path("/my/path")


def test_sanitize_path_pathlike() -> None:
    class MyPathLike:
        def __fspath__(self) -> str:
            return "something/./../"

    assert sanitize_path(MyPathLike()) == Path.cwd()


def test_sanitize_path_pathlike_uri() -> None:
    class MyPathLike:
        def __fspath__(self) -> str:
            return "file:///my/path/something/./../"

    assert sanitize_path(MyPathLike()) == Path("/my/path")


def test_sanitize_path_path_uri_not_parsed() -> None:
    # Known limitation: unlike an equivalent str, a Path built from a
    # file URI is not recognized as a URI because Path already
    # collapses the "///" into a single "/" (e.g. "file:///a" becomes
    # "file:/a"), so it no longer starts with "file://" and is instead
    # treated as a plain relative path.
    path = Path("file:///my/path/something/./../")
    assert sanitize_path(path) == Path.cwd() / "file:/my/path"


#######################################
#     Tests for working_directory     #
#######################################


def test_working_directory_path() -> None:
    cwd_before = Path.cwd()
    new_path = cwd_before.parent
    with working_directory(new_path):
        assert Path.cwd() == new_path

    assert Path.cwd() == cwd_before


def test_working_directory_str() -> None:
    cwd_before = Path.cwd()
    new_path = cwd_before.parent
    with working_directory(str(new_path)):
        assert Path.cwd() == new_path

    assert Path.cwd() == cwd_before


def test_working_directory_pathlike() -> None:
    cwd_before = Path.cwd()
    new_path = cwd_before.parent

    class MyPathLike:
        def __fspath__(self) -> str:
            return str(new_path)

    with working_directory(MyPathLike()):
        assert Path.cwd() == new_path

    assert Path.cwd() == cwd_before


def test_working_directory_relative() -> None:
    cwd_before = Path.cwd()
    with working_directory(".."):
        assert Path.cwd() == cwd_before.parent

    assert Path.cwd() == cwd_before


def test_working_directory_nested() -> None:
    cwd_before = Path.cwd()
    parent = cwd_before.parent
    grandparent = parent.parent
    with working_directory(parent):
        assert Path.cwd() == parent
        with working_directory(grandparent):
            assert Path.cwd() == grandparent
        assert Path.cwd() == parent

    assert Path.cwd() == cwd_before


def test_working_directory_error() -> None:
    cwd_before = Path.cwd()
    with (  # noqa: PT012
        pytest.raises(RuntimeError, match=r"Exception"),
        working_directory(cwd_before.parent),
    ):
        msg = "Exception"
        raise RuntimeError(msg)

    assert Path.cwd() == cwd_before
