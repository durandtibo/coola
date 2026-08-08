from __future__ import annotations

from pathlib import Path

import pytest

from coola.utils.path import find_root_package_parent, sanitize_path, working_directory

##############################################
#     Tests for find_root_package_parent     #
##############################################


# --- Nested package resolution ---


def test_find_root_package_parent_from_file_in_nested_package(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    sub = pkg / "sub"
    sub.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    (sub / "__init__.py").touch()
    module = sub / "module.py"
    module.touch()

    assert find_root_package_parent(module) == tmp_path


def test_find_root_package_parent_from_directory_in_nested_package(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    sub = pkg / "sub"
    sub.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    (sub / "__init__.py").touch()

    assert find_root_package_parent(sub) == tmp_path


def test_find_root_package_parent_from_top_level_package_file(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    module = pkg / "module.py"
    module.touch()

    assert find_root_package_parent(module) == tmp_path


def test_find_root_package_parent_stops_at_filesystem_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate every directory (including the real filesystem root) containing
    # an __init__.py, forcing the walk to continue all the way up until
    # parent == current at the filesystem root.
    monkeypatch.setattr(Path, "is_file", lambda self: True)  # noqa: ARG005

    result = find_root_package_parent(tmp_path)

    assert result == result.parent


# --- Non-package input ---


def test_find_root_package_parent_file_without_init_returns_own_dir(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.touch()

    assert find_root_package_parent(module) == tmp_path


def test_find_root_package_parent_directory_without_init_returns_itself(tmp_path: Path) -> None:
    directory = tmp_path / "not_a_package"
    directory.mkdir()

    assert find_root_package_parent(directory) == directory


# --- String input ---


def test_find_root_package_parent_accepts_string_path(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    module = pkg / "module.py"
    module.touch()

    assert find_root_package_parent(str(module)) == tmp_path


# --- Return type ---


def test_find_root_package_parent_returns_path_instance(tmp_path: Path) -> None:
    directory = tmp_path / "not_a_package"
    directory.mkdir()

    assert isinstance(find_root_package_parent(directory), Path)


# --- Relative path resolution ---


def test_find_root_package_parent_resolves_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    module = pkg / "module.py"
    module.touch()

    monkeypatch.chdir(tmp_path)
    assert find_root_package_parent(Path("pkg") / "module.py") == tmp_path


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
