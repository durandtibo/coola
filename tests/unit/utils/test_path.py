from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from coola.utils.path import (
    add_uuid_suffix,
    find_root_package_parent,
    sanitize_path,
)

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
    monkeypatch.setattr(Path, "is_file", lambda _: True)

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


def test_find_root_package_parent_missing_file(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    assert find_root_package_parent(pkg / "missing.py") == tmp_path


def test_sanitize_path_file_uri_with_host_raises() -> None:
    with pytest.raises(ValueError, match="host is not supported"):
        sanitize_path("file://server/share/x")


def test_sanitize_path_file_uri_with_localhost() -> None:
    assert sanitize_path("file://localhost/my/path") == Path("/my/path")


def test_sanitize_path_unknown_user_raises() -> None:
    with pytest.raises(ValueError, match="home directory"):
        sanitize_path("~coola_unknown_user_xyz/data")


def test_sanitize_path_resolve_false_keeps_a_relative_path() -> None:
    assert sanitize_path("data/a.txt", resolve=False) == Path("data/a.txt")


def test_sanitize_path_resolve_false_is_not_absolute() -> None:
    assert not sanitize_path("data/a.txt", resolve=False).is_absolute()


def test_sanitize_path_resolve_false_returns_a_path() -> None:
    assert isinstance(sanitize_path("data/a.txt", resolve=False), Path)


def test_sanitize_path_resolve_false_path_like() -> None:
    class PathLike:
        def __fspath__(self) -> str:
            return "data/a.txt"

    assert sanitize_path(PathLike(), resolve=False) == Path("data/a.txt")


def test_sanitize_path_resolve_false_expands_the_home_directory() -> None:
    assert sanitize_path("~/a.txt", resolve=False) == Path.home() / "a.txt"


def test_sanitize_path_resolve_false_decodes_a_file_uri() -> None:
    assert sanitize_path("file:///a/b.txt", resolve=False) == Path("/a/b.txt")


def test_sanitize_path_resolve_false_keeps_a_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.touch()
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    assert sanitize_path(link, resolve=False) == link


def test_sanitize_path_resolve_true_resolves_a_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.touch()
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    assert sanitize_path(link) == target.resolve()


def test_add_uuid_suffix_no_suffix(tmp_path: Path) -> None:
    with patch("coola.utils.path.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert add_uuid_suffix(tmp_path.joinpath("data")) == tmp_path.joinpath("data-a1b2c3")


def test_add_uuid_suffix_one_suffix(tmp_path: Path) -> None:
    with patch("coola.utils.path.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert add_uuid_suffix(tmp_path.joinpath("data.json")) == tmp_path.joinpath(
            "data-a1b2c3.json"
        )


def test_add_uuid_suffix_two_suffixes(tmp_path: Path) -> None:
    with patch("coola.utils.path.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert add_uuid_suffix(tmp_path.joinpath("data.tar.gz")) == tmp_path.joinpath(
            "data-a1b2c3.tar.gz"
        )


def test_add_uuid_suffix_dir(tmp_path: Path) -> None:
    with patch("coola.utils.path.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert add_uuid_suffix(tmp_path.joinpath("data/")) == tmp_path.joinpath("data-a1b2c3")


def test_add_uuid_suffix_dots_in_stem(tmp_path: Path) -> None:
    with patch("coola.utils.path.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert add_uuid_suffix(tmp_path.joinpath("v1.2.model")) == tmp_path.joinpath(
            "v1.2-a1b2c3.model"
        )


def test_add_uuid_suffix_compound_extension_with_dots_in_stem(tmp_path: Path) -> None:
    with patch("coola.utils.path.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert add_uuid_suffix(tmp_path.joinpath("v1.2.tar.gz")) == tmp_path.joinpath(
            "v1.2-a1b2c3.tar.gz"
        )
