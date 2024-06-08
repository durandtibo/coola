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


#######################################
#     Tests for working_directory     #
#######################################


def test_working_directory() -> None:
    cwd_before = Path.cwd()
    new_path = cwd_before.parent
    with working_directory(new_path):
        assert Path.cwd() == new_path

    assert Path.cwd() == cwd_before


def test_working_directory_error() -> None:
    cwd_before = Path.cwd()
    with (  # noqa: PT012
        pytest.raises(RuntimeError, match="Exception"),
        working_directory(cwd_before.parent),
    ):
        msg = "Exception"
        raise RuntimeError(msg)

    assert Path.cwd() == cwd_before
