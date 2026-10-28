from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from coola.tools.file_size import get_file_size

if TYPE_CHECKING:
    from pathlib import Path

###################################
#     Tests for get_file_size     #
###################################


def test_get_file_size_file(tmp_path: Path) -> None:
    file = tmp_path / "data.txt"
    file.write_bytes(b"0123456789")

    assert get_file_size(file) == 10


def test_get_file_size_empty_file(tmp_path: Path) -> None:
    file = tmp_path / "empty.txt"
    file.touch()

    assert get_file_size(file) == 0


def test_get_file_size_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"12345")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_bytes(b"1234567890")

    assert get_file_size(tmp_path) == 15


def test_get_file_size_empty_directory(tmp_path: Path) -> None:
    assert get_file_size(tmp_path) == 0


def test_get_file_size_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match=r"path does not exist"):
        get_file_size(missing)


def test_get_file_size_returns_int(tmp_path: Path) -> None:
    file = tmp_path / "data.txt"
    file.write_bytes(b"123")

    assert isinstance(get_file_size(file), int)


def test_get_file_size_directory_ignores_symlinks(tmp_path: Path) -> None:
    tmp_path.joinpath("a.txt").write_text("abcde")
    tmp_path.joinpath("link.txt").symlink_to(tmp_path.joinpath("a.txt"))
    assert get_file_size(tmp_path) == 5


def test_get_file_size_directory_ignores_broken_symlinks(tmp_path: Path) -> None:
    tmp_path.joinpath("a.txt").write_text("abcde")
    tmp_path.joinpath("broken.txt").symlink_to(tmp_path.joinpath("missing.txt"))
    assert get_file_size(tmp_path) == 5


def test_get_file_size_ignores_a_file_that_vanishes_during_the_scan(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"12345")
    with patch("coola.tools.file_size.os.lstat", side_effect=FileNotFoundError):
        assert get_file_size(tmp_path) == 0
