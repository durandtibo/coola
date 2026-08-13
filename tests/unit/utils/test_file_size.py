from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.utils.file_size import get_file_size

if TYPE_CHECKING:
    from pathlib import Path

#####################################
#     Tests for get_file_size      #
#####################################


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
