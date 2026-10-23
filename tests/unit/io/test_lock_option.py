from __future__ import annotations

from typing import TYPE_CHECKING

from coola.io import load_json, load_pickle, load_text, save_json, save_pickle, save_text

if TYPE_CHECKING:
    from pathlib import Path

#############################################
#     Tests for the lock option of save     #
#############################################


def test_save_text_lock_false(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    save_text("hello", path, lock=False)
    assert load_text(path) == "hello"


def test_save_json_lock_false(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json({"a": 1}, path, lock=False)
    assert load_json(path) == {"a": 1}


def test_save_pickle_lock_false(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pkl")
    save_pickle({"a": 1}, path, lock=False)
    assert load_pickle(path) == {"a": 1}


def test_save_text_lock_false_creates_no_lock_file(tmp_path: Path) -> None:
    save_text("hello", tmp_path.joinpath("data.txt"), lock=False)
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]
