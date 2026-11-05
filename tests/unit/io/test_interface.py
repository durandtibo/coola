from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.io import load, load_json, load_pickle, load_text, save, save_json
from coola.testing.marks import torch_available

if TYPE_CHECKING:
    from pathlib import Path

DATA = {"key": [1, 2, 3]}

##########################
#     Tests for save     #
##########################


@pytest.mark.parametrize("name", ["data.json", "data.JSON"])
def test_save_json(tmp_path: Path, name: str) -> None:
    save(DATA, tmp_path / name)
    assert load_json(tmp_path / name) == DATA


@pytest.mark.parametrize("name", ["data.pkl", "data.pickle"])
def test_save_pickle(tmp_path: Path, name: str) -> None:
    save(DATA, tmp_path / name)
    assert load_pickle(tmp_path / name) == DATA


def test_save_text(tmp_path: Path) -> None:
    save("abc", tmp_path / "data.txt")
    assert load_text(tmp_path / "data.txt") == "abc"


def test_save_str_path(tmp_path: Path) -> None:
    save(DATA, str(tmp_path / "data.json"))
    assert load_json(tmp_path / "data.json") == DATA


def test_save_keyword_arguments_of_the_saver(tmp_path: Path) -> None:
    save({"a": 1}, tmp_path / "data.json", indent=2)
    assert (tmp_path / "data.json").read_text() == '{\n  "a": 1\n}'


def test_save_file_exists_raises(tmp_path: Path) -> None:
    save_json(DATA, tmp_path / "data.json")
    with pytest.raises(FileExistsError, match=r"already exists"):
        save(DATA, tmp_path / "data.json")


def test_save_exist_ok(tmp_path: Path) -> None:
    save_json(DATA, tmp_path / "data.json")
    save({"b": 2}, tmp_path / "data.json", exist_ok=True)
    assert load_json(tmp_path / "data.json") == {"b": 2}


def test_save_fsync(tmp_path: Path) -> None:
    save(DATA, tmp_path / "data.json", fsync=True)
    assert load_json(tmp_path / "data.json") == DATA


@pytest.mark.parametrize("name", ["data.csv", "data", "data.json.bak"])
def test_save_unsupported_suffix_raises(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match=r"suffix of the path"):
        save(DATA, tmp_path / name)


def test_save_unsupported_suffix_does_not_create_a_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"suffix of the path"):
        save(DATA, tmp_path / "data.csv")
    assert list(tmp_path.iterdir()) == []


@torch_available
@pytest.mark.parametrize("name", ["data.pt", "data.pth"])
def test_save_torch(tmp_path: Path, name: str) -> None:
    save(DATA, tmp_path / name)
    assert load(tmp_path / name) == DATA


##########################
#     Tests for load     #
##########################


@pytest.mark.parametrize("name", ["data.json", "data.JSON"])
def test_load_json(tmp_path: Path, name: str) -> None:
    save_json(DATA, tmp_path / name)
    assert load(tmp_path / name) == DATA


def test_load_pickle(tmp_path: Path) -> None:
    save(DATA, tmp_path / "data.pkl")
    assert load(tmp_path / "data.pkl") == DATA


def test_load_text(tmp_path: Path) -> None:
    save("abc", tmp_path / "data.txt")
    assert load(tmp_path / "data.txt") == "abc"


def test_load_str_path(tmp_path: Path) -> None:
    save_json(DATA, tmp_path / "data.json")
    assert load(str(tmp_path / "data.json")) == DATA


def test_load_keyword_arguments_of_the_loader(tmp_path: Path) -> None:
    save("é", tmp_path / "data.txt", encoding="latin-1")
    assert load(tmp_path / "data.txt", encoding="latin-1") == "é"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"data.json"):
        load(tmp_path / "data.json")


@pytest.mark.parametrize("name", ["data.csv", "data", "data.json.bak"])
def test_load_unsupported_suffix_raises(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match=r"suffix of the path"):
        load(tmp_path / name)
