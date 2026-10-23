from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.io import (
    JsonLoader,
    JsonSaver,
    PickleLoader,
    PickleSaver,
    TextLoader,
    TextSaver,
    TorchLoader,
    TorchSaver,
    load_json,
    load_pickle,
    load_text,
    load_torch,
    save_json,
    save_pickle,
    save_text,
    save_torch,
)
from coola.testing.fixtures import torch_available

if TYPE_CHECKING:
    from pathlib import Path

###########################################
#     Tests for the str paths of the io   #
###########################################


@pytest.fixture
def str_path(tmp_path: Path) -> str:
    return str(tmp_path.joinpath("out", "data.bin"))


def test_save_text_str_path(str_path: str) -> None:
    save_text("hello", str_path)
    assert load_text(str_path) == "hello"


def test_save_json_str_path(str_path: str) -> None:
    save_json({"a": 1}, str_path)
    assert load_json(str_path) == {"a": 1}


def test_save_pickle_str_path(str_path: str) -> None:
    save_pickle({"a": 1}, str_path)
    assert load_pickle(str_path) == {"a": 1}


@torch_available
def test_save_torch_str_path(str_path: str) -> None:
    save_torch({"a": 1}, str_path)
    assert load_torch(str_path, weights_only=True) == {"a": 1}


def test_text_saver_str_path_creates_the_file(str_path: str, tmp_path: Path) -> None:
    TextSaver().save("hello", str_path)
    assert tmp_path.joinpath("out", "data.bin").is_file()


def test_text_loader_str_path(str_path: str) -> None:
    TextSaver().save("hello", str_path)
    assert TextLoader().load(str_path) == "hello"


def test_json_loader_str_path(str_path: str) -> None:
    JsonSaver().save([1, 2], str_path)
    assert JsonLoader().load(str_path) == [1, 2]


def test_json_loader_str_path_with_encoding(str_path: str) -> None:
    JsonSaver(encoding="utf-8").save(["é"], str_path)
    assert JsonLoader(encoding="utf-8").load(str_path) == ["é"]


def test_pickle_loader_str_path(str_path: str) -> None:
    PickleSaver().save([1, 2], str_path)
    assert PickleLoader().load(str_path) == [1, 2]


@torch_available
def test_torch_loader_str_path(str_path: str) -> None:
    TorchSaver().save([1, 2], str_path)
    assert TorchLoader().load(str_path) == [1, 2]


def test_save_text_str_path_exist_ok_false_raises(str_path: str) -> None:
    save_text("a", str_path)
    with pytest.raises(FileExistsError, match="already exists"):
        save_text("b", str_path)


def test_save_text_str_path_exist_ok_true_overwrites(str_path: str) -> None:
    save_text("a", str_path)
    save_text("b", str_path, exist_ok=True)
    assert load_text(str_path) == "b"


def test_save_text_str_path_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError, match="is a directory"):
        save_text("a", str(tmp_path))


def test_save_text_incorrect_path_type_raises() -> None:
    with pytest.raises(TypeError, match=r"expected str, bytes or os\.PathLike"):
        save_text("a", 1)


def test_load_text_incorrect_path_type_raises() -> None:
    with pytest.raises(TypeError, match=r"expected str, bytes or os\.PathLike"):
        load_text(1)
