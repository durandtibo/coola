from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.io import JsonLoader, JsonSaver, load_json, save_json

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_json(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.json")
    save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
    return path


################################
#     Tests for JsonLoader     #
################################


def test_json_loader_repr() -> None:
    assert repr(JsonLoader()) == "JsonLoader()"


def test_json_loader_str() -> None:
    assert str(JsonLoader()) == "JsonLoader()"


def test_json_loader_equal_true() -> None:
    assert JsonLoader().equal(JsonLoader())


def test_json_loader_equal_false() -> None:
    assert not JsonLoader().equal(JsonSaver())


def test_json_loader_equal_false_child() -> None:
    class Child(JsonLoader): ...

    assert not JsonLoader().equal(Child())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_json_loader_equal_nan(equal_nan: bool) -> None:
    assert JsonLoader().equal(JsonLoader(), equal_nan=equal_nan)


def test_json_loader_load(path_json: Path) -> None:
    assert JsonLoader().load(path_json) == {"key1": [1, 2, 3], "key2": "abc"}


###############################
#     Tests for JsonSaver     #
###############################


def test_json_saver_repr() -> None:
    assert repr(JsonSaver()) == "JsonSaver(encoding='utf-8')"


def test_json_saver_str() -> None:
    assert str(JsonSaver()) == "JsonSaver(encoding=utf-8)"


def test_json_saver_equal_true() -> None:
    assert JsonSaver().equal(JsonSaver())


def test_json_saver_equal_false() -> None:
    assert not JsonSaver().equal(JsonLoader())


def test_json_saver_equal_false_child() -> None:
    class Child(JsonSaver): ...

    assert not JsonSaver().equal(Child())


def test_json_saver_equal_false_encoding() -> None:
    assert not JsonSaver(encoding="utf-8").equal(JsonSaver(encoding="latin-1"))


def test_json_saver_equal_true_encoding() -> None:
    assert JsonSaver(encoding="latin-1").equal(JsonSaver(encoding="latin-1"))


@pytest.mark.parametrize("equal_nan", [True, False])
def test_json_saver_equal_nan(equal_nan: bool) -> None:
    assert JsonSaver().equal(JsonSaver(), equal_nan=equal_nan)


def test_json_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    saver = JsonSaver()
    saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


def test_json_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
    saver = JsonSaver()
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


def test_json_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
    saver = JsonSaver()
    saver.save({"key1": [3, 2, 1], "key2": "meow"}, path, exist_ok=True)
    assert path.is_file()
    assert load_json(path) == {"key1": [3, 2, 1], "key2": "meow"}


def test_json_saver_save_respects_encoding(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    content = {"key": "Résultats: €1.2B"}
    JsonSaver(encoding="utf-8").save(content, path)
    assert path.read_text(encoding="utf-8") == '{"key": "R\\u00e9sultats: \\u20ac1.2B"}'
    assert JsonLoader().load(path) == content


def test_json_saver_save_not_serializable_no_leftover_tmp_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    saver = JsonSaver()
    with pytest.raises(TypeError):
        saver.save({"key1": object()}, path)
    assert not path.is_file()
    assert list(path.parent.iterdir()) == []


def test_json_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    path.mkdir(parents=True, exist_ok=True)
    saver = JsonSaver()
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


###############################
#     Tests for load_json     #
###############################


def test_load_json(path_json: Path) -> None:
    assert load_json(path_json) == {"key1": [1, 2, 3], "key2": "abc"}


###############################
#     Tests for save_json     #
###############################


def test_save_json(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


def test_save_json_respects_encoding(tmp_path: Path) -> None:
    content = {"key": "Résultats: €1.2B"}
    path = tmp_path.joinpath("data.json")
    save_json(content, path, encoding="utf-8")
    assert load_json(path) == content


def test_save_json_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        save_json({"key1": [1, 2, 3], "key2": "abc"}, path)


def test_save_json_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
    save_json({"key1": [3, 2, 1], "key2": "meow"}, path, exist_ok=True)
    assert path.is_file()
    assert load_json(path) == {"key1": [3, 2, 1], "key2": "meow"}


def test_save_json_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
