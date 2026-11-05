from __future__ import annotations

import contextlib
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
    assert repr(JsonLoader()) == "JsonLoader(encoding=None)"


def test_json_loader_str() -> None:
    assert str(JsonLoader()) == "JsonLoader(encoding=None)"


def test_json_loader_equal_true() -> None:
    assert JsonLoader().equal(JsonLoader())


def test_json_loader_equal_true_encoding() -> None:
    assert JsonLoader(encoding="latin-1").equal(JsonLoader(encoding="latin-1"))


def test_json_loader_equal_false_different_encoding_default_encoding() -> None:
    assert not JsonLoader(encoding="latin-1").equal(JsonLoader())


def test_json_loader_equal_false_different_encoding_other_encoding() -> None:
    assert not JsonLoader(encoding="latin-1").equal(JsonLoader(encoding="utf-8"))


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
    assert repr(JsonSaver()) == "JsonSaver(encoding='utf-8', allow_nan=False, jsonable=False)"


def test_json_saver_str() -> None:
    assert str(JsonSaver()) == "JsonSaver(encoding=utf-8, allow_nan=False, jsonable=False)"


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


def test_json_saver_save_file_exist_ok_is_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
    saver = JsonSaver()
    saver.save({"key1": [3, 2, 1], "key2": "meow"}, path, exist_ok=True)
    assert path.is_file()


def test_json_saver_save_file_exist_ok_content(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
    saver = JsonSaver()
    saver.save({"key1": [3, 2, 1], "key2": "meow"}, path, exist_ok=True)
    assert load_json(path) == {"key1": [3, 2, 1], "key2": "meow"}


def test_json_saver_save_respects_encoding_file_content(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    content = {"key": "Résultats: €1.2B"}
    JsonSaver(encoding="utf-8").save(content, path)
    assert path.read_text(encoding="utf-8") == '{"key": "R\\u00e9sultats: \\u20ac1.2B"}'


def test_json_saver_save_respects_encoding_loaded_content(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    content = {"key": "Résultats: €1.2B"}
    JsonSaver(encoding="utf-8").save(content, path)
    assert JsonLoader().load(path) == content


def test_json_saver_save_not_serializable_no_leftover_tmp_file_raises(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    saver = JsonSaver()
    with pytest.raises(TypeError, match="not JSON serializable"):
        saver.save({"key1": object()}, path)


def test_json_saver_save_not_serializable_no_leftover_tmp_file_no_leftover_tmp_file_target_is_not_created(
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    saver = JsonSaver()
    with contextlib.suppress(TypeError):
        saver.save({"key1": object()}, path)
    assert not path.is_file()


def test_json_saver_save_not_serializable_no_leftover_tmp_file_no_leftover_tmp_file_directory_is_empty(
    tmp_path: Path,
) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    saver = JsonSaver()
    with contextlib.suppress(TypeError):
        saver.save({"key1": object()}, path)
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


def test_save_json_file_exist_ok_is_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
    save_json({"key1": [3, 2, 1], "key2": "meow"}, path, exist_ok=True)
    assert path.is_file()


def test_save_json_file_exist_ok_content(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
    save_json({"key1": [3, 2, 1], "key2": "meow"}, path, exist_ok=True)
    assert load_json(path) == {"key1": [3, 2, 1], "key2": "meow"}


def test_save_json_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.json")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        save_json({"key1": [1, 2, 3], "key2": "abc"}, path)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "latin-1", "cp1252"])
def test_json_saver_loader_round_trip_with_encoding(tmp_path: Path, encoding: str) -> None:
    path = tmp_path.joinpath("data.json")
    data = {"key": "\u00e9t\u00e9"}
    JsonSaver(encoding=encoding, ensure_ascii=False).save(data, path)
    assert JsonLoader(encoding=encoding).load(path) == data


def test_load_json_encoding(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json({"key": "\u00e9t\u00e9"}, path, encoding="latin-1", ensure_ascii=False)
    assert load_json(path, encoding="latin-1") == {"key": "\u00e9t\u00e9"}


def test_load_json_without_encoding_detects_utf16(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json({"key": "\u00e9t\u00e9"}, path, encoding="utf-16", ensure_ascii=False)
    assert load_json(path) == {"key": "\u00e9t\u00e9"}


def test_json_loader_without_encoding_cannot_read_latin1(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json({"key": "\u00e9t\u00e9"}, path, encoding="latin-1", ensure_ascii=False)
    with pytest.raises(UnicodeDecodeError, match="invalid continuation byte"):
        load_json(path)


###########################################
#     Tests for the str paths of json     #
###########################################


@pytest.fixture
def str_path(tmp_path: Path) -> str:
    return str(tmp_path.joinpath("out", "data.bin"))


def test_save_json_str_path(str_path: str) -> None:
    save_json({"a": 1}, str_path)
    assert load_json(str_path) == {"a": 1}


def test_json_loader_str_path(str_path: str) -> None:
    JsonSaver().save([1, 2], str_path)
    assert JsonLoader().load(str_path) == [1, 2]


def test_json_loader_str_path_with_encoding(str_path: str) -> None:
    JsonSaver(encoding="utf-8").save(["é"], str_path)
    assert JsonLoader(encoding="utf-8").load(str_path) == ["é"]


################################
#     Tests for allow_nan      #
################################


def test_json_saver_save_nan_raises_by_default(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Out of range float values are not JSON compliant"):
        JsonSaver().save({"a": float("nan")}, tmp_path.joinpath("data.json"))


def test_json_saver_save_infinity_raises_by_default(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Out of range float values are not JSON compliant"):
        JsonSaver().save([float("inf")], tmp_path.joinpath("data.json"))


def test_json_saver_save_nan_failure_leaves_no_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    with contextlib.suppress(ValueError):
        JsonSaver().save([float("nan")], path)
    assert not path.exists()


def test_json_saver_save_allow_nan(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    JsonSaver(allow_nan=True).save([float("inf")], path)
    assert path.read_text() == "[Infinity]"


def test_save_json_nan_raises_by_default(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Out of range float values are not JSON compliant"):
        save_json([float("nan")], tmp_path.joinpath("data.json"))


def test_save_json_allow_nan(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json([float("nan")], path, allow_nan=True)
    assert path.read_text() == "[NaN]"


def test_json_saver_allow_nan_is_keyword_only() -> None:
    with pytest.raises(TypeError, match=r"positional argument"):
        JsonSaver("utf-8", True)  # ty: ignore[too-many-positional-arguments]


def test_json_saver_repr_shows_allow_nan() -> None:
    assert "allow_nan=False" in repr(JsonSaver())


def test_json_saver_equal_false_allow_nan() -> None:
    assert not JsonSaver(allow_nan=True).equal(JsonSaver())


###############################
#     Tests for jsonable      #
###############################


def test_json_saver_save_without_jsonable_a_tuple_key_value_is_a_list(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    JsonSaver().save({"a": (1, 2)}, path)
    assert load_json(path) == {"a": [1, 2]}


def test_json_saver_save_without_jsonable_a_set_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"Object of type set is not JSON serializable"):
        JsonSaver().save({"a": {1}}, tmp_path.joinpath("data.json"))


def test_json_saver_save_jsonable(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    JsonSaver(jsonable=True).save({"a": {2, 1}}, path)
    assert load_json(path) == {"a": [1, 2]}


def test_json_saver_save_jsonable_unsupported_type_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"Cannot convert an object of type object to JSON"):
        JsonSaver(jsonable=True).save(object(), tmp_path.joinpath("data.json"))


def test_save_json_jsonable(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json({"a": {2, 1}}, path, jsonable=True)
    assert load_json(path) == {"a": [1, 2]}


def test_save_json_without_jsonable_a_set_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"Object of type set is not JSON serializable"):
        save_json({"a": {1}}, tmp_path.joinpath("data.json"))


def test_json_saver_equal_false_jsonable() -> None:
    assert not JsonSaver(jsonable=True).equal(JsonSaver())
