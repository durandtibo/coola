from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.io import TextLoader, TextSaver, load_text, save_text

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_text(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.txt")
    save_text("hello", path)
    return path


################################
#     Tests for TextLoader     #
################################

# --- str ---


def test_text_loader_repr() -> None:
    assert repr(TextLoader()) == "TextLoader(encoding='utf-8')"


def test_text_loader_str() -> None:
    assert str(TextLoader()) == "TextLoader(encoding=utf-8)"


# --- equal ---


def test_text_loader_equal_true() -> None:
    assert TextLoader().equal(TextLoader())


def test_text_loader_equal_false() -> None:
    assert not TextLoader().equal(TextSaver())


def test_text_loader_equal_false_child() -> None:
    class Child(TextLoader): ...

    assert not TextLoader().equal(Child())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_text_loader_equal_nan(equal_nan: bool) -> None:
    assert TextLoader().equal(TextLoader(), equal_nan=equal_nan)


def test_text_loader_equal_false_different_encoding() -> None:
    assert not TextLoader(encoding="utf-8").equal(TextLoader(encoding="latin-1"))


def test_text_loader_equal_true_same_encoding() -> None:
    assert TextLoader(encoding="latin-1").equal(TextLoader(encoding="latin-1"))


# --- load ---


def test_text_loader_load(path_text: Path) -> None:
    assert TextLoader().load(path_text) == "hello"


def test_text_loader_load_respects_encoding(tmp_path: Path) -> None:
    content = "Résultats: €1.2B"
    path = tmp_path / "data.txt"
    path.write_text(content, encoding="utf-8")
    assert TextLoader(encoding="utf-8").load(path) == content


###############################
#     Tests for TextSaver     #
###############################

# --- str ---


def test_text_saver_repr() -> None:
    assert repr(TextSaver()) == "TextSaver(encoding='utf-8')"


def test_text_saver_str() -> None:
    assert str(TextSaver()) == "TextSaver(encoding=utf-8)"


# --- equal ---


def test_text_saver_equal_true() -> None:
    assert TextSaver().equal(TextSaver())


def test_text_saver_equal_false() -> None:
    assert not TextSaver().equal(TextLoader())


def test_text_saver_equal_false_child() -> None:
    class Child(TextSaver): ...

    assert not TextSaver().equal(Child())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_text_saver_equal_nan(equal_nan: bool) -> None:
    assert TextSaver().equal(TextSaver(), equal_nan=equal_nan)


def test_text_saver_equal_false_different_encoding() -> None:
    assert not TextSaver(encoding="utf-8").equal(TextSaver(encoding="latin-1"))


def test_text_saver_equal_true_same_encoding() -> None:
    assert TextSaver(encoding="latin-1").equal(TextSaver(encoding="latin-1"))


# --- save ---


def test_text_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    TextSaver().save("hello", path)
    assert path.is_file()


def test_text_saver_save_list(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    TextSaver().save([1, 2, 3], path)
    assert path.is_file()
    assert load_text(path) == "[1, 2, 3]"


def test_text_saver_save_respects_encoding(tmp_path: Path) -> None:
    content = "Résultats: €1.2B"
    path = tmp_path / "data.txt"
    TextSaver(encoding="utf-8").save(content, path)
    assert load_text(path, encoding="utf-8") == content


def test_text_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        TextSaver().save("hello", path)


def test_text_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    TextSaver().save("meow", path, exist_ok=True)
    assert path.is_file()
    assert load_text(path) == "meow"


def test_text_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        TextSaver().save("hello", path)


###############################
#     Tests for load_text     #
###############################


def test_load_text(path_text: Path) -> None:
    assert load_text(path_text) == "hello"


def test_load_text_respects_encoding(tmp_path: Path) -> None:
    content = "Résultats: €1.2B"
    path = tmp_path / "data.txt"
    path.write_text(content, encoding="utf-8")
    assert load_text(path, encoding="utf-8") == content


###############################
#     Tests for save_text     #
###############################


def test_save_text(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    assert path.is_file()


def test_save_text_list(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text([1, 2, 3], path)
    assert path.is_file()
    assert load_text(path) == "[1, 2, 3]"


def test_save_text_respects_encoding(tmp_path: Path) -> None:
    content = "Résultats: €1.2B"
    path = tmp_path / "data.txt"
    save_text(content, path, encoding="utf-8")
    assert load_text(path, encoding="utf-8") == content


def test_save_text_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        save_text("hello", path)


def test_save_text_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    save_text("meow", path, exist_ok=True)
    assert path.is_file()
    assert load_text(path) == "meow"


def test_save_text_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        save_text("hello", path)
