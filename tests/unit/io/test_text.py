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


@pytest.mark.parametrize("value", [None, 123, b"hello", [1, 2, 3]])
def test_text_saver_save_rejects_non_str(tmp_path: Path, value: object) -> None:
    with pytest.raises(TypeError, match=r"TextSaver expects a str, got"):
        TextSaver().save(value, tmp_path.joinpath("data.txt"))  # ty: ignore[invalid-argument-type]


def test_text_saver_save_rejects_non_str_message(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"got NoneType; call str\(\) on the data"):
        TextSaver().save(None, tmp_path.joinpath("data.txt"))  # ty: ignore[invalid-argument-type]


def test_text_saver_save_rejects_non_str_creates_no_file(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"expects a str"):
        TextSaver().save(None, tmp_path.joinpath("data.txt"))  # ty: ignore[invalid-argument-type]
    assert list(tmp_path.iterdir()) == []


def test_text_saver_save_rejects_non_str_keeps_existing_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    save_text("hello", path)
    with pytest.raises(TypeError, match=r"expects a str"):
        TextSaver().save(123, path, exist_ok=True)  # ty: ignore[invalid-argument-type]
    assert load_text(path) == "hello"


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


def test_text_saver_save_file_exist_ok_is_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    TextSaver().save("meow", path, exist_ok=True)
    assert path.is_file()


def test_text_saver_save_file_exist_ok_content(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    TextSaver().save("meow", path, exist_ok=True)
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


def test_save_text_none_raises(tmp_path: Path) -> None:
    # Regression: None used to be written as the text "None".
    with pytest.raises(TypeError, match=r"expects a str, got NoneType"):
        save_text(None, tmp_path.joinpath("data.txt"))  # ty: ignore[invalid-argument-type]


def test_save_text_int_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"expects a str, got int"):
        save_text(123, tmp_path.joinpath("data.txt"))  # ty: ignore[invalid-argument-type]


def test_save_text_str_of_list(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text(str([1, 2, 3]), path)
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


def test_save_text_file_exist_ok_is_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    save_text("meow", path, exist_ok=True)
    assert path.is_file()


def test_save_text_file_exist_ok_content(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    save_text("hello", path)
    save_text("meow", path, exist_ok=True)
    assert load_text(path) == "meow"


def test_save_text_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.txt")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        save_text("hello", path)


def test_save_text_load_text_preserves_line_endings_file_content(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    save_text("a\r\nb\rc\n", path)
    assert path.read_bytes() == b"a\r\nb\rc\n"


def test_save_text_load_text_preserves_line_endings_loaded_content(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    save_text("a\r\nb\rc\n", path)
    assert load_text(path) == "a\r\nb\rc\n"


###########################################
#     Tests for the str paths of text     #
###########################################


@pytest.fixture
def str_path(tmp_path: Path) -> str:
    return str(tmp_path.joinpath("out", "data.bin"))


def test_save_text_str_path(str_path: str) -> None:
    save_text("hello", str_path)
    assert load_text(str_path) == "hello"


def test_text_saver_str_path_creates_the_file(str_path: str, tmp_path: Path) -> None:
    TextSaver().save("hello", str_path)
    assert tmp_path.joinpath("out", "data.bin").is_file()


def test_text_loader_str_path(str_path: str) -> None:
    TextSaver().save("hello", str_path)
    assert TextLoader().load(str_path) == "hello"


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
        save_text("a", 1)  # ty: ignore[invalid-argument-type]


def test_load_text_incorrect_path_type_raises() -> None:
    with pytest.raises(TypeError, match=r"expected str, bytes or os\.PathLike"):
        load_text(1)  # ty: ignore[invalid-argument-type]
