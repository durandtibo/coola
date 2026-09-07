from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import pytest

from coola.equality.interface import objects_are_equal
from coola.io import PickleLoader, PickleSaver, load_pickle, save_pickle, save_text

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def path_pickle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.pkl")
    PickleSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
    return path


##################################
#     Tests for PickleLoader     #
##################################


def test_pickle_loader_repr() -> None:
    assert repr(PickleLoader()) == "PickleLoader()"


def test_pickle_loader_str() -> None:
    assert str(PickleLoader()) == "PickleLoader()"


def test_pickle_loader_equal_true() -> None:
    assert PickleLoader().equal(PickleLoader())


def test_pickle_loader_equal_false() -> None:
    assert not PickleLoader().equal(PickleSaver())


def test_pickle_loader_equal_false_child() -> None:
    class Child(PickleLoader): ...

    assert not PickleLoader().equal(Child())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_pickle_loader_equal_nan(equal_nan: bool) -> None:
    assert PickleLoader().equal(PickleLoader(), equal_nan=equal_nan)


def test_pickle_loader_load(path_pickle: Path) -> None:
    assert PickleLoader().load(path_pickle) == {"key1": [1, 2, 3], "key2": "abc"}


#################################
#     Tests for PickleSaver     #
#################################


def test_pickle_saver_repr() -> None:
    assert repr(PickleSaver()) == "PickleSaver()"


def test_pickle_saver_repr_with_kwargs() -> None:
    assert repr(PickleSaver(protocol=5)) == "PickleSaver(protocol=5)"


def test_pickle_saver_str() -> None:
    assert str(PickleSaver()) == "PickleSaver()"


def test_pickle_saver_str_with_kwargs() -> None:
    assert str(PickleSaver(protocol=5)) == "PickleSaver(protocol=5)"


def test_pickle_saver_equal_true() -> None:
    assert PickleSaver().equal(PickleSaver())


def test_pickle_saver_equal_false_different_kwargs() -> None:
    assert not PickleSaver(protocol=5).equal(PickleSaver(protocol=4))


def test_pickle_saver_equal_false_different_type() -> None:
    assert not PickleSaver().equal(PickleLoader())


def test_pickle_saver_equal_false_different_type_child() -> None:
    class Child(PickleSaver): ...

    assert not PickleSaver().equal(Child())


@pytest.mark.parametrize("equal_nan", [True, False])
def test_pickle_saver_equal_nan(equal_nan: bool) -> None:
    assert PickleSaver().equal(PickleSaver(), equal_nan=equal_nan)


def test_pickle_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    saver = PickleSaver()
    saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


def test_pickle_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_text("hello", path)
    saver = PickleSaver()
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


def test_pickle_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_text("hello", path)
    saver = PickleSaver()
    saver.save({"key1": [3, 2, 1], "key2": "abc"}, path, exist_ok=True)
    assert path.is_file()
    assert load_pickle(path) == {"key1": [3, 2, 1], "key2": "abc"}


def test_pickle_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    path.mkdir(parents=True, exist_ok=True)
    saver = PickleSaver()
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


@pytest.mark.parametrize("protocol", list(range(1, pickle.HIGHEST_PROTOCOL + 1)))
def test_pickle_saver_save_protocol(tmp_path: Path, protocol: int) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    data = {"key1": [1, 2, 3], "key2": "abc"}
    saver = PickleSaver(protocol=protocol)
    saver.save(data, path)
    assert path.is_file()
    assert objects_are_equal(load_pickle(path), data)


#################################
#     Tests for load_pickle     #
#################################


def test_load_pickle(path_pickle: Path) -> None:
    assert load_pickle(path_pickle) == {"key1": [1, 2, 3], "key2": "abc"}


#################################
#     Tests for save_pickle     #
#################################


def test_save_pickle(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_pickle({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


def test_save_pickle_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_text("hello", path)
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        save_pickle({"key1": [1, 2, 3], "key2": "abc"}, path)


def test_save_pickle_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    save_text("hello", path)
    save_pickle({"key1": [3, 2, 1], "key2": "abc"}, path, exist_ok=True)
    assert path.is_file()
    assert load_pickle(path) == {"key1": [3, 2, 1], "key2": "abc"}


def test_save_pickle_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pkl")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        save_pickle({"key1": [1, 2, 3], "key2": "abc"}, path)


@pytest.mark.parametrize("protocol", list(range(1, pickle.HIGHEST_PROTOCOL + 1)))
def test_save_pickle_protocol(tmp_path: Path, protocol: int) -> None:
    path = tmp_path.joinpath("data", "data.pkl")
    data = {"key1": [1, 2, 3], "key2": "abc"}
    save_pickle(data, path, protocol=protocol)
    assert path.is_file()
    assert objects_are_equal(load_pickle(path), data)
