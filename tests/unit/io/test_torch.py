from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.equality.interface import objects_are_equal
from coola.io import TorchLoader, TorchSaver, load_torch, save_text, save_torch
from coola.testing.fixtures import torch_available, torch_not_available
from coola.utils.imports import is_torch_available

if TYPE_CHECKING:
    from pathlib import Path

if is_torch_available():
    import torch


@pytest.fixture(scope="module")
def path_torch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tmp").joinpath("data.pt")
    TorchSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
    return path


################################
#     Tests for TorchLoader     #
################################


@torch_not_available
def test_torch_loader_no_torch() -> None:
    with pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."):
        TorchLoader()


@torch_available
def test_torch_loader_repr() -> None:
    assert repr(TorchLoader()) == "TorchLoader()"


@torch_available
def test_torch_loader_repr_with_kwargs() -> None:
    assert repr(TorchLoader(weights_only=False)) == "TorchLoader(weights_only=False)"


@torch_available
def test_torch_loader_str() -> None:
    assert str(TorchLoader()) == "TorchLoader()"


@torch_available
def test_torch_loader_equal_true() -> None:
    assert TorchLoader().equal(TorchLoader())


@torch_available
def test_torch_loader_equal_false_different_kwargs() -> None:
    assert not TorchLoader(weights_only=False).equal(TorchLoader(weights_only=True))


@torch_available
def test_torch_loader_equal_false_different_type() -> None:
    assert not TorchLoader().equal(TorchSaver())


@torch_available
def test_torch_loader_equal_false_different_type_child() -> None:
    class Child(TorchLoader): ...

    assert not TorchLoader().equal(Child())


@torch_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_torch_loader_equal_nan(equal_nan: bool) -> None:
    assert TorchLoader().equal(TorchLoader(), equal_nan=equal_nan)


@torch_available
def test_torch_loader_load(path_torch: Path) -> None:
    assert TorchLoader().load(path_torch) == {"key1": [1, 2, 3], "key2": "abc"}


@torch_available
def test_torch_loader_load_tensor(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tensor.pt")
    tensor = torch.ones(2, 3)
    TorchSaver().save(tensor, path)
    assert objects_are_equal(TorchLoader().load(path), tensor)


###############################
#     Tests for TorchSaver     #
###############################


@torch_not_available
def test_torch_saver_no_torch() -> None:
    with pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."):
        TorchSaver()


@torch_available
def test_torch_saver_repr() -> None:
    assert repr(TorchSaver()) == "TorchSaver()"


@torch_available
def test_torch_saver_repr_with_kwargs() -> None:
    assert repr(TorchSaver(pickle_protocol=5)) == "TorchSaver(pickle_protocol=5)"


@torch_available
def test_torch_saver_str() -> None:
    assert str(TorchSaver()) == "TorchSaver()"


@torch_available
def test_torch_saver_equal_true() -> None:
    assert TorchSaver().equal(TorchSaver())


@torch_available
def test_torch_saver_equal_false_different_kwargs() -> None:
    assert not TorchSaver(pickle_protocol=5).equal(TorchSaver(pickle_protocol=4))


@torch_available
def test_torch_saver_equal_false_different_type() -> None:
    assert not TorchSaver().equal(TorchLoader())


@torch_available
def test_torch_saver_equal_false_different_type_child() -> None:
    class Child(TorchSaver): ...

    assert not TorchSaver().equal(Child())


@torch_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_torch_saver_equal_nan(equal_nan: bool) -> None:
    assert TorchSaver().equal(TorchSaver(), equal_nan=equal_nan)


@torch_available
def test_torch_saver_save(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    saver = TorchSaver()
    saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


@torch_available
def test_torch_saver_save_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    save_text("hello", path)
    saver = TorchSaver()
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


@torch_available
def test_torch_saver_save_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    save_text("hello", path)
    saver = TorchSaver()
    saver.save({"key1": [3, 2, 1], "key2": "abc"}, path, exist_ok=True)
    assert path.is_file()
    assert load_torch(path) == {"key1": [3, 2, 1], "key2": "abc"}


@torch_available
def test_torch_saver_save_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    path.mkdir(parents=True, exist_ok=True)
    saver = TorchSaver()
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        saver.save({"key1": [1, 2, 3], "key2": "abc"}, path)


################################
#     Tests for load_torch     #
################################


@torch_available
def test_load_torch(path_torch: Path) -> None:
    assert load_torch(path_torch) == {"key1": [1, 2, 3], "key2": "abc"}


################################
#     Tests for save_torch     #
################################


@torch_available
def test_save_torch(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    save_torch({"key1": [1, 2, 3], "key2": "abc"}, path)
    assert path.is_file()


@torch_available
def test_save_torch_file_exist(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    save_text("hello", path)
    with pytest.raises(FileExistsError, match=r"path .* already exists."):
        save_torch({"key1": [1, 2, 3], "key2": "abc"}, path)


@torch_available
def test_save_torch_file_exist_ok(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    save_text("hello", path)
    save_torch({"key1": [3, 2, 1], "key2": "abc"}, path, exist_ok=True)
    assert path.is_file()
    assert load_torch(path) == {"key1": [3, 2, 1], "key2": "abc"}


@torch_available
def test_save_torch_file_exist_ok_dir(tmp_path: Path) -> None:
    path = tmp_path.joinpath("tmp/data.pt")
    path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IsADirectoryError, match=r"path .* is a directory"):
        save_torch({"key1": [1, 2, 3], "key2": "abc"}, path)
