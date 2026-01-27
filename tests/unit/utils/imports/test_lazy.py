from __future__ import annotations

import logging

import pytest

from coola.utils.imports import (
    LazyModule,
    lazy_import,
)

logger = logging.getLogger(__name__)


######################
#     LazyModule     #
######################


def test_lazy_module() -> None:
    os = LazyModule("os")
    assert isinstance(os.getcwd(), str)
    assert isinstance(os.cpu_count(), int)
    with pytest.raises(AttributeError, match=r"module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_module_getattr_first() -> None:
    pathlib = LazyModule("pathlib")
    assert isinstance(pathlib.Path.cwd().as_posix(), str)
    assert "Path" in dir(pathlib)


def test_lazy_module_dir_first() -> None:
    pathlib = LazyModule("pathlib")
    assert "Path" in dir(pathlib)
    assert isinstance(pathlib.Path.cwd().as_posix(), str)


def test_lazy_module_missing_attribute_first() -> None:
    os = LazyModule("os")
    with pytest.raises(AttributeError, match=r"module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_module_missing() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"No module named 'missing'"):
        str(LazyModule("missing"))


#######################
#     lazy_import     #
#######################


def test_lazy_import() -> None:
    os = lazy_import("os")
    assert isinstance(os.getcwd(), str)
    assert isinstance(os.cpu_count(), int)
    with pytest.raises(AttributeError, match=r"module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_import_getattr_first() -> None:
    pathlib = lazy_import("pathlib")
    assert isinstance(pathlib.Path.cwd().as_posix(), str)
    assert "Path" in dir(pathlib)


def test_lazy_import_dir_first() -> None:
    pathlib = lazy_import("pathlib")
    assert "Path" in dir(pathlib)
    assert isinstance(pathlib.Path.cwd().as_posix(), str)


def test_lazy_import_missing_attribute_first() -> None:
    os = lazy_import("os")
    with pytest.raises(AttributeError, match=r"module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_import_missing() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"No module named 'missing'"):
        str(lazy_import("missing"))
