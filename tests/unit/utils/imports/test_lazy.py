from __future__ import annotations

import logging
import os
from collections import OrderedDict

import pytest

from coola.utils.imports import (
    LazyModule,
    import_object,
    lazy_exports,
    lazy_import,
)

logger = logging.getLogger(__name__)


######################
#     LazyModule     #
######################


def test_lazy_module() -> None:
    os = LazyModule("os")
    assert isinstance(os.getcwd(), str)


def test_lazy_module_second_function() -> None:
    os = LazyModule("os")
    assert isinstance(os.cpu_count(), int)


def test_lazy_module_missing_attribute() -> None:
    os = LazyModule("os")
    with pytest.raises(AttributeError, match=r"module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_module_getattr_first() -> None:
    pathlib = LazyModule("pathlib")
    assert isinstance(pathlib.Path.cwd().as_posix(), str)


def test_lazy_module_getattr_first_then_dir() -> None:
    pathlib = LazyModule("pathlib")
    pathlib.Path.cwd()
    assert "Path" in dir(pathlib)


def test_lazy_module_dir_first() -> None:
    pathlib = LazyModule("pathlib")
    assert "Path" in dir(pathlib)


def test_lazy_module_dir_first_then_getattr() -> None:
    pathlib = LazyModule("pathlib")
    dir(pathlib)
    assert isinstance(pathlib.Path.cwd().as_posix(), str)


def test_lazy_module_missing_attribute_first() -> None:
    os = LazyModule("os")
    with pytest.raises(AttributeError, match=r"module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_module_missing() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"No module named 'missing'"):
        _ = LazyModule("missing").attr


#######################
#     lazy_import     #
#######################


def test_lazy_import() -> None:
    os = lazy_import("os")
    assert isinstance(os.getcwd(), str)


def test_lazy_import_second_function() -> None:
    os = lazy_import("os")
    assert isinstance(os.cpu_count(), int)


def test_lazy_import_missing_attribute() -> None:
    os = lazy_import("os")
    with pytest.raises(AttributeError, match=r"module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_import_getattr_first() -> None:
    pathlib = lazy_import("pathlib")
    assert isinstance(pathlib.Path.cwd().as_posix(), str)


def test_lazy_import_getattr_first_then_dir() -> None:
    pathlib = lazy_import("pathlib")
    pathlib.Path.cwd()
    assert "Path" in dir(pathlib)


def test_lazy_import_dir_first() -> None:
    pathlib = lazy_import("pathlib")
    assert "Path" in dir(pathlib)


def test_lazy_import_dir_first_then_getattr() -> None:
    pathlib = lazy_import("pathlib")
    dir(pathlib)
    assert isinstance(pathlib.Path.cwd().as_posix(), str)


def test_lazy_import_missing_attribute_first() -> None:
    os = lazy_import("os")
    with pytest.raises(AttributeError, match=r"module 'os' has no attribute 'missing'"):
        os.missing()


def test_lazy_import_missing() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"No module named 'missing'"):
        _ = lazy_import("missing").attr


def test_lazy_module_special_attribute_does_not_import() -> None:
    assert not hasattr(LazyModule("missing"), "__wrapped__")


def test_lazy_module_str_does_not_import() -> None:
    assert str(LazyModule("missing")) == "<module 'missing'>"


###########################
#     import_object       #
###########################


def test_import_object() -> None:
    assert import_object("collections:OrderedDict") is OrderedDict


def test_import_object_submodule() -> None:
    assert import_object("os.path:join") is os.path.join


@pytest.mark.parametrize("path", ["collections", "collections:", ":OrderedDict", ""])
def test_import_object_invalid_path(path: str) -> None:
    with pytest.raises(ValueError, match="Expected a path with the form"):
        import_object(path)


def test_import_object_missing_module() -> None:
    with pytest.raises(ModuleNotFoundError, match="missing_module"):
        import_object("missing_module:name")


def test_import_object_missing_name() -> None:
    with pytest.raises(AttributeError, match="missing_name"):
        import_object("collections:missing_name")


############################
#     lazy_exports         #
############################


def test_lazy_exports_getattr_imports_the_module() -> None:
    getattr_, _ = lazy_exports("example", {"OrderedDict": "collections"}, {})
    assert getattr_("OrderedDict") is OrderedDict


def test_lazy_exports_getattr_adds_the_name_to_the_namespace() -> None:
    namespace = {}
    getattr_, _ = lazy_exports("example", {"OrderedDict": "collections"}, namespace)
    getattr_("OrderedDict")
    assert namespace == {"OrderedDict": OrderedDict}


def test_lazy_exports_getattr_does_not_import_the_module_before_the_access() -> None:
    namespace = {}
    lazy_exports("example", {"OrderedDict": "collections"}, namespace)
    assert namespace == {}


def test_lazy_exports_getattr_unknown_name() -> None:
    getattr_, _ = lazy_exports("example", {"OrderedDict": "collections"}, {})
    with pytest.raises(AttributeError, match="module 'example' has no attribute 'missing'"):
        getattr_("missing")


def test_lazy_exports_dir_contains_the_lazy_names_and_the_namespace() -> None:
    _, dir_ = lazy_exports("example", {"OrderedDict": "collections"}, {"x": 1})
    assert dir_() == ["OrderedDict", "x"]
