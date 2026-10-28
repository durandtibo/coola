from __future__ import annotations

import logging
import sys
from collections import OrderedDict

import pytest

from coola.utils.imports import (
    is_module_imported,
    lazy_exports,
)

logger = logging.getLogger(__name__)


##############################
#     is_module_imported     #
##############################


@pytest.mark.parametrize("target", ["collections.Counter", "collections", "os.path.join"])
def test_is_module_imported_true(target: str) -> None:
    assert is_module_imported(target)


@pytest.mark.parametrize("target", ["missing_module.Name", "missing_module", "missing.sub.Name"])
def test_is_module_imported_false(target: str) -> None:
    assert not is_module_imported(target)


@pytest.mark.parametrize("target", [None, 1, ["collections.Counter"]])
def test_is_module_imported_not_a_string(target: object) -> None:
    assert is_module_imported(target)


def test_is_module_imported_does_not_import() -> None:
    is_module_imported("missing_module.Name")
    assert "missing_module" not in sys.modules


def test_is_module_imported_uses_parent_module() -> None:
    assert not is_module_imported("collections.missing_sub.Name")


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
