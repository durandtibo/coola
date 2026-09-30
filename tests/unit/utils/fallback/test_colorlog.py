from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.colorlog import colorlog


def test_colorlog_is_module_type() -> None:
    assert isinstance(colorlog, ModuleType)


def test_colorlog_module_name() -> None:
    assert colorlog.__name__ == "colorlog"


@pytest.mark.parametrize("name", ["StreamHandler", "ColoredFormatter"])
def test_colorlog_class_is_class(name: str) -> None:
    assert isinstance(getattr(colorlog, name), type)


@pytest.mark.parametrize("name", ["StreamHandler", "ColoredFormatter"])
def test_colorlog_class_instantiation(name: str) -> None:
    with pytest.raises(RuntimeError, match=r"'colorlog' package is required but not installed."):
        getattr(colorlog, name)()


def test_colorlog_class_instantiation_with_args() -> None:
    with pytest.raises(RuntimeError, match=r"'colorlog' package is required but not installed."):
        colorlog.ColoredFormatter("%(message)s")
