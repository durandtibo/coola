from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.pydantic import pydantic


def test_pydantic_is_module_type() -> None:
    assert isinstance(pydantic, ModuleType)


def test_pydantic_module_name() -> None:
    assert pydantic.__name__ == "pydantic"


def test_pydantic_base_model_class_exists() -> None:
    assert hasattr(pydantic, "BaseModel")


def test_pydantic_base_model_is_class() -> None:
    assert isinstance(pydantic.BaseModel, type)


def test_pydantic_base_model_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'pydantic' package is required but not installed."):
        pydantic.BaseModel()


def test_pydantic_base_model_instantiation_with_args() -> None:
    with pytest.raises(RuntimeError, match=r"'pydantic' package is required but not installed."):
        pydantic.BaseModel([1, 2, 3])
