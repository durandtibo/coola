r"""Contain fallback implementations used when ``pydantic`` dependency
is not available."""

from __future__ import annotations

__all__ = ["BaseModel", "SecretBytes", "SecretStr", "pydantic"]

from typing import TYPE_CHECKING

from coola.utils.fallback.factory import make_fake_class, make_fake_module
from coola.utils.imports import raise_pydantic_missing_error

if TYPE_CHECKING:
    from types import ModuleType

FakeClass: type = make_fake_class(raise_pydantic_missing_error)

BaseModel = FakeClass
SecretBytes = FakeClass
SecretStr = FakeClass

# Create a fake pydantic package
pydantic: ModuleType = make_fake_module(
    "pydantic", BaseModel=BaseModel, SecretBytes=SecretBytes, SecretStr=SecretStr
)
