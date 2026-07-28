r"""Contain fallback implementations used when ``pydantic`` dependency
is not available."""

from __future__ import annotations

__all__ = ["BaseModel", "SecretBytes", "SecretStr", "pydantic"]

from types import ModuleType

from coola.utils.fallback.factory import make_fake_class
from coola.utils.imports import raise_pydantic_missing_error

FakeClass: type = make_fake_class(raise_pydantic_missing_error)

BaseModel = FakeClass
SecretBytes = FakeClass
SecretStr = FakeClass

# Create a fake pydantic package
pydantic: ModuleType = ModuleType("pydantic")
pydantic.BaseModel = BaseModel
pydantic.SecretBytes = SecretBytes
pydantic.SecretStr = SecretStr
