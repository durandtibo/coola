r"""Contain fallback implementations used when ``pydantic`` dependency
is not available."""

from __future__ import annotations

__all__ = ["pydantic"]

from types import ModuleType

from coola.utils.fallback.factory import make_fake_class
from coola.utils.imports import raise_pydantic_missing_error

FakeClass = make_fake_class(raise_pydantic_missing_error)

# Create a fake pydantic package
pydantic: ModuleType = ModuleType("pydantic")
pydantic.BaseModel = FakeClass
pydantic.SecretBytes = FakeClass
pydantic.SecretStr = FakeClass
