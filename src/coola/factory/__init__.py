r"""Contain factory utilities."""

from __future__ import annotations

__all__ = [
    "OBJECT_INIT",
    "OBJECT_TARGET",
    "factory",
    "import_object",
    "instantiate_object",
    "is_object_config",
    "resolve_object",
]

from coola.factory.config import is_object_config
from coola.factory.constants import OBJECT_INIT, OBJECT_TARGET
from coola.factory.instantiation import import_object, instantiate_object
from coola.factory.resolve import factory, resolve_object
