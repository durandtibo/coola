r"""Provide utilities to import and instantiate objects from configurations.

A configuration is a plain ``dict`` (for example loaded from JSON or YAML)
that names an object by its fully qualified name, under the key
``OBJECT_TARGET`` (``"_target_"``), and gives its keyword arguments. The
functions of the package:

- ``import_object``: import a class or a function from its fully
  qualified name.
- ``instantiate_object``: instantiate an object from a target and its
  arguments.
- ``factory``: import and instantiate an object from a fully qualified
  name, without any registration.
- ``resolve_object``: return an object as is, or build it from a
  configuration.
- ``is_object_config``: check if a configuration targets a given class.

Unlike ``coola.registry``, an object does not need to be registered: any
importable object can be built.
"""

from __future__ import annotations

__all__ = [
    "OBJECT_INIT",
    "OBJECT_PREFIXES",
    "OBJECT_TARGET",
    "factory",
    "import_object",
    "instantiate_object",
    "is_object_config",
    "resolve_object",
]

from coola.factory.config import is_object_config
from coola.factory.constants import OBJECT_INIT, OBJECT_PREFIXES, OBJECT_TARGET
from coola.factory.instantiation import import_object, instantiate_object
from coola.factory.resolve import factory, resolve_object
