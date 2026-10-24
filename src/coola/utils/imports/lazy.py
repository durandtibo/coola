r"""Implement some utility functions to lazily import a module."""

from __future__ import annotations

__all__ = ["LazyModule", "import_object", "is_module_imported", "lazy_exports", "lazy_import"]

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


# The implementation of LazyModule is based on
# https://github.com/Lightning-AI/utilities/blob/b4232544a6e974fee2bebeb029849bd53916bbda/src/lightning_utilities/core/imports.py#L254C1-L292C52
# with license found in the licenses/LICENSE_lightning_utilities
class LazyModule(ModuleType):
    r"""Define a proxy module that lazily imports a module.

    The module is imported the first time it is actually used.

    Args:
        name: The fully-qualified module name to import.

    Example:
        ```pycon
        >>> from coola.utils.imports import LazyModule
        >>> # Lazy version of import numpy as np
        >>> np = LazyModule("numpy")
        >>> # The module is imported the first time it is actually used.
        >>> np.ones((2, 3))
        array([[1., 1., 1.],
               [1., 1., 1.]])

        ```
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._module: object = None

    def __getattr__(self, name: str) -> Any:
        # Special names are looked up by the interpreter and by libraries
        # (``hasattr``, ``str``, ``inspect``...) on the proxy itself: they
        # must not trigger the import. ``_module`` is missing on an object
        # created without ``__init__``.
        if name == "_module" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        if self._module is None:
            self._import_module()
        return getattr(self._module, name)

    def __dir__(self) -> list[str]:
        if self._module is None:
            self._import_module()
        return dir(self._module)

    def _import_module(self) -> None:
        self._module = importlib.import_module(self.__name__)

        # Update this object's dict so that attribute references are efficient
        # (__getattr__ is only called on lookups that fail)
        self.__dict__.update(self._module.__dict__)


def lazy_import(name: str) -> LazyModule:
    r"""Return a proxy of the module/package to lazily import.

    Args:
        name: The fully-qualified module name to import.

    Returns:
        A proxy module that lazily imports a module the first time
            it is actually used.

    Example:
        ```pycon
        >>> from coola.utils.imports import lazy_import
        >>> # Lazy version of import numpy as np
        >>> np = lazy_import("numpy")
        >>> # The module is imported the first time it is actually used.
        >>> np.ones((2, 3))
        array([[1., 1., 1.],
               [1., 1., 1.]])

        ```
    """
    return LazyModule(name)


def import_object(path: str) -> Any:
    r"""Import an object from a module.

    Args:
        path: The path of the object, with the form
            ``"package.module:name"``.

    Returns:
        The object, which is not imported until this function is called.

    Raises:
        ValueError: If ``path`` does not have the form
            ``"package.module:name"``.

    Example:
        ```pycon
        >>> from coola.utils.imports import import_object
        >>> import_object("collections:OrderedDict")
        <class 'collections.OrderedDict'>

        ```
    """
    module_name, separator, name = path.partition(":")
    if not (module_name and separator and name):
        msg = f"Expected a path with the form 'package.module:name', got {path!r}"
        raise ValueError(msg)
    return getattr(importlib.import_module(module_name), name)


def is_module_imported(target: object) -> bool:
    r"""Indicate if the module of a target is already imported.

    The module is not imported. A target that is not a string is
    reported as imported, so ``coola.factory.import_object`` raises its
    usual error.

    Args:
        target: The fully qualified name of the object (for example
            ``"collections.Counter"``), or a module name.

    Returns:
        ``True`` if the module is in ``sys.modules`` or if ``target``
            is not a string, otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_module_imported
        >>> is_module_imported("collections.Counter")
        True
        >>> is_module_imported("missing_module.Name")
        False

        ```
    """
    if not isinstance(target, str):
        return True
    module_name = target.rpartition(".")[0] or target
    return module_name in sys.modules


def lazy_exports(
    package: str, exports: Mapping[str, str], namespace: dict[str, Any]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    r"""Create the ``__getattr__`` and ``__dir__`` of a package that
    imports some of its names when they are used.

    The names are in the ``__all__`` of the package, so ``from package
    import name`` works as usual, but the module that defines a name is
    imported the first time the name is used. This avoids importing the
    optional libraries that a module needs when the package is
    imported.

    Args:
        package: The name of the package, usually ``__name__``.
        exports: The module that defines each lazy name.
        namespace: The namespace of the package, usually ``globals()``.
            A name is added to it when it is imported, so the next
            access does not call ``__getattr__``.

    Returns:
        The ``__getattr__`` and ``__dir__`` functions of the package.
        The returned ``__getattr__`` raises an ``AttributeError`` if the
        name is not in ``exports``.

    Example:
        ```pycon
        >>> namespace = {}
        >>> getattr_, dir_ = lazy_exports("example", {"OrderedDict": "collections"}, namespace)
        >>> getattr_("OrderedDict")
        <class 'collections.OrderedDict'>
        >>> "OrderedDict" in namespace
        True

        ```
    """

    def __getattr__(name: str) -> Any:  # noqa: N807
        module_name = exports.get(name)
        if module_name is None:
            msg = f"module {package!r} has no attribute {name!r}"
            raise AttributeError(msg)
        value = getattr(importlib.import_module(module_name), name)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:  # noqa: N807
        return sorted({*namespace, *exports})

    return __getattr__, __dir__
