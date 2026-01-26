r"""Implement some utility functions to lazily import a module."""

from __future__ import annotations

__all__ = ["LazyModule", "lazy_import"]

import importlib
from types import ModuleType
from typing import Any


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

    def __getattr__(self, item: str) -> Any:
        if self._module is None:
            self._import_module()
        return getattr(self._module, item)

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
