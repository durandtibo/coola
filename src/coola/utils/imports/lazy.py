r"""Implement some utility functions to lazily import a module."""

from __future__ import annotations

__all__ = ["is_module_imported", "lazy_exports"]

import importlib
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


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
        r"""Import a lazy name the first time it is used.

        The value is stored in the namespace of the package, so this
        function is not called again for this name.

        Args:
            name: The name of the attribute.

        Returns:
            The value of the attribute.

        Raises:
            AttributeError: If ``name`` is not one of the lazy names.
        """
        module_name = exports.get(name)
        if module_name is None:
            msg = f"module {package!r} has no attribute {name!r}"
            raise AttributeError(msg)
        value = getattr(importlib.import_module(module_name), name)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:  # noqa: N807
        r"""List the names of the package.

        Returns:
            The sorted names of the namespace and of the lazy names.
        """
        return sorted({*namespace, *exports})

    return __getattr__, __dir__
