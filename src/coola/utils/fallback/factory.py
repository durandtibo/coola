r"""Shared helpers to build the fake classes/functions used by the per-
backend fallback modules.

Each optional-dependency fallback module (``numpy``, ``torch``,
``pandas``, ``polars``, ``jax``, ``xarray``, ``pyarrow``, ``pydantic``)
needs a ``FakeClass``/fake function that raises the backend's "missing
dependency" error on use. This module factors out that repeated
boilerplate so each fallback module only has to describe its own fake
package shape.
"""

from __future__ import annotations

__all__ = ["make_fake_class", "make_fake_function", "make_fake_module"]

from types import ModuleType
from typing import TYPE_CHECKING, Any, Never

from coola.utils.imports import raise_package_missing_error

if TYPE_CHECKING:
    from collections.abc import Callable


def make_fake_class(package: str) -> type:
    r"""Create a fake class whose constructor raises the "missing
    dependency" error of a package.

    Args:
        package: The name of the missing optional dependency, e.g.
            ``"numpy"``.

    Returns:
        A class that can be used as a stand-in for a real backend type
        (e.g. ``numpy.ndarray``) when the backend is not installed.
        Instantiating it raises the given error.
    """

    def init(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise_package_missing_error(package)

    return type(
        "FakeClass",
        (),
        {
            "__doc__": (
                "Fake class that raises an error because a required dependency is not installed."
            ),
            "__init__": init,
        },
    )


def make_fake_function(package: str) -> Callable[..., Never]:
    r"""Create a fake function that raises the "missing dependency" error
    of a package when called.

    Args:
        package: The name of the missing optional dependency, e.g.
            ``"torch"``.

    Returns:
        A function that can be used as a stand-in for a real backend
        function (e.g. ``torch.tensor``) when the backend is not
        installed. Calling it raises the given error.
    """

    def fake_function(*args: Any, **kwargs: Any) -> Never:  # noqa: ARG001
        raise_package_missing_error(package)

    return fake_function


def make_fake_module(name: str, **attributes: Any) -> ModuleType:
    r"""Create a fake module exposing the given attributes.

    Args:
        name: The fully qualified name of the module
            (e.g. ``"torch.nn"``).
        **attributes: The attributes to expose on the module.

    Returns:
        The fake module.

    Example:
        ```pycon
        >>> from coola.utils.fallback.factory import make_fake_module
        >>> module = make_fake_module("pkg", value=1)
        >>> module.value
        1

        ```
    """
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module
