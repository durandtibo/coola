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

__all__ = ["make_fake_class", "make_fake_function"]

from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Callable


def make_fake_class(raise_error: Callable[[], NoReturn]) -> type:
    r"""Create a fake class whose constructor raises the given "missing
    dependency" error.

    Args:
        raise_error: A no-argument callable that raises the
            ``RuntimeError`` for the missing optional dependency, e.g.
            ``coola.utils.imports.raise_numpy_missing_error``.

    Returns:
        A class that can be used as a stand-in for a real backend type
        (e.g. ``numpy.ndarray``) when the backend is not installed.
        Instantiating it raises the given error.
    """

    def init(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise_error()

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


def make_fake_function(raise_error: Callable[[], NoReturn]) -> Callable[..., NoReturn]:
    r"""Create a fake function that raises the given "missing dependency"
    error when called.

    Args:
        raise_error: A no-argument callable that raises the
            ``RuntimeError`` for the missing optional dependency.

    Returns:
        A function that can be used as a stand-in for a real backend
        function (e.g. ``torch.tensor``) when the backend is not
        installed. Calling it raises the given error.
    """

    def fake_function(*args: Any, **kwargs: Any) -> NoReturn:  # noqa: ARG001
        raise_error()

    return fake_function
